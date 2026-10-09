#!/usr/bin/env python3
"""Run one pinned model/track against sealed pilot or formal speed inputs."""
import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--track", choices=("latency-b1", "throughput-b4"), required=True)
    p.add_argument("--purpose", choices=("pilot", "formal"), default="pilot")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "started", "purpose": args.purpose,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "model": args.model, "track": args.track,
                "environment_id": os.environ.get("ASR_ENVIRONMENT_ID")}
    write_json(args.output / "status.json", metadata)
    audit = None
    try:
        import torch
        from openkoasr.configs import get_model_config
        from openkoasr.evaluation.provenance import capture_reproducibility, code_metadata
        from openkoasr.evaluation.speed import (GenerationAudit, digest, load_inputs, measure_workload,
                                               summarize_requests, summarize_termination, termination_policy)
        from openkoasr.metrics.character_error_rate import character_error_rate
        from openkoasr.model import ModelFactory
        from openkoasr.normalization import normalize_text
        from scripts.check_benchmark_environment import package_inventory

        raw_protocol = args.protocol.read_bytes()
        protocol = json.loads(raw_protocol)
        preflight = json.loads(args.preflight.read_bytes())
        if preflight["status"] != "passed" or preflight["failures"]:
            raise ValueError("A passing preflight is required")
        for key, variable in (("image_id", "ASR_IMAGE_ID"), ("environment_id", "ASR_ENVIRONMENT_ID")):
            if preflight[key] != os.environ.get(variable):
                raise ValueError(f"Preflight {key} does not match this container")
        if preflight["image_id"] != protocol["environment"]["image_id"]:
            raise ValueError("Image has not been sealed in the current protocol")
        if code_metadata()["source_sha256"] != preflight["code"]["source_sha256"]:
            raise ValueError("Code differs from preflight")
        if package_inventory() != preflight["packages"]:
            raise ValueError("Packages differ from preflight")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Exactly one CUDA GPU is required")
        gpu = preflight["checks"]["gpu"]
        if torch.get_num_threads() != gpu["torch_threads"] or torch.get_num_interop_threads() != gpu["torch_interop_threads"]:
            raise ValueError("Torch thread settings differ from preflight")
        smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
        if smi != preflight["nvidia_smi"]:
            raise ValueError("GPU/driver identity differs from preflight")
        contract = protocol["input_contract"]
        pilot = args.purpose == "pilot"
        policy = termination_policy(protocol)
        expected = contract["pilot"]["manifest_sha256"] if pilot else contract["manifest_sha256"]
        samples = load_inputs(args.inputs, "pilot_manifest.jsonl" if pilot else "manifest.jsonl",
                              expected, 192 if pilot else 768)
        track = next(t for t in protocol["tracks"] if t["id"] == args.track)
        entry = next(m for m in protocol["models"] if m["repo"] == args.model)
        batch_size = track["batch_size"]
        repeats = contract["pilot"]["repetitions"] if pilot else protocol["execution"]["repetitions"]
        config = get_model_config(args.model)
        for key in ("model_revision", "processor_revision", "dtype", "language", "max_new_tokens"):
            setattr(config, key, entry[key])
        for key, value in entry["call_overrides"].items():
            setattr(config, key, value)
        config.device = "cuda:0"
        if config.family == "qwen3_asr":
            config.max_inference_batch_size = track["qwen_backend_batch_size"]
        print(f"Loading {args.model}, {args.track}, {args.purpose}", flush=True)
        model = ModelFactory.load_model(config)
        backend = model.model if hasattr(model.model, "config") else model.model.model
        backend.eval()
        audit = GenerationAudit(backend, entry["max_new_tokens"], strict=not pilot and policy == "require_eos")
        execution = {**protocol["execution"], "batch_size": batch_size, "repetitions": repeats,
                     "samples_per_repeat": len(samples), "purpose": args.purpose,
                     "model_loads": 1, "input_manifest_sha256": expected, "termination_policy": policy}
        provenance = capture_reproducibility(model, config, execution, {})
        if provenance["model_revision"] != entry["model_revision"] or provenance["processor_revision"] != entry["processor_revision"]:
            raise ValueError("Loaded model/processor revision differs from protocol")
        if provenance["inference"]["effective_dtype"] != "bfloat16":
            raise ValueError("All model parameters must use BF16")
        metadata.update({"protocol_id": protocol["id"], "protocol_sha256": digest(raw_protocol),
                         "input_manifest_sha256": expected, "preflight_sha256": digest(args.preflight.read_bytes()),
                         "image_id": preflight["image_id"], "execution": execution,
                         "preflight": preflight, "reproducibility": provenance})
        write_json(args.output / "metadata.json", metadata)
        requests, warmups = [], []
        with (args.output / "requests.jsonl").open("x", encoding="utf-8") as stream, \
                (args.output / "warmup.jsonl").open("x", encoding="utf-8") as warmup_stream:
            def record_warmup(row):
                warmup_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                warmup_stream.flush()
                warmups.append(row)

            def record(request):
                rows = []
                for sample, prediction in zip(request.pop("samples"), request.pop("predictions")):
                    reference = normalize_text(sample["text"], preset="kspon")
                    normalized = normalize_text(prediction, preset="kspon")
                    cer = character_error_rate(normalized, reference)
                    rows.append({"id": sample["metadata"]["id"],
                                 "selection_order": sample["metadata"]["selection_order"],
                                 "reference": sample["text"], "prediction": prediction,
                                 "diagnostic_cer": cer["cer"] if math_isfinite(cer["cer"]) else None})
                request["samples"] = rows
                stream.write(json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                requests.append(request)
                if len(requests) % 16 == 0:
                    print(f"Measured {sum(r['batch_size'] for r in requests)}/{len(samples) * repeats} samples", flush=True)
            with torch.inference_mode():
                completed = measure_workload(model, samples, batch_size, repeats,
                                             protocol["execution"]["warmup_samples"], record,
                                             torch.cuda.synchronize, after_call=audit.check,
                                             on_warmup=record_warmup)
        if completed != len(samples) * repeats:
            raise ValueError("Incomplete workload")
        summary = summarize_requests(requests, batch_size)
        termination = summarize_termination(requests, warmups, policy)
        measured_missing = termination["measured"]["token_limit"]
        warmup_missing = termination["warmup"]["token_limit"]
        passed = termination["contract_passed"]
        summary.update({"status": "passed" if passed else "failed", "purpose": args.purpose, "model": args.model,
                        "track": args.track, "environment_id": metadata["environment_id"],
                        "samples_measured": completed, "model_loads": 1,
                        "warmup_samples_excluded": protocol["execution"]["warmup_samples"] * 3 * repeats,
                        "all_generations_reached_eos": termination["all_generations_reached_eos"],
                        "termination": termination,
                        "measured_sequences_without_eos": measured_missing,
                        "warmup_sequences_without_eos": warmup_missing,
                        "failures": [] if passed else ["generation_without_eos_requires_protocol_review"],
                        "official_leaderboard_eligible": False})
        write_json(args.output / "summary.json", summary)
        metadata["reproducibility"] = capture_reproducibility(model, config, execution, {})
        metadata["status"] = summary["status"]
        metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "metadata.json", metadata)
        write_json(args.output / "status.json", {k: metadata[k] for k in ("status", "purpose", "model", "track", "completed_at_utc")})
        print(f"{summary['status'].upper()} {args.model} {args.track}: {completed} measured samples, {measured_missing} measured and {warmup_missing} warmup sequences without EOS", flush=True)
        return int(not passed)
    except Exception as error:
        metadata.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        write_json(args.output / "status.json", metadata)
        traceback.print_exc()
        return 1
    finally:
        if audit is not None:
            audit.close()


def math_isfinite(value):
    import math
    return math.isfinite(value)


if __name__ == "__main__":
    raise SystemExit(main())
