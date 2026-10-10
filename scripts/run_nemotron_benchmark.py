#!/usr/bin/env python3
"""Bounded Nemotron pilot / warmed speed evaluation on the existing sealed inputs."""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MANIFESTS = {
    "pilot": ("pilot_manifest.jsonl", "f5a2173091f13876eb4d82d666e163942d43d019cecec23fc39524009b5fc7b9", 192, 1),
    "formal": ("manifest.jsonl", "feab69a04d1de01f00190bcacb1c81a4e4049b82e9b6ebc1510e8019956cbfe1", 768, 3),
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--purpose", choices=MANIFESTS, default="pilot")
    p.add_argument("--batch-size", type=int, choices=(1, 4), required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-source", required=True)
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "purpose": args.purpose, "batch_size": args.batch_size, "model": MODEL}
    write_json(args.output / "status.json", state)
    try:
        import torch
        torch.set_num_threads(8)
        torch.set_num_interop_threads(8)
        from openkoasr.configs import get_model_config
        from openkoasr.evaluation.provenance import capture_reproducibility, code_metadata
        from openkoasr.evaluation.speed import load_inputs, measure_workload, summarize_requests, digest
        from openkoasr.metrics.character_error_rate import character_error_rate
        from openkoasr.model import ModelFactory
        from openkoasr.normalization import normalize_text
        from scripts.check_benchmark_environment import package_inventory

        if metadata.version("transformers") != "5.13.0":
            raise ValueError("This protocol requires Transformers 5.13.0")
        if code_metadata()["source_sha256"] != args.expected_source:
            raise ValueError("Source hash differs from launch contract")
        if not os.environ.get("ASR_IMAGE_ID", "").startswith("sha256:"):
            raise ValueError("Launcher must record the immutable image ID")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Exactly one CUDA GPU is required")
        if torch.get_num_threads() != 8 or torch.get_num_interop_threads() != 8:
            raise ValueError("Expected 8 intra/inter-op threads")
        name, manifest_hash, count, repeats = MANIFESTS[args.purpose]
        samples = load_inputs(args.inputs, name, manifest_hash, count)
        config = get_model_config(MODEL)
        print(f"Loading {MODEL}: {count} samples B{args.batch_size}, {repeats} repeat(s)", flush=True)
        model = ModelFactory.load_model(config)
        if model.processor.prompt_dictionary["ko-KR"] != 14 or model.model.max_symbols_per_step != 10:
            raise ValueError("Korean prompt/decoder contract changed")
        execution = {"purpose": args.purpose, "batch_size": args.batch_size, "repetitions": repeats,
                     "samples_per_repeat": count, "warmup_samples": 16,
                     "warmup_mode": "single_sample_per_group_per_repeat", "model_loads": 1,
                     "input_manifest_sha256": manifest_hash, "termination_policy": "require_encoder_exhaustion"}
        environment = {"image_id": os.environ["ASR_IMAGE_ID"], "packages": package_inventory(),
                       "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                       "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()}
        provenance = capture_reproducibility(model, config, execution, environment)
        if provenance["model_revision"] != config.model_revision or provenance["inference"]["effective_dtype"] != config.dtype:
            raise ValueError("Loaded model differs from the declared revision/dtype")
        write_json(args.output / "metadata.json", {**state, "execution": execution, "environment": environment,
                    "reproducibility": provenance, "harness_sha256": digest(Path(__file__).read_bytes()),
                    "official_leaderboard_eligible": False, "mode": "offline_full_utterance"})
        requests, warmups = [], []
        torch.cuda.reset_peak_memory_stats()
        with (args.output / "requests.jsonl").open("x", encoding="utf-8") as stream, \
                (args.output / "warmup.jsonl").open("x", encoding="utf-8") as warm_stream:
            def record_warmup(row):
                warmups.append(row)
                warm_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                warm_stream.flush()

            def record(row):
                predictions = row.pop("predictions")
                original = row.pop("samples")
                row["samples"] = []
                for sample, prediction in zip(original, predictions):
                    cer = character_error_rate(normalize_text(prediction, preset="kspon"),
                                               normalize_text(sample["text"], preset="kspon"))["cer"]
                    import math
                    row["samples"].append({"id": sample["metadata"]["id"],
                                           "reference": sample["text"], "prediction": prediction,
                                           "diagnostic_cer": cer if math.isfinite(cer) else None})
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                requests.append(row)
                if len(requests) % 16 == 0:
                    print(f"Measured {sum(r['batch_size'] for r in requests)}/{count * repeats}", flush=True)

            completed = measure_workload(model, samples, args.batch_size, repeats, 16, record,
                                         torch.cuda.synchronize, after_call=model.audit_last_generation,
                                         on_warmup=record_warmup)
        if completed != count * repeats or len(warmups) != 48 * repeats:
            raise ValueError("Incomplete workload")
        for row in requests + warmups:
            expected = row.get("batch_size", 1)
            if row["generation_audit"]["sequences"] != expected:
                raise ValueError("Audit/output cardinality mismatch")
        if code_metadata()["source_sha256"] != args.expected_source:
            raise ValueError("Source changed while running")
        summary = summarize_requests(requests, args.batch_size)
        summary.update({"status": "passed", "purpose": args.purpose, "samples_measured": completed,
                        "warmup_samples_excluded": len(warmups), "all_encoder_frames_consumed": True,
                        "forced_frame_advances": sum(s["forced_frame_advances"] for r in requests
                                                     for s in r["generation_audit"]["rows"]),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "official_leaderboard_eligible": False})
        write_json(args.output / "summary.json", summary)
        state.update(status="passed", completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output / "status.json", state)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(args.output / "status.json", state)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
