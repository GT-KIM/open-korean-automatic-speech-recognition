#!/usr/bin/env python3
"""Validate private Nemotron artifacts and emit a report without transcripts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openkoasr.evaluation.results import AggregateResult, SampleResult, OutlierPolicy
from openkoasr.evaluation.speed import GROUPS, summarize_requests
from openkoasr.metrics.character_error_rate import character_error_rate
from openkoasr.normalization import normalize_text
from scripts.aggregate_aihub_all import _aggregate_runs
from scripts.run_nemotron_benchmark import MODEL, MANIFESTS, write_json
from scripts.validate_full_accuracy import public_numbers, same


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def check(value, message):
    if not value:
        raise ValueError(message)


def check_audit(audit, count):
    check(audit["sequences"] == len(audit["rows"]) == count, "Audit cardinality mismatch")
    check(audit["all_encoder_frames_consumed"], "Unconsumed encoder frames")
    for row in audit["rows"]:
        check(row["reason"] == "encoder_exhausted" and row["prompt_id"] == 14, "Wrong RNNT termination/prompt")
        check(row["encoder_frames"] == row["consumed_frames"] > 0, "Wrong RNNT frame count")
        check(0 <= row["forced_frame_advances"] <= row["encoder_frames"], "Invalid frame guard count")


def check_identity(rep, protocol):
    check(rep["model_revision"] == rep["processor_revision"] == protocol["revision"], "Revision mismatch")
    check(rep["code"]["source_sha256"] == protocol["runtime"]["source_sha256"], "Source mismatch")
    check(rep["inference"]["effective_dtype"] == "bfloat16", "Dtype mismatch")
    options = rep["inference"]["decoding"]["transcription_options"]
    check(options == {"language": "ko-KR", "mode": "offline_full_utterance",
                       "num_lookahead_tokens": 3, "max_symbols_per_step": 10}, "Transcription options mismatch")


def validate_speed(root, manifest, protocol):
    check(sha(manifest) == MANIFESTS["formal"][1], "Speed manifest changed")
    inputs = read_lines(manifest)
    ordered = ["/".join((r["selection"]["dataset"], r["selection"]["subset"], r["selection"]["sample_id"])) for r in inputs]
    check(len(ordered) == len(set(ordered)) == 768, "Speed input identities invalid")
    tracks = {}
    for batch in (1, 4):
        folder = root / f"speed-b{batch}"
        summary = json.loads((folder / "summary.json").read_bytes())
        meta = json.loads((folder / "metadata.json").read_bytes())
        requests, warmups = read_lines(folder / "requests.jsonl"), read_lines(folder / "warmup.jsonl")
        check(json.loads((folder / "status.json").read_bytes())["status"] == "passed", "Speed run not passed")
        check_identity(meta["reproducibility"], protocol)
        check(meta["environment"]["image_id"] == protocol["runtime"]["image_id"], "Speed image mismatch")
        check(meta["harness_sha256"] == protocol["runtime"]["harness_sha256"]["run_nemotron_benchmark.py"], "Speed harness mismatch")
        packages_hash = hashlib.sha256(json.dumps(meta["environment"]["packages"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        check(packages_hash == protocol["runtime"]["package_inventory_sha256"], "Speed packages mismatch")
        check(meta["execution"]["input_manifest_sha256"] == sha(manifest), "Speed metadata input mismatch")
        check(len(requests) == 2304 // batch and len(warmups) == 144, "Incomplete speed run")
        for repeat in range(1, 4):
            rows = [r for r in requests if r["repeat"] == repeat]
            check([s["id"] for r in rows for s in r["samples"]] == ordered, "Speed input order mismatch")
            for offset, request in enumerate(rows):
                check(request["batch_size"] == len(request["samples"]) == batch, "Speed batch mismatch")
                check(request["group"] == GROUPS[(offset * batch) // 256], "Speed group mismatch")
                check(request["batch_index"] == offset % (256 // batch), "Speed batch boundary mismatch")
                check(math.isfinite(request["seconds"]) and request["seconds"] > 0, "Invalid timing")
                expected_inputs = inputs[offset * batch:(offset + 1) * batch]
                check(math.isclose(request["audio_seconds"], sum(r["frames"] / 16000 for r in expected_inputs), abs_tol=1e-10), "Speed audio duration mismatch")
                check_audit(request["generation_audit"], batch)
                for sample, expected in zip(request["samples"], expected_inputs):
                    check(sample["reference"] == expected["text"] and isinstance(sample["prediction"], str), "Invalid speed transcript/reference")
                    cer = character_error_rate(normalize_text(sample["prediction"], preset="kspon"), normalize_text(sample["reference"], preset="kspon"))["cer"]
                    same(cer if math.isfinite(cer) else None, sample["diagnostic_cer"])
        expected_warmup = [(repeat, group, sample_id) for repeat in range(1, 4)
                           for index, group in enumerate(GROUPS) for sample_id in ordered[index * 256:index * 256 + 16]]
        check([(r["repeat"], r["group"], r["id"]) for r in warmups] == expected_warmup, "Warmup order mismatch")
        for row in warmups:
            check_audit(row["generation_audit"], 1)
        recomputed = summarize_requests(requests, batch)
        for key, value in recomputed.items():
            same(value, summary[key])
        forced = sum(s["forced_frame_advances"] for r in requests for s in r["generation_audit"]["rows"])
        check(summary["forced_frame_advances"] == forced and summary["samples_measured"] == 2304, "Speed summary counts mismatch")
        tracks[f"b{batch}"] = {**recomputed, "measured_samples": 2304, "warmups": 144,
                               "forced_frame_advances": forced, "peak_allocated_bytes": summary["peak_allocated_bytes"],
                               "requests_sha256": sha(folder / "requests.jsonl"), "metadata_sha256": sha(folder / "metadata.json")}
    return tracks


def validate_accuracy(root, inputs, protocol):
    inventory_path = inputs / "inventory.json"
    check(sha(inventory_path) == "83b29a2264edaa4d9ad6b8e11d4458656d87d4fa23b8d2e0edf21f7dcb275526", "Accuracy inventory changed")
    inventory = json.loads(inventory_path.read_bytes())
    rows, proofs, phone = [], [], []
    for entry in inventory["datasets"]:
        folder = root / "accuracy" / entry["id"]
        completion = json.loads((folder / "completion.json").read_bytes())
        run = folder / completion["run_id"]
        summary_path = run / "summary.json"
        summary = json.loads(summary_path.read_bytes())
        row_path = run / "leaderboard_row.json"
        row = json.loads(row_path.read_bytes())
        meta = summary["metadata"]
        rep = meta["reproducibility"]
        check_identity(rep, protocol)
        evidence = rep["nemotron_accuracy"]
        check(evidence["image_id"] == protocol["runtime"]["image_id"], "Accuracy image mismatch")
        check(evidence["harness_sha256"] == protocol["runtime"]["harness_sha256"]["run_nemotron_accuracy.py"], "Accuracy harness mismatch")
        packages_hash = hashlib.sha256(json.dumps(evidence["packages"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        check(packages_hash == protocol["runtime"]["package_inventory_sha256"], "Accuracy packages mismatch")
        check(sha(summary_path) == completion["summary_sha256"] and completion["status"] == "passed", "Accuracy summary/completion mismatch")
        check(meta["limit"] is None and meta["is_full_evaluation"] and meta["batch_size"] == 4, "Not full B4 accuracy")
        check(meta["evaluation_protocol"] == "v1/kspon/cer>1.0", "Scoring protocol changed")
        check(row["reproducibility"] == rep and row["model_repo"] == meta["model_repo"] == MODEL, "Row model/provenance mismatch")
        check(row["dataset"] == meta["dataset_name"] == entry["dataset"] and row["subset"] == meta["dataset_subset"] == entry["subset"], "Dataset metadata mismatch")
        manifest = inputs / entry["manifest"]
        check(sha(manifest) == entry["sha256"] == evidence["input_manifest_sha256"], "Accuracy manifest changed")
        expected = read_lines(manifest)
        journal, saved = read_lines(folder / "journal.jsonl"), read_lines(run / "samples.jsonl")
        check(len(expected) == len(journal) == len(saved) == entry["samples"], "Accuracy sample count mismatch")
        check(public_numbers(journal) == saved, "Journal differs from saved outputs")
        forced = 0
        for index, (sample, source) in enumerate(zip(journal, expected)):
            check(sample["index"] == source["index"] == index and sample["sample_id"] == source["sample_id"], "Accuracy input order mismatch")
            check(hashlib.sha256(sample["reference"].encode()).hexdigest() == source["reference_sha256"], "Reference changed")
            check(sample["sample_rate"] == source["sample_rate"] == 16000, "Sample rate mismatch")
            check(math.isclose(sample["audio_duration"], source["frames"] / 16000, abs_tol=1e-12), "Audio duration mismatch")
            for raw, normalized in (("reference", "normalized_reference"), ("prediction", "normalized_prediction")):
                check(normalize_text(sample[raw], preset="kspon") == sample[normalized], "Normalization mismatch")
            cer = character_error_rate(sample["normalized_prediction"], sample["normalized_reference"])
            for key, value in cer.items():
                same(public_numbers(value), public_numbers(sample["metrics"][key]))
            check(sample["is_outlier"] == OutlierPolicy("cer", 1.0).is_outlier(sample["metrics"]), "Outlier policy mismatch")
            check(math.isfinite(sample["processing_time"]) and sample["processing_time"] > 0, "Invalid accuracy timing")
            detail = sample["metadata"]["rnnt_generation"]
            check_audit({"sequences": 1, "rows": [detail], "all_encoder_frames_consumed": True}, 1)
            forced += detail["forced_frame_advances"]
        warmups = read_lines(folder / "warmup.jsonl")
        check(len(warmups) == 16, "Accuracy warmup count mismatch")
        for audit in warmups:
            check_audit(audit, 1)
        counts = {"measured": entry["samples"], "warmup": 16, "measured_forced_frame_advances": forced,
                  "warmup_forced_frame_advances": sum(s["forced_frame_advances"] for a in warmups for s in a["rows"])}
        check(counts == evidence["termination"] == completion["termination"], "Accuracy termination summary mismatch")
        aggregate = AggregateResult.from_samples([SampleResult(**s) for s in journal]).to_dict()
        same(public_numbers(aggregate), summary["aggregate"])
        same(public_numbers(row["metrics"]["macro"]), public_numbers(aggregate["macro_average"]))
        same(public_numbers(row["metrics"]["micro"]), public_numbers(aggregate["micro_average"]))
        same(public_numbers(row["metrics"]["all_samples_micro"]), public_numbers(aggregate["all_samples_micro_average"]))
        row["_artifact"] = "results/nemotron-20261010/accuracy/" + entry["id"] + "/" + completion["run_id"] + "/leaderboard_row.json"
        row["source"] = "verified Nemotron full evaluation (native Linux, BF16, B4; unpublished)"
        rows.append(row)
        proofs.append({"dataset": entry["id"], "samples": len(journal), "forced_frame_advances": forced,
                       "summary_sha256": sha(summary_path), "samples_sha256": sha(run / "samples.jsonl")})
        if entry["dataset"] == "AIHubLowQualityTelephone":
            phone.append({"row": row, "row_path": row_path, "sample_path": run / "samples.jsonl"})
    check(sum(r["samples"] for r in proofs) == 45916, "Incomplete full corpus")
    rows.append(_aggregate_runs((MODEL, MODEL), phone))
    return rows, proofs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--speed-manifest", type=Path, required=True)
    p.add_argument("--accuracy-inputs", type=Path)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--speed-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_bytes())
    report = {"status": "speed_verified" if args.speed_only else "completed_and_verified",
              "validated_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": sha(args.protocol), "validator_sha256": sha(Path(__file__)),
              "official_leaderboard_eligible": False, "published": False,
              "speed": validate_speed(args.root, args.speed_manifest, protocol)}
    if not args.speed_only:
        report["accuracy_rows"], report["accuracy_proofs"] = validate_accuracy(args.root, args.accuracy_inputs, protocol)
    write_json(args.output, public_numbers(report))
    print(report["status"])


if __name__ == "__main__":
    main()
