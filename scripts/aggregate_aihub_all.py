#!/usr/bin/env python3
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


AIHUB_SUBSETS = ("D01", "D02", "D03", "D04")
RATE_METRICS = ("wer", "cer", "mer", "jer", "ser", "rtf", "latency")
EDIT_METRICS = ("wer", "cer", "mer", "jer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/full")
    parser.add_argument("--output_dir", default="results/aggregated")
    args = parser.parse_args()

    groups = _load_aihub_domain_runs(Path(args.results_dir))
    output_dir = Path(args.output_dir)
    written = 0
    for model_key, runs in sorted(groups.items()):
        subsets = {run["row"].get("subset") for run in runs}
        if subsets != set(AIHUB_SUBSETS):
            missing = sorted(set(AIHUB_SUBSETS) - subsets)
            print(f"Skipping {model_key}: missing {missing}")
            continue
        row = _aggregate_runs(model_key, runs)
        run_dir = output_dir / row["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "leaderboard_row.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "summary.json").write_text(
            json.dumps({"leaderboard_row": row}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written += 1
        print(f"Wrote {run_dir / 'leaderboard_row.json'}")
    print(f"Wrote {written} aggregate AIHub all row(s).")


def _load_aihub_domain_runs(results_dir):
    groups = {}
    for row_path in results_dir.glob("**/leaderboard_row.json"):
        try:
            row = json.loads(row_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("dataset") != "AIHubLowQualityTelephone":
            continue
        if row.get("subset") not in AIHUB_SUBSETS:
            continue
        if row.get("is_full_evaluation") is not True:
            continue
        sample_path = row_path.parent / "samples.jsonl"
        if not sample_path.is_file():
            continue
        key = (row.get("model"), row.get("model_repo"))
        groups.setdefault(key, []).append(
            {"row": row, "row_path": row_path, "sample_path": sample_path}
        )
    return groups


def _aggregate_runs(model_key, runs):
    rows = sorted(runs, key=lambda run: run["row"].get("subset"))
    samples = []
    for run in rows:
        samples.extend(_read_samples(run["sample_path"]))

    valid_samples = [sample for sample in samples if not sample.get("is_outlier")]
    base = rows[0]["row"]
    model, model_repo = model_key
    run_id = f"aggregate-aihub-all-{_slug(model)}-{_utc_stamp()}"
    command = (
        "python -m openkoasr.main "
        "--dataset_name AIHubLowQualityTelephone "
        "--dataset_rootpath $AIHUB_TELEPHONE_ROOT "
        "--dataset_subset all "
        f"--model_name {model} "
        "--normalization_preset kspon "
        "--output_dir results/full"
    )
    return {
        "run_id": run_id,
        "model": model,
        "model_repo": model_repo,
        "dataset": "AIHubLowQualityTelephone",
        "subset": "all",
        "gpu": base.get("gpu"),
        "torch": base.get("torch"),
        "cuda": base.get("cuda"),
        "metrics": {
            "macro": _macro_average(valid_samples),
            "micro": _micro_average(valid_samples),
            "latency_percentiles": _latency_percentiles(
                [sample.get("processing_time") for sample in valid_samples]
            ),
        },
        "model_metrics": base.get("model_metrics", {}),
        "total_samples": len(samples),
        "dataset_total_samples": sum(int(run["row"].get("dataset_total_samples", 0)) for run in rows),
        "evaluated_samples": sum(int(run["row"].get("evaluated_samples", 0)) for run in rows),
        "is_full_evaluation": True,
        "valid_samples": len(valid_samples),
        "outlier_count": len(samples) - len(valid_samples),
        "outlier_policy": base.get("outlier_policy", {"metric": "cer", "threshold": 1.0}),
        "command": command,
        "source": "aggregated from full AIHub D01-D04 runs",
        "aggregation": {
            "subsets": list(AIHUB_SUBSETS),
            "source_run_ids": [run["row"].get("run_id") for run in rows],
        },
    }


def _read_samples(path):
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _macro_average(samples):
    totals = {}
    counts = {}
    for sample in samples:
        metrics = sample.get("metrics", {})
        for metric in RATE_METRICS:
            value = metrics.get(metric)
            if _is_finite_number(value):
                totals[metric] = totals.get(metric, 0.0) + float(value)
                counts[metric] = counts.get(metric, 0) + 1
    return {
        metric: totals[metric] / counts[metric]
        for metric in sorted(totals)
        if counts.get(metric)
    }


def _micro_average(samples):
    averages = {}
    for metric in EDIT_METRICS:
        substitutions = _sum_metric(samples, f"{metric}_substitutions")
        deletions = _sum_metric(samples, f"{metric}_deletions")
        insertions = _sum_metric(samples, f"{metric}_insertions")
        hits = _sum_metric(samples, f"{metric}_hits")
        denominator = hits + substitutions + deletions
        if denominator > 0:
            averages[metric] = (substitutions + deletions + insertions) / denominator

    ser_errors = _sum_metric(samples, "ser_error_sentences")
    ser_total = _sum_metric(samples, "ser_total_sentences")
    if ser_total > 0:
        averages["ser"] = ser_errors / ser_total
    return averages


def _latency_percentiles(values):
    values = [float(value) for value in values if _is_finite_number(value)]
    if not values:
        return {}
    return {
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _sum_metric(samples, metric):
    return sum(float(sample.get("metrics", {}).get(metric, 0.0)) for sample in samples)


def _percentile(values, percentile):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _is_finite_number(value):
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def _slug(value):
    return str(value).replace("/", "-").replace("_", "-")


def _utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    main()
