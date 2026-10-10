#!/usr/bin/env python3
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .validate_leaderboard_data import validate_row
    from .leaderboard_metadata import recover_run_metadata, aggregate_metadata
else:
    from validate_leaderboard_data import validate_row
    from leaderboard_metadata import recover_run_metadata, aggregate_metadata


AIHUB_SUBSETS = ("D01", "D02", "D03", "D04")
RATE_METRICS = ("wer", "cer", "mer", "jer", "ser", "rtfx", "latency")
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
        problems = []
        validate_row("AIHub aggregate", 0, row, problems)
        if problems:
            raise ValueError("\n".join(problems))
        run_dir = output_dir / row["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "leaderboard_row.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
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
        row = recover_run_metadata(row, row_path.parent / "summary.json")
        key = (row.get("model"), row.get("model_repo"))
        groups.setdefault(key, []).append(
            {"row": row, "row_path": row_path, "sample_path": sample_path}
        )
    return groups


def _aggregate_runs(model_key, runs):
    # Match leaderboard selection: newest run_id wins within each subset.
    # Break ties by artifact path so filesystem traversal order cannot affect results.
    latest_by_subset = {}
    for run in sorted(
        runs,
        key=lambda run: (str(run["row"].get("run_id", "")), run["row_path"].as_posix()),
    ):
        latest_by_subset[run["row"]["subset"]] = run
    rows = [latest_by_subset[subset] for subset in sorted(latest_by_subset)]
    if set(latest_by_subset) != set(AIHUB_SUBSETS):
        raise ValueError("AIHub aggregation requires D01-D04")
    base = rows[0]["row"]
    normalization = base.get("normalization_preset")
    if not isinstance(normalization, str) or not normalization.strip():
        raise ValueError(f"{rows[0]['row_path']}: normalization_preset is required")
    policy = base.get("outlier_policy")
    if not isinstance(policy, dict) or not {"metric", "threshold"} <= policy.keys():
        raise ValueError(f"{rows[0]['row_path']}: outlier_policy is required")
    samples = []
    for run in rows:
        row = run["row"]
        for field in ("normalization_preset", "outlier_policy"):
            if row.get(field) != base.get(field):
                raise ValueError(f"{run['row_path']}: inconsistent {field} across AIHub subsets")
        domain_samples = _read_samples(run["sample_path"])
        for field in ("total_samples", "evaluated_samples", "dataset_total_samples"):
            if type(row.get(field)) is not int or row[field] != len(domain_samples):
                raise ValueError(f"{run['row_path']}: {field} does not match samples.jsonl")
        sample_ids = set()
        for sample in domain_samples:
            sample_id = sample.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
                raise ValueError(f"{run['sample_path']}: missing or duplicate sample_id")
            sample_ids.add(sample_id)
            if type(sample.get("is_outlier")) is not bool:
                raise ValueError(f"{run['sample_path']}: is_outlier must be boolean")
        outliers = sum(sample["is_outlier"] for sample in domain_samples)
        if type(row.get("outlier_count")) is not int or row["outlier_count"] != outliers:
            raise ValueError(f"{run['row_path']}: outlier_count does not match samples.jsonl")
        if "valid_samples" in row and (
            type(row["valid_samples"]) is not int
            or row["valid_samples"] != len(domain_samples) - outliers
        ):
            raise ValueError(f"{run['row_path']}: valid_samples does not match samples.jsonl")
        samples.extend(domain_samples)

    valid_samples = [sample for sample in samples if not sample.get("is_outlier")]
    model, model_repo = model_key
    run_id = f"aggregate-aihub-all-{_slug(model)}-{_utc_stamp()}"
    command = (
        "python -m openkoasr.main "
        "--dataset_name AIHubLowQualityTelephone "
        "--dataset_rootpath $AIHUB_TELEPHONE_ROOT "
        "--dataset_subset all "
        f"--model_name {model} "
        f"--normalization_preset {normalization} "
        f"--outlier_metric {policy['metric']} --outlier_threshold {policy['threshold']} "
        "--output_dir results/full"
    )
    return {
        **aggregate_metadata([run["row"] for run in rows]),
        "run_id": run_id,
        "model": model,
        "model_repo": model_repo,
        "dataset": "AIHubLowQualityTelephone",
        "subset": "all",
        "normalization_preset": normalization,
        "gpu": base.get("gpu"),
        "torch": base.get("torch"),
        "cuda": base.get("cuda"),
        "metrics": {
            "macro": _macro_average(valid_samples),
            "micro": _micro_average(valid_samples),
            "all_samples_micro": _micro_average(samples),
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
        "outlier_policy": policy,
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
