#!/usr/bin/env python3
import argparse
import json
import math
import re
import sys
from pathlib import Path

if __package__:
    from .leaderboard_metadata import LEGACY_RUN_IDS
else:
    from leaderboard_metadata import LEGACY_RUN_IDS
from openkoasr.protocol import evaluation_protocol_id


REQUIRED_FIELDS = {
    "run_id",
    "model",
    "model_repo",
    "dataset",
    "metrics",
    "total_samples",
    "dataset_total_samples",
    "evaluated_samples",
    "is_full_evaluation",
    "outlier_count",
    "outlier_policy",
    "normalization_preset",
    "evaluation_protocol",
    "metadata_status",
    "reproducibility",
}

REQUIRED_MACRO_METRICS = ("wer", "cer", "mer", "jer")
OPTIONAL_MACRO_METRICS = ("ser", "rtfx", "latency")
LOCAL_PATH_PATTERNS = (
    re.compile(r"C:\\Users\\", re.IGNORECASE),
    re.compile(r"/mnt/[a-z]/", re.IGNORECASE),
    re.compile("Pycharm" + "Projects", re.IGNORECASE),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        default=["doc/submitted_results.json", "doc/leaderboard_data.json"],
        help="Leaderboard JSON files to validate.",
    )
    args = parser.parse_args()

    problems = []
    for path in args.paths:
        validate_file(Path(path), problems)

    if problems:
        print("Leaderboard validation failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("Leaderboard validation passed.")
    return 0


def validate_file(path, problems):
    if not path.is_file():
        problems.append(f"{path}: file does not exist")
        return
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        problems.append(f"{path}: invalid JSON: {error}")
        return
    if not isinstance(rows, list):
        problems.append(f"{path}: expected a JSON list")
        return
    validate_rows(path, rows, problems)


def validate_rows(path, rows, problems):
    seen_runs = set()
    seen_slices = set()
    for index, row in enumerate(rows):
        validate_row(path, index, row, problems)
        if not isinstance(row, dict):
            continue
        run_id = row.get("run_id")
        if isinstance(run_id, str):
            if run_id in seen_runs:
                problems.append(f"{path}[{index}]: duplicate run_id: {run_id}")
            seen_runs.add(run_id)
        key = (row.get("model"), row.get("dataset"), row.get("subset"), row.get("evaluation_protocol"))
        if all(value is None or isinstance(value, str) for value in key):
            if key in seen_slices:
                problems.append(f"{path}[{index}]: duplicate model/dataset/subset/protocol: {key}")
            seen_slices.add(key)


def validate_row(path, index, row, problems):
    label = f"{path}[{index}]"
    if not isinstance(row, dict):
        problems.append(f"{label}: expected object")
        return

    missing = sorted(REQUIRED_FIELDS - set(row))
    if missing:
        problems.append(f"{label}: missing fields: {', '.join(missing)}")

    for field in ("run_id", "model", "model_repo", "dataset"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            problems.append(f"{label}: {field} must be a nonempty string")
    for field in ("subset", "normalization_preset"):
        if row.get(field) is not None and (
            not isinstance(row[field], str) or not row[field].strip()
        ):
            problems.append(f"{label}: {field} must be a nonempty string when provided")

    if row.get("model") == "mock" or row.get("dataset") == "mock":
        problems.append(f"{label}: mock rows must not be published")

    if row.get("is_full_evaluation") is not True:
        problems.append(f"{label}: is_full_evaluation must be true")

    counts = {}
    for field in ("total_samples", "evaluated_samples", "dataset_total_samples",
                  "outlier_count", "valid_samples"):
        if field == "valid_samples" and field not in row:
            continue
        value = row.get(field)
        minimum = 0 if field in {"outlier_count", "valid_samples"} else 1
        if type(value) is not int or value < minimum:
            problems.append(f"{label}: {field} must be an integer >= {minimum}")
        else:
            counts[field] = value
    for field in ("total_samples", "dataset_total_samples"):
        if field in counts and "evaluated_samples" in counts:
            if counts[field] != counts["evaluated_samples"]:
                problems.append(f"{label}: {field} must equal evaluated_samples")
    if "outlier_count" in counts and "total_samples" in counts:
        if counts["outlier_count"] > counts["total_samples"]:
            problems.append(f"{label}: outlier_count must not exceed total_samples")
        if "valid_samples" in counts:
            if counts["valid_samples"] + counts["outlier_count"] != counts["total_samples"]:
                problems.append(f"{label}: valid_samples + outlier_count must equal total_samples")

    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        problems.append(f"{label}: metrics must be an object")
        metrics = {}
        macro = {}
    else:
        macro = metrics.get("macro")

    if not isinstance(macro, dict):
        problems.append(f"{label}: metrics.macro must be an object")
        macro = {}

    for metric in REQUIRED_MACRO_METRICS:
        _validate_metric(f"{label}: metrics.macro.{metric}", metric, macro.get(metric), problems)

    for metric in OPTIONAL_MACRO_METRICS:
        value = macro.get(metric)
        if value is not None:
            _validate_metric(f"{label}: metrics.macro.{metric}", metric, value, problems)
    if "rtf" in macro:
        problems.append(f"{label}: metrics.macro.rtf is legacy; use metrics.macro.rtfx")

    for group in ("micro", "all_samples_micro"):
        micro = metrics.get(group, {})
        if not isinstance(micro, dict):
            problems.append(f"{label}: metrics.{group} must be an object")
        else:
            for metric, value in micro.items():
                _validate_metric(f"{label}: metrics.{group}.{metric}", metric, value, problems)

    percentiles = metrics.get("latency_percentiles", {})
    if not isinstance(percentiles, dict):
        problems.append(f"{label}: metrics.latency_percentiles must be an object")
    else:
        previous = 0.0
        for percentile in ("p50", "p90", "p95", "p99"):
            if percentile not in percentiles:
                continue
            value = _as_number(percentiles[percentile])
            if value is None or value < previous:
                problems.append(f"{label}: latency_percentiles.{percentile} must be finite, "
                                "nonnegative and nondecreasing")
            else:
                previous = value

    policy = row.get("outlier_policy")
    if not isinstance(policy, dict) or "metric" not in policy or "threshold" not in policy:
        problems.append(f"{label}: outlier_policy requires metric and threshold")
    else:
        if not isinstance(policy["metric"], str) or not policy["metric"].strip():
            problems.append(f"{label}: outlier_policy.metric must be a nonempty string")
        threshold = _as_number(policy["threshold"])
        if threshold is None or threshold < 0:
            problems.append(f"{label}: outlier_policy.threshold must be finite and nonnegative")

    command = row.get("command", "")
    if not isinstance(command, str):
        problems.append(f"{label}: command must be a string")
    elif command:
        for pattern in LOCAL_PATH_PATTERNS:
            if pattern.search(command):
                problems.append(f"{label}: command contains a local path")

    _validate_metadata(label, row, problems)


def _validate_metadata(label, row, problems):
    status = row.get("metadata_status")
    if status == "legacy":
        if (row.get("run_id") not in LEGACY_RUN_IDS or row.get("source") != "README legacy table"
                or row.get("normalization_preset") is not None
                or row.get("evaluation_protocol") is not None or row.get("reproducibility") != {}):
            problems.append(f"{label}: legacy exemption is limited to the existing unverified README rows")
        return
    try:
        expected = evaluation_protocol_id(row.get("normalization_preset"), row.get("outlier_policy"))
        if row.get("evaluation_protocol") != expected:
            problems.append(f"{label}: evaluation_protocol does not match normalization/outlier rules")
    except (ValueError, TypeError, OverflowError) as error:
        problems.append(f"{label}: {error}")
    _validate_reproducibility(label, status, row.get("reproducibility"), problems, row.get("run_id"))
    if status == "aggregate" and isinstance(row.get("reproducibility"), dict):
        sources = row.get("reproducibility", {}).get("source_runs", [])
        if isinstance(sources, list):
            ids = [item.get("run_id") for item in sources if isinstance(item, dict)]
            aggregation = row.get("aggregation")
            if not isinstance(aggregation, dict) or ids != aggregation.get("source_run_ids") or len(set(map(str, ids))) != 4:
                problems.append(f"{label}: aggregate provenance must match its four source runs")


def _validate_reproducibility(label, status, data, problems, run_id):
    if not isinstance(data, dict):
        problems.append(f"{label}: reproducibility must be an object")
        return
    if status == "aggregate":
        sources = data.get("source_runs")
        if not isinstance(sources, list) or len(sources) != 4:
            problems.append(f"{label}: aggregate requires four source_runs")
            return
        for source in sources:
            if not isinstance(source, dict) or source.get("metadata_status") not in ("recorded", "recovered"):
                problems.append(f"{label}: invalid aggregate source metadata")
                continue
            _validate_reproducibility(label, source["metadata_status"], source.get("reproducibility"), problems, source.get("run_id"))
        return
    if status not in ("recorded", "recovered"):
        problems.append(f"{label}: unknown metadata_status")
        return
    for field in ("code", "model_config", "execution", "environment"):
        if not isinstance(data.get(field), dict):
            problems.append(f"{label}: reproducibility.{field} must be an object")
    if "model_revision" not in data or (data["model_revision"] is not None and not isinstance(data["model_revision"], str)):
        problems.append(f"{label}: model_revision must be recorded or explicitly null")
    execution = data.get("execution", {})
    if isinstance(execution, dict):
        for key, minimum in (("batch_size", 1), ("num_workers", 0), ("warmup_samples", 0)):
            if type(execution.get(key)) is not int or execution[key] < minimum:
                problems.append(f"{label}: invalid execution.{key}")
        if "limit" not in execution or execution["limit"] is not None:
            problems.append(f"{label}: full evaluation execution.limit must be null")
    code = data.get("code", {})
    if isinstance(code, dict):
        for field, length in (("commit", 40), ("source_sha256", 64)):
            value = code.get(field)
            if field not in code or (value is not None and (not isinstance(value, str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", value))):
                problems.append(f"{label}: invalid code.{field}")
        if "dirty" not in code or (code["dirty"] is not None and type(code["dirty"]) is not bool):
            problems.append(f"{label}: code.dirty must be boolean or null")
        if status == "recorded" and code.get("source_sha256") is None:
            problems.append(f"{label}: new runs must record code.source_sha256")
    if status == "recovered":
        evidence = data.get("evidence", {})
        if not isinstance(evidence, dict) or not isinstance(run_id, str) or evidence.get("run_id") != run_id or not re.fullmatch("[0-9a-f]{64}", str(evidence.get("summary_sha256", ""))):
            problems.append(f"{label}: recovered metadata requires summary evidence")


def _validate_metric(label, metric, value, problems):
    number = _as_number(value)
    if number is None or number < 0:
        problems.append(f"{label} must be finite and nonnegative")
    elif metric == "ser" and number > 1:
        problems.append(f"{label} must not exceed 1")


def _as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except OverflowError:
            return None
        return number if math.isfinite(number) else None
    return None


if __name__ == "__main__":
    sys.exit(main())
