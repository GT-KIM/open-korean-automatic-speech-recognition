#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path


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
}

REQUIRED_MACRO_METRICS = ("wer", "cer", "mer", "jer")
OPTIONAL_MACRO_METRICS = ("ser", "rtf", "latency")
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
    for index, row in enumerate(rows):
        validate_row(path, index, row, problems)


def validate_row(path, index, row, problems):
    label = f"{path}[{index}]"
    if not isinstance(row, dict):
        problems.append(f"{label}: expected object")
        return

    missing = sorted(REQUIRED_FIELDS - set(row))
    if missing:
        problems.append(f"{label}: missing fields: {', '.join(missing)}")

    if row.get("model") == "mock" or row.get("dataset") == "mock":
        problems.append(f"{label}: mock rows must not be published")

    if row.get("is_full_evaluation") is not True:
        problems.append(f"{label}: is_full_evaluation must be true")

    evaluated_samples = _as_number(row.get("evaluated_samples"))
    dataset_total_samples = _as_number(row.get("dataset_total_samples"))
    total_samples = _as_number(row.get("total_samples"))
    if evaluated_samples is None or evaluated_samples <= 0:
        problems.append(f"{label}: evaluated_samples must be positive")
    if dataset_total_samples is None or dataset_total_samples <= 0:
        problems.append(f"{label}: dataset_total_samples must be positive")
    if total_samples is None or total_samples <= 0:
        problems.append(f"{label}: total_samples must be positive")
    if (
        evaluated_samples is not None
        and dataset_total_samples is not None
        and evaluated_samples != dataset_total_samples
    ):
        problems.append(f"{label}: evaluated_samples must equal dataset_total_samples")

    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        problems.append(f"{label}: metrics must be an object")
        macro = {}
    else:
        macro = metrics.get("macro")

    if not isinstance(macro, dict):
        problems.append(f"{label}: metrics.macro must be an object")
        macro = {}

    for metric in REQUIRED_MACRO_METRICS:
        value = macro.get(metric)
        if _as_number(value) is None:
            problems.append(f"{label}: metrics.macro.{metric} is required and must be numeric")

    for metric in OPTIONAL_MACRO_METRICS:
        value = macro.get(metric)
        if value is not None and _as_number(value) is None:
            problems.append(f"{label}: metrics.macro.{metric} must be numeric")

    policy = row.get("outlier_policy")
    if not isinstance(policy, dict) or "metric" not in policy or "threshold" not in policy:
        problems.append(f"{label}: outlier_policy requires metric and threshold")

    command = row.get("command", "")
    if command:
        for pattern in LOCAL_PATH_PATTERNS:
            if pattern.search(command):
                problems.append(f"{label}: command contains a local path")


def _as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


if __name__ == "__main__":
    sys.exit(main())
