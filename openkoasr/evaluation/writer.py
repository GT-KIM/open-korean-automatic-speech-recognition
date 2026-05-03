import csv
import json
import math
from pathlib import Path

from openkoasr.evaluation.error_analysis import sample_error_analysis


class ResultWriter:
    def __init__(self, output_dir, save_predictions=False):
        self.output_dir = Path(output_dir)
        self.save_predictions = save_predictions

    def write(self, result):
        run_dir = self.output_dir / result.metadata.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "run_dir": run_dir,
            "summary": run_dir / "summary.json",
            "samples": run_dir / "samples.jsonl",
            "outliers": run_dir / "outliers.jsonl",
            "leaderboard_row": run_dir / "leaderboard_row.json",
        }

        _write_json(paths["summary"], result.to_dict(include_samples=False))
        _write_jsonl(
            paths["samples"],
            [
                sample.to_dict(include_predictions=self.save_predictions)
                for sample in result.samples
            ],
        )
        _write_jsonl(
            paths["outliers"],
            [
                sample.to_dict(include_predictions=self.save_predictions)
                for sample in result.samples
                if sample.is_outlier
            ],
        )
        _write_json(paths["leaderboard_row"], _leaderboard_row(result))

        if self.save_predictions:
            paths["predictions"] = run_dir / "predictions.csv"
            paths["error_analysis"] = run_dir / "error_analysis.jsonl"
            _write_predictions_csv(paths["predictions"], result.samples)
            _write_jsonl(
                paths["error_analysis"],
                [sample_error_analysis(sample) for sample in result.samples],
            )

        return paths


def _leaderboard_row(result):
    metadata = result.metadata
    aggregate = result.aggregate
    macro = aggregate.macro_average
    micro = aggregate.micro_average
    return {
        "run_id": metadata.run_id,
        "model": metadata.model_name,
        "model_repo": metadata.model_repo,
        "dataset": metadata.dataset_name,
        "subset": metadata.dataset_subset,
        "gpu": metadata.environment.get("gpu"),
        "torch": metadata.environment.get("torch"),
        "cuda": metadata.environment.get("cuda"),
        "metrics": {
            "macro": macro,
            "micro": micro,
            "latency_percentiles": aggregate.latency_percentiles,
        },
        "model_metrics": result.model_metrics,
        "total_samples": aggregate.total_samples,
        "dataset_total_samples": metadata.dataset_total_samples,
        "evaluated_samples": metadata.evaluated_samples,
        "is_full_evaluation": metadata.is_full_evaluation,
        "valid_samples": aggregate.valid_samples,
        "outlier_count": aggregate.outlier_count,
        "outlier_policy": metadata.outlier_policy.to_dict(),
        "command": metadata.command,
    }


def _write_predictions_csv(path, samples):
    fieldnames = [
        "index",
        "sample_id",
        "reference",
        "prediction",
        "normalized_reference",
        "normalized_prediction",
        "is_outlier",
        "wer",
        "cer",
        "mer",
        "jer",
        "ser",
        "rtf",
        "latency",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row = {
                "index": sample.index,
                "sample_id": sample.sample_id,
                "reference": sample.reference,
                "prediction": sample.prediction,
                "normalized_reference": sample.normalized_reference,
                "normalized_prediction": sample.normalized_prediction,
                "is_outlier": sample.is_outlier,
            }
            for metric in ("wer", "cer", "mer", "jer", "ser", "rtf", "latency"):
                row[metric] = sample.metrics.get(metric)
            writer.writerow(_json_safe(row))


def _write_json(path, data):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(data), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False))
            handle.write("\n")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value
