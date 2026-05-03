import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


RATE_METRICS = ("wer", "cer", "mer", "jer", "ser", "rtf", "latency")
EDIT_METRICS = ("wer", "cer", "mer", "jer")


def utc_run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


@dataclass
class OutlierPolicy:
    metric: str = "cer"
    threshold: float = 1.0

    def is_outlier(self, metrics: Dict[str, Any]) -> bool:
        value = metrics.get(self.metric)
        return isinstance(value, (float, int)) and value > self.threshold

    def to_dict(self):
        return asdict(self)


@dataclass
class RunMetadata:
    run_id: str
    dataset_name: str
    dataset_subset: Optional[str]
    model_name: str
    model_family: Optional[str]
    model_repo: Optional[str]
    metrics: List[str]
    normalization_preset: str
    outlier_policy: OutlierPolicy
    batch_size: int
    num_workers: int
    limit: Optional[int]
    dataset_total_samples: Optional[int]
    evaluated_samples: int
    is_full_evaluation: bool
    warmup_samples: int
    log_interval: int
    command: str
    environment: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        data = asdict(self)
        data["outlier_policy"] = self.outlier_policy.to_dict()
        return data


@dataclass
class SampleResult:
    index: int
    sample_id: str
    reference: str
    prediction: str
    normalized_reference: str
    normalized_prediction: str
    metrics: Dict[str, Any]
    processing_time: float
    audio_duration: float
    sample_rate: int
    is_outlier: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_predictions=True):
        data = asdict(self)
        if not include_predictions:
            data.pop("reference", None)
            data.pop("prediction", None)
            data.pop("normalized_reference", None)
            data.pop("normalized_prediction", None)
        return data


@dataclass
class AggregateResult:
    total_samples: int
    valid_samples: int
    outlier_count: int
    macro_average: Dict[str, float]
    micro_average: Dict[str, float]
    latency_percentiles: Dict[str, float]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_samples(cls, samples: List[SampleResult]):
        valid_samples = [sample for sample in samples if not sample.is_outlier]
        macro_average = _macro_average(valid_samples)
        micro_average = _micro_average(valid_samples)
        latency_percentiles = _latency_percentiles(
            [sample.processing_time for sample in valid_samples]
        )
        return cls(
            total_samples=len(samples),
            valid_samples=len(valid_samples),
            outlier_count=len(samples) - len(valid_samples),
            macro_average=macro_average,
            micro_average=micro_average,
            latency_percentiles=latency_percentiles,
        )


@dataclass
class EvaluationRunResult:
    metadata: RunMetadata
    aggregate: AggregateResult
    samples: List[SampleResult]
    model_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_samples=False, include_predictions=True):
        data = {
            "metadata": self.metadata.to_dict(),
            "aggregate": self.aggregate.to_dict(),
            "model_metrics": self.model_metrics,
        }
        if include_samples:
            data["samples"] = [
                sample.to_dict(include_predictions=include_predictions)
                for sample in self.samples
            ]
        return data


def _is_finite_number(value):
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def _macro_average(samples):
    totals = {}
    counts = {}
    for sample in samples:
        for metric in RATE_METRICS:
            value = sample.metrics.get(metric)
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
        substitutions = sum(float(sample.metrics.get(f"{metric}_substitutions", 0.0)) for sample in samples)
        deletions = sum(float(sample.metrics.get(f"{metric}_deletions", 0.0)) for sample in samples)
        insertions = sum(float(sample.metrics.get(f"{metric}_insertions", 0.0)) for sample in samples)
        hits = sum(float(sample.metrics.get(f"{metric}_hits", 0.0)) for sample in samples)
        denominator = hits + substitutions + deletions
        if denominator > 0:
            averages[metric] = (substitutions + deletions + insertions) / denominator

    ser_errors = sum(float(sample.metrics.get("ser_error_sentences", 0.0)) for sample in samples)
    ser_total = sum(float(sample.metrics.get("ser_total_sentences", 0.0)) for sample in samples)
    if ser_total > 0:
        averages["ser"] = ser_errors / ser_total
    return averages


def _latency_percentiles(values):
    if not values:
        return {}
    return {
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _percentile(values, percentile):
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
