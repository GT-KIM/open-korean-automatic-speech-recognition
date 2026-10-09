import math
import platform
import sys
import time
from collections.abc import Sequence
from copy import deepcopy
from typing import Optional

from openkoasr.configs import get_dataset_config, get_model_config
from openkoasr.dataset import DatasetFactory
from openkoasr.dataset.sample import (
    get_audio_duration_seconds,
    get_sample_audio,
    get_sample_id,
    get_sample_metadata,
    get_sample_rate,
    get_sample_text,
    iter_samples,
)
from openkoasr.evaluation.results import (
    AggregateResult,
    EvaluationRunResult,
    OutlierPolicy,
    RunMetadata,
    SampleResult,
    utc_run_id,
)
from openkoasr.evaluation.provenance import capture_reproducibility
from openkoasr.protocol import evaluation_protocol_id
from openkoasr.metrics import Evaluator
from openkoasr.model import ModelFactory
from openkoasr.normalization import normalize_text
from openkoasr.utils.logger import logger

try:
    import torch
except Exception:  # pragma: no cover - optional in mock-only environments.
    torch = None


class EvaluationRunner:
    def __init__(
        self,
        dataset_config,
        model_config,
        batch_size=1,
        num_workers=0,
        limit: Optional[int] = None,
        outlier_policy: Optional[OutlierPolicy] = None,
        normalization_preset="punctuation_agnostic",
        warmup_samples=0,
        log_interval=1,
        log_outliers=False,
        command="",
    ):
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit = limit
        self.outlier_policy = outlier_policy or OutlierPolicy()
        self.normalization_preset = normalization_preset
        self.warmup_samples = warmup_samples
        self.log_interval = max(1, int(log_interval or 1))
        self.log_outliers = bool(log_outliers)
        self.command = command

    @classmethod
    def from_names(
        cls,
        dataset_name,
        model_name,
        manifest_path=None,
        dataset_rootpath=None,
        dataset_subset=None,
        model_overrides=None,
        **kwargs,
    ):
        dataset_overrides = {}
        if manifest_path is not None:
            dataset_overrides["manifest_path"] = manifest_path
        if dataset_rootpath is not None:
            dataset_overrides["rootpath"] = dataset_rootpath
        if dataset_subset is not None:
            dataset_overrides["subset"] = dataset_subset
        model_config = deepcopy(get_model_config(model_name))
        for key, value in (model_overrides or {}).items():
            if key not in {"model_revision", "processor_revision", "dtype", "max_inference_batch_size"}:
                raise ValueError(f"Unsupported model override: {key}")
            setattr(model_config, key, value)
        if model_overrides:
            from openkoasr.model.revisions import pinned_revisions
            pinned_revisions(model_config)
            if getattr(model_config, "family", None) not in {"whisper", "qwen3_asr", "hf_ctc"}:
                raise ValueError("Model overrides require a local Transformers model")
            inner_batch = getattr(model_config, "max_inference_batch_size", None)
            if inner_batch is not None and (type(inner_batch) is not int or inner_batch < 1):
                raise ValueError("max_inference_batch_size must be a positive integer")
        return cls(
            dataset_config=get_dataset_config(dataset_name, **dataset_overrides),
            model_config=model_config,
            **kwargs,
        )

    def run(self):
        protocol = evaluation_protocol_id(self.normalization_preset, self.outlier_policy.to_dict())
        logger.info("Loading dataset and model.")
        eval_dataset = DatasetFactory.load_dataset(self.dataset_config)
        dataset_total_samples = _safe_len(eval_dataset)
        eval_dataloader = eval_dataset.generate_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        model = ModelFactory.load_model(self.model_config)
        environment = _environment_metadata()
        evaluator = Evaluator(self.model_config.evaluation.metrics)

        model_metrics = self._model_metrics(model, evaluator, eval_dataloader)
        self._warmup(model, eval_dataloader)

        samples = []
        for batch in eval_dataloader:
            batch_samples = list(iter_samples(batch))
            if self.limit is not None:
                remaining = self.limit - len(samples)
                if remaining <= 0:
                    break
                batch_samples = batch_samples[:remaining]

            if getattr(model, "supports_batch_transcribe", False) and len(batch_samples) > 1:
                sample_results = self._evaluate_batch(
                    model=model,
                    evaluator=evaluator,
                    batch_samples=batch_samples,
                    start_index=len(samples),
                )
            else:
                sample_results = [
                    self._evaluate_sample(
                        model=model,
                        evaluator=evaluator,
                        sample=sample,
                        index=len(samples) + offset,
                    )
                    for offset, sample in enumerate(batch_samples)
                ]

            for sample_result in sample_results:
                samples.append(sample_result)
                should_log = (
                    sample_result.index == 0
                    or (self.log_outliers and sample_result.is_outlier)
                    or (sample_result.index + 1) % self.log_interval == 0
                    or (
                        self.limit is None
                        and dataset_total_samples is not None
                        and sample_result.index + 1 == dataset_total_samples
                    )
                )
                if should_log:
                    self._log_sample(sample_result, total=dataset_total_samples or "?")
            if self.limit is not None and len(samples) >= self.limit:
                break

        aggregate = AggregateResult.from_samples(samples)
        # Capture after inference so call-time overrides and backend defaults are available.
        reproducibility = capture_reproducibility(model, self.model_config, {
            "batch_size": self.batch_size, "num_workers": self.num_workers,
            "warmup_samples": self.warmup_samples, "warmup_mode": "single_sample",
            "limit": self.limit,
        }, environment)
        is_full_evaluation = (
            self.limit is None
            and dataset_total_samples is not None
            and aggregate.total_samples == dataset_total_samples
        )
        metadata = RunMetadata(
            run_id=utc_run_id(),
            dataset_name=getattr(self.dataset_config, "name", None),
            dataset_subset=getattr(self.dataset_config, "subset", None),
            model_name=getattr(self.model_config, "name", None),
            model_family=getattr(self.model_config, "family", None),
            model_repo=getattr(self.model_config, "repo_name", None),
            metrics=list(getattr(self.model_config.evaluation, "metrics", [])),
            normalization_preset=self.normalization_preset,
            outlier_policy=self.outlier_policy,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            limit=self.limit,
            dataset_total_samples=dataset_total_samples,
            evaluated_samples=aggregate.total_samples,
            is_full_evaluation=is_full_evaluation,
            warmup_samples=self.warmup_samples,
            log_interval=self.log_interval,
            command=self.command,
            environment=environment,
            evaluation_protocol=protocol,
            reproducibility=reproducibility,
        )
        return EvaluationRunResult(
            metadata=metadata,
            aggregate=aggregate,
            samples=samples,
            model_metrics=model_metrics,
        )

    def _evaluate_sample(self, model, evaluator, sample, index):
        sample_rate = get_sample_rate(sample)
        audio = get_sample_audio(sample)
        audio_duration = get_audio_duration_seconds(audio, sample_rate)
        reference = get_sample_text(sample)

        _synchronize_model_device(self.model_config)
        start_time = time.perf_counter()
        prediction = model.transcribe(sample, sampling_rate=sample_rate)
        _synchronize_model_device(self.model_config)
        processing_time = time.perf_counter() - start_time
        _validate_prediction(prediction, index)
        model_processing_time = getattr(model, "last_processing_time", None)
        if model_processing_time is not None:
            processing_time = model_processing_time
        processing_time = _validate_processing_time(processing_time)

        normalized_reference = normalize_text(reference, preset=self.normalization_preset)
        normalized_prediction = normalize_text(prediction, preset=self.normalization_preset)
        metrics = evaluator.evaluate(
            sentence1=normalized_prediction,
            sentence2=normalized_reference,
            total_processing_time=processing_time,
            total_audio_length=audio_duration,
        )
        is_outlier = self.outlier_policy.is_outlier(metrics)

        return SampleResult(
            index=index,
            sample_id=get_sample_id(sample, index),
            reference=reference,
            prediction=prediction,
            normalized_reference=normalized_reference,
            normalized_prediction=normalized_prediction,
            metrics=metrics,
            processing_time=processing_time,
            audio_duration=audio_duration,
            sample_rate=sample_rate,
            is_outlier=is_outlier,
            metadata=get_sample_metadata(sample),
        )

    def _evaluate_batch(self, model, evaluator, batch_samples, start_index):
        sample_rates = [get_sample_rate(sample) for sample in batch_samples]
        references = [get_sample_text(sample) for sample in batch_samples]
        audio_durations = [
            get_audio_duration_seconds(get_sample_audio(sample), sample_rate)
            for sample, sample_rate in zip(batch_samples, sample_rates)
        ]

        _synchronize_model_device(self.model_config)
        start_time = time.perf_counter()
        predictions = model.transcribe_batch(batch_samples, sampling_rates=sample_rates)
        _synchronize_model_device(self.model_config)
        batch_processing_time = _validate_processing_time(time.perf_counter() - start_time)
        per_sample_time = batch_processing_time / max(1, len(batch_samples))

        if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes, bytearray)):
            raise ValueError("Batch predictions must be a sequence of strings.")
        if len(predictions) != len(batch_samples):
            raise ValueError(
                f"Batch prediction count mismatch: expected {len(batch_samples)}, "
                f"got {len(predictions)} (start index {start_index})."
            )
        for offset, prediction in enumerate(predictions):
            _validate_prediction(prediction, start_index + offset)

        results = []
        for offset, (sample, sample_rate, reference, prediction, audio_duration) in enumerate(
            zip(batch_samples, sample_rates, references, predictions, audio_durations)
        ):
            normalized_reference = normalize_text(reference, preset=self.normalization_preset)
            normalized_prediction = normalize_text(prediction, preset=self.normalization_preset)
            metrics = evaluator.evaluate(
                sentence1=normalized_prediction,
                sentence2=normalized_reference,
                total_processing_time=per_sample_time,
                total_audio_length=audio_duration,
            )
            is_outlier = self.outlier_policy.is_outlier(metrics)
            index = start_index + offset
            results.append(
                SampleResult(
                    index=index,
                    sample_id=get_sample_id(sample, index),
                    reference=reference,
                    prediction=prediction,
                    normalized_reference=normalized_reference,
                    normalized_prediction=normalized_prediction,
                    metrics=metrics,
                    processing_time=per_sample_time,
                    audio_duration=audio_duration,
                    sample_rate=sample_rate,
                    is_outlier=is_outlier,
                    metadata=get_sample_metadata(sample),
                )
            )
        return results

    def _model_metrics(self, model, evaluator, dataloader):
        model_metrics = {}
        if "params" in evaluator.metrics and hasattr(model, "model"):
            try:
                model_metrics.update(evaluator.metric_functions["params"](model=model.model))
            except Exception as error:
                logger.error(f"Could not calculate parameter count: {error}")
        if "flops" in evaluator.metrics and hasattr(model, "extract_input_features"):
            try:
                first_batch = next(iter(dataloader))
                first_sample = next(iter(iter_samples(first_batch)))
                sample_rate = get_sample_rate(first_sample)
                dummy_input = model.extract_input_features(first_sample, sample_rate=sample_rate)
                model_metrics.update(
                    evaluator.metric_functions["flops"](model=model.model, dummy_input=dummy_input)
                )
            except Exception as error:
                logger.error(f"Could not calculate FLOPS: {error}")
        return model_metrics

    def _warmup(self, model, dataloader):
        if self.warmup_samples <= 0:
            return
        logger.info(f"Running {self.warmup_samples} warmup sample(s).")
        warmup_count = 0
        for batch in dataloader:
            for sample in iter_samples(batch):
                if warmup_count >= self.warmup_samples:
                    return
                sample_rate = get_sample_rate(sample)
                _synchronize_model_device(self.model_config)
                prediction = model.transcribe(sample, sampling_rate=sample_rate)
                _synchronize_model_device(self.model_config)
                _validate_prediction(prediction, warmup_count)
                warmup_count += 1

    def _log_sample(self, sample, total):
        logger.info(f"Sample {sample.index + 1}/{total}:")
        logger.info(f"  - Ground Truth: {sample.normalized_reference}")
        logger.info(f"  - Prediction:   {sample.normalized_prediction}")
        if sample.is_outlier:
            logger.info(
                f"  - [OUTLIER DETECTED] based on "
                f"{self.outlier_policy.metric.upper()} > {self.outlier_policy.threshold}"
            )
        for metric in ("wer", "cer", "mer", "jer", "ser", "rtfx", "latency"):
            if metric in sample.metrics:
                logger.info(f"  - {metric.upper()}:          {sample.metrics[metric]:.4f}")
        logger.info("-" * 30)


def _validate_prediction(prediction, index):
    if not isinstance(prediction, str):
        raise ValueError(
            f"Prediction for sample index {index} must be a string, "
            f"got {type(prediction).__name__}."
        )


def _validate_processing_time(value):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("Processing time must be a finite positive number.")
    return float(value)


def _synchronize_model_device(model_config):
    if torch is None or not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return
    device = str(getattr(model_config, "device", ""))
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _environment_metadata():
    data = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if torch is None:
        data.update({"torch": None, "cuda": None, "gpu": None})
        return data
    data["torch"] = getattr(torch, "__version__", None)
    data["cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
    if torch.cuda.is_available():
        data["gpu"] = torch.cuda.get_device_name(0)
        data["gpu_count"] = torch.cuda.device_count()
    else:
        data["gpu"] = None
        data["gpu_count"] = 0
    return data


def _safe_len(value):
    try:
        return len(value)
    except Exception:
        return None
