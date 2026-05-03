import platform
import sys
import time
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
)
from openkoasr.evaluation.results import (
    AggregateResult,
    EvaluationRunResult,
    OutlierPolicy,
    RunMetadata,
    SampleResult,
    utc_run_id,
)
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
        self.command = command

    @classmethod
    def from_names(
        cls,
        dataset_name,
        model_name,
        manifest_path=None,
        dataset_rootpath=None,
        dataset_subset=None,
        **kwargs,
    ):
        dataset_overrides = {}
        if manifest_path is not None:
            dataset_overrides["manifest_path"] = manifest_path
        if dataset_rootpath is not None:
            dataset_overrides["rootpath"] = dataset_rootpath
        if dataset_subset is not None:
            dataset_overrides["subset"] = dataset_subset
        return cls(
            dataset_config=get_dataset_config(dataset_name, **dataset_overrides),
            model_config=get_model_config(model_name),
            **kwargs,
        )

    def run(self):
        logger.info("Loading dataset and model.")
        eval_dataset = DatasetFactory.load_dataset(self.dataset_config)
        dataset_total_samples = _safe_len(eval_dataset)
        eval_dataloader = eval_dataset.generate_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        model = ModelFactory.load_model(self.model_config)
        evaluator = Evaluator(self.model_config.evaluation.metrics)

        model_metrics = self._model_metrics(model, evaluator, eval_dataloader)
        self._warmup(model, eval_dataloader)

        samples = []
        for raw_index, sample in enumerate(eval_dataloader):
            if self.limit is not None and len(samples) >= self.limit:
                break

            sample_result = self._evaluate_sample(
                model=model,
                evaluator=evaluator,
                sample=sample,
                index=raw_index,
            )
            samples.append(sample_result)
            should_log = (
                sample_result.index == 0
                or sample_result.is_outlier
                or (sample_result.index + 1) % self.log_interval == 0
                or (
                    self.limit is None
                    and dataset_total_samples is not None
                    and sample_result.index + 1 == dataset_total_samples
                )
            )
            if should_log:
                self._log_sample(sample_result, total=dataset_total_samples or "?")

        aggregate = AggregateResult.from_samples(samples)
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
            environment=_environment_metadata(),
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

    def _model_metrics(self, model, evaluator, dataloader):
        model_metrics = {}
        if "params" in evaluator.metrics and hasattr(model, "model"):
            try:
                model_metrics.update(evaluator.metric_functions["params"](model=model.model))
            except Exception as error:
                logger.error(f"Could not calculate parameter count: {error}")
        if "flops" in evaluator.metrics and hasattr(model, "extract_input_features"):
            try:
                first_sample = next(iter(dataloader))
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
        for index, sample in enumerate(dataloader):
            if index >= self.warmup_samples:
                break
            sample_rate = get_sample_rate(sample)
            _synchronize_model_device(self.model_config)
            model.transcribe(sample, sampling_rate=sample_rate)
            _synchronize_model_device(self.model_config)

    def _log_sample(self, sample, total):
        logger.info(f"Sample {sample.index + 1}/{total}:")
        logger.info(f"  - Ground Truth: {sample.normalized_reference}")
        logger.info(f"  - Prediction:   {sample.normalized_prediction}")
        if sample.is_outlier:
            logger.info(
                f"  - [OUTLIER DETECTED] based on "
                f"{self.outlier_policy.metric.upper()} > {self.outlier_policy.threshold}"
            )
        for metric in ("wer", "cer", "mer", "jer", "ser", "rtf", "latency"):
            if metric in sample.metrics:
                logger.info(f"  - {metric.upper()}:          {sample.metrics[metric]:.4f}")
        logger.info("-" * 30)


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
