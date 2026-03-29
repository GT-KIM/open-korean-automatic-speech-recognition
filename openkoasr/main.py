# Copyright (c) 2025 Gwantae Kim. All Rights Reserved.
# Licensed under the MIT License.
import torch
import argparse
import time
from collections import defaultdict

from openkoasr.configs import dataset_config_dict, get_model_config
from openkoasr.dataset import DatasetFactory
from openkoasr.model import ModelFactory
from openkoasr.metrics import Evaluator
from openkoasr.utils.logger import logger

from whisper.normalizers.basic import BasicTextNormalizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", dest='dataset_name', default='KsponSpeech')
    parser.add_argument("--model_name", dest='model_name', default='whisper_tiny')
    parser.add_argument("--outlier_metric", dest='outlier_metric', default='cer', help="Metric to use for outlier detection (e.g., 'cer', 'wer').")
    parser.add_argument("--outlier_threshold", dest='outlier_threshold', type=float, default=1.0, help="Threshold for outlier detection.")

    args = parser.parse_args()

    logger.info(f"Starting ASR evaluation with the following configuration:")
    logger.info(f"  - Dataset: {args.dataset_name}")
    logger.info(f"  - Model: {args.model_name}")
    logger.info("-" * 30)

    dataset_config = dataset_config_dict[args.dataset_name]
    eval_dataset = DatasetFactory.load_dataset(dataset_config)
    eval_dataloader = eval_dataset.generate_dataloader(
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model_config = get_model_config(args.model_name)
    logger.info(f"Resolved Model Family: {getattr(model_config, 'family', 'unknown')}")
    logger.info(f"Resolved Model Repo:   {model_config.repo_name}")
    logger.info("-" * 30)
    model = ModelFactory.load_model(model_config)

    # Initialize Evaluator
    evaluator = Evaluator(model_config.evaluation.metrics)
    normalizer = BasicTextNormalizer()

    # Get model parameters and FLOPS once
    model_performance = {}
    if "params" in evaluator.metrics:
        model_performance.update(evaluator.metric_functions["params"](model=model.model))
    if "flops" in evaluator.metrics:
        try:
            first_sample = next(iter(eval_dataloader))
            dummy_input = model.extract_input_features(first_sample, sample_rate=16000)
            model_performance.update(evaluator.metric_functions["flops"](model=model.model, dummy_input=dummy_input))
        except Exception as e:
            logger.error(f"Could not calculate FLOPS: {e}")
            logger.error("Skipping FLOPS calculation.")

    if model_performance:
        logger.info("Model Performance Metrics:")
        for key, value in model_performance.items():
            logger.info(f"  - {key}: {value}")
        logger.info("-" * 30)

    total_metrics = defaultdict(float)
    total_samples = 0
    outlier_count = 0

    for i, sample in enumerate(eval_dataloader):
        start_time = time.time()
        transcription = model.inference_sample(sample, sampling_rate=16000)
        end_time = time.time()

        processing_time = end_time - start_time
        audio_duration = len(sample[0][0]) / 16000  # Assuming 16kHz sampling rate

        # Note: sample['text'] is a list/tuple due to the dataloader's collate_fn
        ground_truth = sample[1][0] if isinstance(sample[1], (list, tuple)) else sample[1]

        ground_truth = normalizer(ground_truth)
        transcription = normalizer(transcription)

        eval_args = {
            "sentence1": transcription,
            "sentence2": ground_truth,
            "total_processing_time": processing_time,
            "total_audio_length": audio_duration,
        }
        
        results = evaluator.evaluate(**eval_args)
        
        logger.info(f"Sample {i+1}/{len(eval_dataloader)}:")
        logger.info(f"  - Ground Truth: {ground_truth}")
        logger.info(f"  - Prediction:   {transcription}")
        # Outlier detection
        is_outlier = False
        if args.outlier_metric in results and results[args.outlier_metric] > args.outlier_threshold:
            is_outlier = True
            outlier_count += 1
            logger.info(f"  - [OUTLIER DETECTED] based on {args.outlier_metric.upper()} > {args.outlier_threshold}")

        for metric, value in results.items():
            if isinstance(value, (float, int)):
                logger.info(f"  - {metric.upper()}:          {value:.4f}" if isinstance(value, float) else f"  - {metric.upper()}:          {value}")
                # Aggregate metrics for non-outliers
                if not is_outlier and metric in ["wer", "cer", "mer", "jer", "ser", "rtf", "latency"]:
                    total_metrics[metric] += value
        logger.info("-" * 30)
        
        total_samples += 1

    valid_samples = total_samples - outlier_count
    if valid_samples > 0:
        average_metrics = {metric: total / valid_samples for metric, total in total_metrics.items()}
        logger.info("\n" + "="*30)
        logger.info("Average Evaluation Metrics (Outliers Excluded):")
        for metric, value in average_metrics.items():
            logger.info(f"  - Average {metric.upper()}: {value:.4f}")
        logger.info("-" * 30)
        logger.info(f"Total Samples: {total_samples}")
        logger.info(f"Outlier Count: {outlier_count} (excluded from average)")
        logger.info(f"Outlier Criterion: {args.outlier_metric.upper()} > {args.outlier_threshold}")
        logger.info("="*30)
    else:
        logger.info("\n" + "="*30)
        logger.info("No valid samples to calculate average metrics.")
        logger.info(f"Total Samples: {total_samples}")
        logger.info(f"Outlier Count: {outlier_count}")
        logger.info(f"Outlier Criterion: {args.outlier_metric.upper()} > {args.outlier_threshold}")
        logger.info("="*30)
