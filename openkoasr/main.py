# Copyright (c) 2025 Gwantae Kim. All Rights Reserved.
# Licensed under the MIT License.
import argparse
import sys

from openkoasr.evaluation import EvaluationRunner, OutlierPolicy, ResultWriter
from openkoasr.normalization import available_presets
from openkoasr.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", dest="dataset_name", default="KsponSpeech")
    parser.add_argument("--model_name", dest="model_name", default="whisper_tiny")
    parser.add_argument(
        "--outlier_metric",
        dest="outlier_metric",
        default="cer",
        help="Metric to use for outlier detection (e.g., 'cer', 'wer').",
    )
    parser.add_argument(
        "--outlier_threshold",
        dest="outlier_threshold",
        type=float,
        default=1.0,
        help="Threshold for outlier detection.",
    )
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--warmup_samples", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument(
        "--log_outliers",
        action="store_true",
        help="Log every outlier sample in addition to periodic progress logs.",
    )
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--dataset_rootpath", default=None)
    parser.add_argument("--dataset_subset", default=None)
    parser.add_argument(
        "--normalization_preset",
        default="punctuation_agnostic",
        choices=available_presets(),
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Persist reference/prediction text, predictions.csv, and error_analysis.jsonl.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Starting ASR evaluation with the following configuration:")
    logger.info(f"  - Dataset: {args.dataset_name}")
    logger.info(f"  - Model: {args.model_name}")
    logger.info(f"  - Output Dir: {args.output_dir}")
    logger.info("-" * 30)

    runner = EvaluationRunner.from_names(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        manifest_path=args.manifest_path,
        dataset_rootpath=args.dataset_rootpath,
        dataset_subset=args.dataset_subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
        outlier_policy=OutlierPolicy(
            metric=args.outlier_metric,
            threshold=args.outlier_threshold,
        ),
        normalization_preset=args.normalization_preset,
        warmup_samples=args.warmup_samples,
        log_interval=args.log_interval,
        log_outliers=args.log_outliers,
        command=" ".join(sys.argv),
    )
    result = runner.run()
    paths = ResultWriter(
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
    ).write(result)

    logger.info("\n" + "=" * 30)
    logger.info("Average Evaluation Metrics (Outliers Excluded):")
    for metric, value in result.aggregate.macro_average.items():
        logger.info(f"  - Macro {metric.upper()}: {value:.4f}")
    for metric, value in result.aggregate.micro_average.items():
        logger.info(f"  - Micro {metric.upper()}: {value:.4f}")
    logger.info("-" * 30)
    logger.info(f"Total Samples: {result.aggregate.total_samples}")
    logger.info(f"Outlier Count: {result.aggregate.outlier_count} (excluded from average)")
    logger.info(f"Outlier Criterion: {args.outlier_metric.upper()} > {args.outlier_threshold}")
    logger.info(f"Artifacts: {paths['run_dir']}")
    logger.info("=" * 30)


if __name__ == "__main__":
    main()
