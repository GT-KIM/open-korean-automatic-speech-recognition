#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = (
    "whisper_tiny",
    "whisper_base",
    "whisper_small",
    "whisper_medium",
    "qwen3_asr_0_6b",
    "qwen3_asr_1_7b",
)
DEFAULT_KSPON_SUBSETS = ("clean", "other")
DEFAULT_AIHUB_SUBSETS = ("D01", "D02", "D03", "D04")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kspon_root", required=True)
    parser.add_argument("--aihub_telephone_root", required=True)
    parser.add_argument("--output_dir", default="results/full")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--kspon_subsets", nargs="+", default=list(DEFAULT_KSPON_SUBSETS))
    parser.add_argument("--aihub_subsets", nargs="+", default=list(DEFAULT_AIHUB_SUBSETS))
    parser.add_argument("--normalization_preset", default="kspon")
    parser.add_argument("--warmup_samples", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--whisper_batch_size", type=int, default=8)
    parser.add_argument("--qwen_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_outliers", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    completed = _completed_runs(output_dir) if args.skip_existing else set()
    jobs = list(_iter_jobs(args))

    print(f"Planned jobs: {len(jobs)}")
    for index, job in enumerate(jobs, start=1):
        key = (job["model"], job["dataset"], job["subset"])
        if key in completed:
            print(f"[{index}/{len(jobs)}] skip existing {key}")
            continue

        command = _command_for_job(args, job)
        print(f"[{index}/{len(jobs)}] run {' '.join(command)}", flush=True)
        if args.dry_run:
            continue

        env = os.environ.copy()
        env.setdefault("OPENKOASR_LOG_DIR", "/tmp/openkoasr-log")
        env.setdefault("TRANSFORMERS_VERBOSITY", "error")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        subprocess.run(command, check=True, env=env)


def _iter_jobs(args):
    for model in args.models:
        for subset in args.kspon_subsets:
            yield {"model": model, "dataset": "KsponSpeech", "subset": subset}
        for subset in args.aihub_subsets:
            yield {"model": model, "dataset": "AIHubLowQualityTelephone", "subset": subset}


def _command_for_job(args, job):
    dataset_root = (
        args.kspon_root
        if job["dataset"] == "KsponSpeech"
        else args.aihub_telephone_root
    )
    batch_size = (
        args.qwen_batch_size
        if job["model"].startswith("qwen3_asr")
        else args.whisper_batch_size
    )
    command = [
        sys.executable,
        "-m",
        "openkoasr.main",
        "--dataset_name",
        job["dataset"],
        "--dataset_rootpath",
        dataset_root,
        "--dataset_subset",
        job["subset"],
        "--model_name",
        job["model"],
        "--normalization_preset",
        args.normalization_preset,
        "--output_dir",
        args.output_dir,
        "--warmup_samples",
        str(args.warmup_samples),
        "--log_interval",
        str(args.log_interval),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(args.num_workers),
    ]
    if args.log_outliers:
        command.append("--log_outliers")
    return command


def _completed_runs(output_dir):
    completed = set()
    for path in output_dir.glob("**/leaderboard_row.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("is_full_evaluation") is True:
            completed.add((row.get("model"), row.get("dataset"), row.get("subset")))
    return completed


if __name__ == "__main__":
    main()
