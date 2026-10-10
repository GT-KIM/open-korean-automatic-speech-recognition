#!/usr/bin/env python3
"""Full B4 Nemotron accuracy on the previously sealed six corpus slices."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_nemotron_benchmark import MODEL, write_json
from scripts.run_full_accuracy import datasets, input_identity


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(args):
    import torch
    from torch.utils.data import DataLoader
    from unittest.mock import patch
    from openkoasr.configs import get_model_config
    from openkoasr.dataset.sample import identity_collate, iter_samples, get_sample_rate
    from openkoasr.evaluation import EvaluationRunner, OutlierPolicy, ResultWriter
    from openkoasr.evaluation.provenance import code_metadata
    from openkoasr.model import ModelFactory
    from scripts.check_benchmark_environment import package_inventory
    from importlib import metadata

    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    if code_metadata()["source_sha256"] != args.expected_source:
        raise ValueError("Source differs from launch contract")
    if metadata.version("transformers") != "5.13.0" or not os.environ["ASR_IMAGE_ID"].startswith("sha256:"):
        raise ValueError("Unsealed runtime")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one CUDA GPU required")
    protocol = json.loads(args.input_protocol.read_bytes())
    inventory_path = args.inputs / "inventory.json"
    if sha(inventory_path) != "83b29a2264edaa4d9ad6b8e11d4458656d87d4fa23b8d2e0edf21f7dcb275526":
        raise ValueError("Full-accuracy input inventory changed")
    inventory = json.loads(inventory_path.read_bytes())
    entry, dataset_config, dataset = next((e, c, d) for e, c, d in datasets(protocol) if e["id"] == args.dataset)
    sealed = next(e for e in inventory["datasets"] if e["id"] == args.dataset)
    manifest = args.inputs / sealed["manifest"]
    if sha(manifest) != sealed["sha256"]:
        raise ValueError("Full-accuracy input manifest changed")
    expected = [json.loads(line) for line in manifest.read_bytes().splitlines()]
    if len(expected) != entry["samples"]:
        raise ValueError("Input count changed")

    class VerifiedDataset:
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            sample = dataset[index]
            if input_identity(sample, index) != expected[index]:
                raise ValueError(f"Input changed at index {index}")
            return sample

        def generate_dataloader(self, batch_size, shuffle, num_workers):
            if shuffle or num_workers:
                raise ValueError("Expected fixed sequential input order")
            return DataLoader(self, batch_size=batch_size, collate_fn=identity_collate)

    config = get_model_config(MODEL)
    model = ModelFactory.load_model(config)
    if model.processor.prompt_dictionary["ko-KR"] != 14 or model.model.max_symbols_per_step != 10:
        raise ValueError("Korean prompt/decoder contract changed")
    counts = {"measured": 0, "warmup": 0, "measured_forced_frame_advances": 0,
              "warmup_forced_frame_advances": 0}
    with (args.output / "journal.jsonl").open("x", encoding="utf-8") as journal, \
            (args.output / "warmup.jsonl").open("x", encoding="utf-8") as warmup:
        class AuditedRunner(EvaluationRunner):
            def _warmup(self, model, dataloader):
                for batch in dataloader:
                    for sample in iter_samples(batch):
                        if counts["warmup"] == 16:
                            return
                        prediction = model.transcribe(sample, sampling_rate=get_sample_rate(sample))
                        if not isinstance(prediction, str):
                            raise ValueError("Invalid warmup output")
                        audit = model.audit_last_generation()
                        if audit["sequences"] != 1:
                            raise ValueError("Warmup audit cardinality mismatch")
                        counts["warmup"] += 1
                        counts["warmup_forced_frame_advances"] += audit["rows"][0]["forced_frame_advances"]
                        warmup.write(json.dumps(audit) + "\n")
                        warmup.flush()

            def record(self, rows):
                audit = model.audit_last_generation()
                if audit["sequences"] != len(rows):
                    raise ValueError("Audit/output cardinality mismatch")
                for row, detail in zip(rows, audit["rows"]):
                    if detail["prompt_id"] != 14:
                        raise ValueError("Wrong language prompt")
                    row.metadata["rnnt_generation"] = detail
                    counts["measured"] += 1
                    counts["measured_forced_frame_advances"] += detail["forced_frame_advances"]
                    journal.write(json.dumps(row.to_dict(include_predictions=True), ensure_ascii=False) + "\n")
                journal.flush()
                return rows

            def _evaluate_batch(self, **kwargs):
                return self.record(super()._evaluate_batch(**kwargs))

            def _evaluate_sample(self, **kwargs):
                return self.record([super()._evaluate_sample(**kwargs)])[0]

            def _log_sample(self, sample, total):
                print(f"EVALUATED {args.dataset}: {sample.index + 1}/{total}", flush=True)

        runner = AuditedRunner(dataset_config, config, batch_size=4, num_workers=0,
                               limit=None, outlier_policy=OutlierPolicy("cer", 1.0),
                               normalization_preset="kspon", warmup_samples=16, log_interval=500,
                               command="python /app/scripts/run_nemotron_accuracy.py " + " ".join(sys.argv[1:]))
        with patch("openkoasr.evaluation.runner.DatasetFactory.load_dataset", return_value=VerifiedDataset()), \
                patch("openkoasr.evaluation.runner.ModelFactory.load_model", return_value=model), torch.inference_mode():
            result = runner.run()
    if counts["measured"] != entry["samples"] or counts["warmup"] != 16 or not result.metadata.is_full_evaluation:
        raise ValueError("Incomplete full evaluation")
    provenance = result.metadata.reproducibility
    if provenance["model_revision"] != config.model_revision or provenance["inference"]["effective_dtype"] != "bfloat16":
        raise ValueError("Model identity/dtype mismatch")
    if code_metadata()["source_sha256"] != args.expected_source:
        raise ValueError("Source changed during evaluation")
    evidence = {"image_id": os.environ["ASR_IMAGE_ID"], "source_sha256": args.expected_source,
                "harness_sha256": sha(Path(__file__)), "input_helper_sha256": sha(Path(__file__).with_name("run_full_accuracy.py")),
                "input_manifest_sha256": sha(manifest), "input_inventory_sha256": sha(inventory_path),
                "packages": package_inventory(), "termination": counts,
                "termination_policy": "require_encoder_exhaustion", "mode": "offline_full_utterance"}
    result.metadata.reproducibility["nemotron_accuracy"] = evidence
    paths = ResultWriter(args.output, save_predictions=True).write(result)
    write_json(args.output / "completion.json", {"status": "passed", "run_id": result.metadata.run_id,
               "model": MODEL, "dataset": args.dataset, "samples": counts["measured"],
               "full_evaluation": True, "summary_sha256": sha(paths["summary"]),
               "official_leaderboard_eligible": False, **evidence})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-protocol", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--dataset", choices=("kspon-clean", "kspon-other", "telephone-D01", "telephone-D02", "telephone-D03", "telephone-D04"), required=True)
    p.add_argument("--expected-source", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "dataset": args.dataset, "started_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(args.output / "status.json", state)
    try:
        evaluate(args)
        state.update(status="passed", completed_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    write_json(args.output / "status.json", state)
    return int(state["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
