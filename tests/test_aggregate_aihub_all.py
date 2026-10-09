import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from metadata_fixtures import public_metadata

from scripts.aggregate_aihub_all import _aggregate_runs, _load_aihub_domain_runs


class AggregateAIHubAllTest(unittest.TestCase):
    def test_all_samples_cer_includes_outliers_across_domains(self):
        temp_root = Path.cwd() / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            for subset in ("D01", "D02", "D03", "D04"):
                item = _sample(1)
                if subset == "D04":
                    item["is_outlier"] = True
                    item["metrics"].update(cer=2.0, cer_hits=1, cer_substitutions=0, cer_insertions=2)
                _write_run(root / subset, f"run-{subset}", subset, sample=item)
                if item["is_outlier"]:
                    path = root / subset / "leaderboard_row.json"
                    row = json.loads(path.read_text(encoding="utf-8"))
                    row["outlier_count"] = 1
                    path.write_text(json.dumps(row), encoding="utf-8")
            key = ("whisper_tiny", "openai/whisper-tiny")
            result = _aggregate_runs(key, _load_aihub_domain_runs(root)[key])
            self.assertEqual(result["metrics"]["macro"]["cer"], 0.25)
            self.assertEqual(result["metrics"]["micro"]["cer"], 0.25)
            self.assertAlmostEqual(result["metrics"]["all_samples_micro"]["cer"], 5 / 13)
            self.assertEqual(result["outlier_count"], 1)

    def test_rejects_mixed_conditions_and_damaged_sample_artifacts(self):
        temp_root = Path.cwd() / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        for change in ("normalization", "policy", "truncated", "duplicate"):
            with self.subTest(change=change), tempfile.TemporaryDirectory(dir=temp_root) as directory:
                root = Path(directory)
                for subset in ("D01", "D02", "D03", "D04"):
                    _write_run(root / subset, f"run-{subset}", subset)
                path = root / "D02" / "leaderboard_row.json"
                row = json.loads(path.read_text(encoding="utf-8"))
                if change == "normalization":
                    row["normalization_preset"] = "raw"
                elif change == "policy":
                    row["outlier_policy"]["threshold"] = 0.5
                elif change == "truncated":
                    (path.parent / "samples.jsonl").write_text("", encoding="utf-8")
                else:
                    row.update(total_samples=2, evaluated_samples=2, dataset_total_samples=2)
                    (path.parent / "samples.jsonl").write_text(
                        (json.dumps(_sample(1)) + "\n") * 2, encoding="utf-8"
                    )
                path.write_text(json.dumps(row), encoding="utf-8")
                key = ("whisper_tiny", "openai/whisper-tiny")
                with self.assertRaises(ValueError):
                    _aggregate_runs(key, _load_aihub_domain_runs(root)[key])

    def test_recovers_normalization_from_matching_summary(self):
        temp_root = Path.cwd() / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            for subset in ("D01", "D02", "D03", "D04"):
                _write_run(root / subset, f"run-{subset}", subset)
                path = root / subset / "leaderboard_row.json"
                row = json.loads(path.read_text(encoding="utf-8"))
                for field in public_metadata():
                    row.pop(field, None)
                path.write_text(json.dumps(row), encoding="utf-8")
                (path.parent / "summary.json").write_text(json.dumps({"metadata": {
                    "run_id": row["run_id"], "normalization_preset": "strict",
                    "model_name": row["model"], "dataset_name": row["dataset"],
                    "dataset_subset": row["subset"], "evaluated_samples": row["evaluated_samples"],
                    "outlier_policy": row["outlier_policy"],
                    **public_metadata()["reproducibility"]["execution"],
                }}), encoding="utf-8")
            key = ("whisper_tiny", "openai/whisper-tiny")
            data = _aggregate_runs(key, _load_aihub_domain_runs(root)[key])
            self.assertEqual(data["normalization_preset"], "strict")
            self.assertIn("--normalization_preset strict", data["command"])
            (root / "D01" / "summary.json").unlink()
            with self.assertRaisesRegex(ValueError, "normalization_preset"):
                _aggregate_runs(key, _load_aihub_domain_runs(root)[key])

    def test_aggregates_domain_rows_from_samples(self):
        temp_root = Path.cwd() / ".tmp_tests" / f"aggregate-aihub-{uuid.uuid4().hex}"
        results_dir = temp_root / "results"
        for index, subset in enumerate(("D01", "D02", "D03", "D04"), start=1):
            run_dir = results_dir / subset
            run_dir.mkdir(parents=True)
            row = {
                "run_id": f"run-{subset}",
                **public_metadata(),
                "model": "whisper_tiny",
                "model_repo": "openai/whisper-tiny",
                "dataset": "AIHubLowQualityTelephone",
                "subset": subset,
                "gpu": "GPU",
                "torch": "2.0",
                "cuda": "12.0",
                "model_metrics": {},
                "dataset_total_samples": 1,
                "evaluated_samples": 1,
                "is_full_evaluation": True,
                "total_samples": 1,
                "outlier_count": 0,
                "normalization_preset": "kspon",
                "outlier_policy": {"metric": "cer", "threshold": 1.0},
            }
            (run_dir / "leaderboard_row.json").write_text(
                json.dumps(row),
                encoding="utf-8",
            )
            sample = _sample(index)
            (run_dir / "samples.jsonl").write_text(
                json.dumps(sample) + "\n",
                encoding="utf-8",
            )

        output_dir = temp_root / "aggregated"
        subprocess.run(
            [
                sys.executable,
                "scripts/aggregate_aihub_all.py",
                "--results_dir",
                str(results_dir),
                "--output_dir",
                str(output_dir),
            ],
            check=True,
        )

        rows = list(output_dir.glob("**/leaderboard_row.json"))
        self.assertEqual(len(rows), 1)
        data = json.loads(rows[0].read_text(encoding="utf-8"))
        self.assertEqual(data["subset"], "all")
        self.assertEqual(data["total_samples"], 4)
        self.assertEqual(data["dataset_total_samples"], 4)
        self.assertTrue(data["is_full_evaluation"])
        self.assertAlmostEqual(data["metrics"]["macro"]["cer"], 0.25)

    def test_aggregates_only_latest_complete_run_per_subset(self):
        temp_root = Path.cwd() / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmpdir:
            results_dir = Path(tmpdir)
            for subset in ("D01", "D02", "D03", "D04"):
                _write_run(results_dir / subset, f"20260101-{subset}", subset)
            replacement = _sample(1)
            replacement["metrics"].update(
                wer=0.75, wer_hits=1, wer_substitutions=3, cer=0.75
            )
            for folder in ("new", "copied-new"):
                _write_run(
                    results_dir / folder, "20260201-D01", "D01", sample=replacement
                )
            _write_run(results_dir / "partial", "20260301-D01", "D01", full=False)
            _write_run(results_dir / "missing", "20260401-D01", "D01", saved=False)

            model_key = ("whisper_tiny", "openai/whisper-tiny")
            runs = _load_aihub_domain_runs(results_dir)[model_key]
            for ordered_runs in (runs, list(reversed(runs))):
                data = _aggregate_runs(model_key, ordered_runs)
                self.assertEqual(data["total_samples"], 4)
                self.assertEqual(data["dataset_total_samples"], 4)
                self.assertEqual(data["evaluated_samples"], 4)
                self.assertEqual(data["valid_samples"], 4)
                self.assertEqual(data["outlier_count"], 0)
                self.assertTrue(data["is_full_evaluation"])
                self.assertAlmostEqual(data["metrics"]["macro"]["cer"], 0.375)
                self.assertAlmostEqual(data["metrics"]["micro"]["wer"], 0.375)
                self.assertEqual(
                    data["aggregation"]["source_run_ids"],
                    ["20260201-D01", "20260101-D02", "20260101-D03", "20260101-D04"],
                )


def _write_run(path, run_id, subset, *, sample=None, full=True, saved=True):
    path.mkdir(parents=True)
    row = {
        "run_id": run_id,
        **public_metadata(),
        "model": "whisper_tiny",
        "model_repo": "openai/whisper-tiny",
        "dataset": "AIHubLowQualityTelephone",
        "subset": subset,
        "dataset_total_samples": 1,
        "evaluated_samples": 1,
        "is_full_evaluation": full,
        "total_samples": 1,
        "outlier_count": 0,
        "normalization_preset": "kspon",
        "outlier_policy": {"metric": "cer", "threshold": 1.0},
    }
    (path / "leaderboard_row.json").write_text(json.dumps(row), encoding="utf-8")
    if saved:
        (path / "samples.jsonl").write_text(
            json.dumps(sample if sample is not None else _sample(1)) + "\n",
            encoding="utf-8",
        )


def _sample(index):
    return {
        "index": index,
        "sample_id": str(index),
        "metrics": {
            "wer": 0.25,
            "wer_hits": 3,
            "wer_substitutions": 1,
            "wer_deletions": 0,
            "wer_insertions": 0,
            "cer": 0.25,
            "cer_hits": 3,
            "cer_substitutions": 1,
            "cer_deletions": 0,
            "cer_insertions": 0,
            "mer": 0.25,
            "mer_hits": 3,
            "mer_substitutions": 1,
            "mer_deletions": 0,
            "mer_insertions": 0,
            "jer": 0.25,
            "jer_hits": 3,
            "jer_substitutions": 1,
            "jer_deletions": 0,
            "jer_insertions": 0,
            "ser": 1.0,
            "ser_error_sentences": 1,
            "ser_total_sentences": 1,
            "rtfx": 10.0,
            "latency": 0.2,
        },
        "processing_time": 0.2,
        "audio_duration": 2.0,
        "is_outlier": False,
    }


if __name__ == "__main__":
    unittest.main()
