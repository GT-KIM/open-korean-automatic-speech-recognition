import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

from scripts.validate_leaderboard_data import validate_file, validate_row
from metadata_fixtures import public_metadata


class ValidateLeaderboardDataTest(unittest.TestCase):
    def test_rejects_nonfinite_negative_and_boolean_metrics(self):
        for group in ("macro", "micro", "all_samples_micro"):
            for value in (float("nan"), float("inf"), -float("inf"), -0.1, True):
                with self.subTest(group=group, value=value):
                    row = _row()
                    row["metrics"].setdefault(group, {})["wer"] = value
                    problems = []
                    validate_row("rows.json", 0, row, problems)
                    self.assertTrue(problems)
                    self.assertIn(f"metrics.{group}.wer", " ".join(problems))

    def test_accepts_insertion_error_rates_above_one(self):
        row = _row()
        row["metrics"]["macro"]["wer"] = 1.5
        row["metrics"]["micro"] = {"wer": 2.0, "ser": 1.0}
        row["metrics"]["all_samples_micro"] = {"cer": 2.5, "wer": 3.0}
        problems = []
        validate_row("rows.json", 0, row, problems)
        self.assertEqual(problems, [])

    def test_rejects_invalid_counts_policy_and_optional_metrics(self):
        changes = (
            {"total_samples": 2},
            {"evaluated_samples": 1.5, "dataset_total_samples": 1.5, "total_samples": 1.5},
            {"outlier_count": -1}, {"outlier_count": True}, {"outlier_count": 2},
            {"valid_samples": 0},
            {"outlier_policy": {"metric": "cer", "threshold": float("nan")}},
            {"outlier_policy": {"metric": "", "threshold": -1}},
            {"command": ["python"]},
        )
        for change in changes:
            with self.subTest(change=change):
                row = _row()
                row.update(change)
                problems = []
                validate_row("rows.json", 0, row, problems)
                self.assertTrue(problems)
        for group, metrics in (
            ("all_samples_micro", None),
            ("all_samples_micro", {"ser": 1.1}),
            ("micro", {"ser": 1.1}),
            ("latency_percentiles", {"p50": 2.0, "p90": 1.0}),
            ("latency_percentiles", {"p50": float("inf")}),
        ):
            with self.subTest(group=group, metrics=metrics):
                row = _row()
                row["metrics"][group] = metrics
                problems = []
                validate_row("rows.json", 0, row, problems)
                self.assertTrue(problems)

    def test_rejects_duplicate_runs_and_model_dataset_slices(self):
        temp_root = Path.cwd() / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            path = Path(directory) / "rows.json"
            for change in ({"run_id": "run-2"}, {"subset": "other"}):
                with self.subTest(change=change):
                    duplicate = _row()
                    duplicate.update(change)
                    path.write_text(json.dumps([_row(), duplicate]), encoding="utf-8")
                    problems = []
                    validate_file(path, problems)
                    self.assertIn("duplicate", " ".join(problems))

    def test_accepts_full_public_row(self):
        temp_root = Path.cwd() / ".tmp_tests" / f"validate-{uuid.uuid4().hex}"
        temp_root.mkdir(parents=True)
        path = temp_root / "rows.json"
        path.write_text(json.dumps([_row()]), encoding="utf-8")

        subprocess.run(
            [sys.executable, "scripts/validate_leaderboard_data.py", str(path)],
            check=True,
        )

    def test_public_leaderboard_files_pass_default_validation(self):
        subprocess.run(
            [sys.executable, "scripts/validate_leaderboard_data.py"],
            check=True,
        )

    def test_rejects_partial_and_local_path(self):
        temp_root = Path.cwd() / ".tmp_tests" / f"validate-{uuid.uuid4().hex}"
        temp_root.mkdir(parents=True)
        row = _row()
        row["is_full_evaluation"] = False
        row["command"] = "python -m openkoasr.main --dataset_rootpath " + "/mnt" + "/f/data/KsponSpeech"
        path = temp_root / "rows.json"
        path.write_text(json.dumps([row]), encoding="utf-8")

        completed = subprocess.run(
            [sys.executable, "scripts/validate_leaderboard_data.py", str(path)],
            text=True,
            capture_output=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("is_full_evaluation", completed.stdout)
        self.assertIn("local path", completed.stdout)

    def test_rejects_missing_required_macro_metric(self):
        temp_root = Path.cwd() / ".tmp_tests" / f"validate-{uuid.uuid4().hex}"
        temp_root.mkdir(parents=True)
        row = _row()
        del row["metrics"]["macro"]["cer"]
        path = temp_root / "rows.json"
        path.write_text(json.dumps([row]), encoding="utf-8")

        completed = subprocess.run(
            [sys.executable, "scripts/validate_leaderboard_data.py", str(path)],
            text=True,
            capture_output=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("metrics.macro.cer", completed.stdout)

    def test_rejects_legacy_rtf_metric(self):
        temp_root = Path.cwd() / ".tmp_tests" / f"validate-{uuid.uuid4().hex}"
        temp_root.mkdir(parents=True)
        row = _row()
        row["metrics"]["macro"]["rtf"] = 0.1
        path = temp_root / "rows.json"
        path.write_text(json.dumps([row]), encoding="utf-8")

        completed = subprocess.run(
            [sys.executable, "scripts/validate_leaderboard_data.py", str(path)],
            text=True,
            capture_output=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("metrics.macro.rtf is legacy", completed.stdout)


def _row():
    return {
        **public_metadata(),
        "run_id": "run-1",
        "model": "whisper_tiny",
        "model_repo": "openai/whisper-tiny",
        "dataset": "KsponSpeech",
        "subset": "clean",
        "metrics": {
            "macro": {
                "wer": 0.1,
                "cer": 0.1,
                "mer": 0.1,
                "jer": 0.1,
                "rtfx": 10.0,
                "latency": 0.1,
            }
        },
        "total_samples": 1,
        "dataset_total_samples": 1,
        "evaluated_samples": 1,
        "is_full_evaluation": True,
        "outlier_count": 0,
        "outlier_policy": {"metric": "cer", "threshold": 1.0},
        "command": "python -m openkoasr.main --dataset_rootpath $KSPON_ROOT",
    }


if __name__ == "__main__":
    unittest.main()
