import json
import subprocess
import sys
import unittest
import uuid
from pathlib import Path


class ValidateLeaderboardDataTest(unittest.TestCase):
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


def _row():
    return {
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
                "rtf": 0.1,
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
