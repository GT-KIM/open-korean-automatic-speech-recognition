import json
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from scripts.generate_leaderboard import sanitize_public_command


class GenerateLeaderboardTest(unittest.TestCase):
    def test_generates_markdown_and_data(self):
        temp_root = Path.cwd() / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        root = temp_root / f"leaderboard-{uuid.uuid4().hex}"
        root.mkdir()
        run_dir = root / "results" / "run-1"
        run_dir.mkdir(parents=True)
        (run_dir / "leaderboard_row.json").write_text(
            json.dumps(
                {
                    "run_id": "run-1",
                    "model": "mock",
                    "dataset": "mock",
                    "subset": None,
                    "gpu": None,
                    "metrics": {"macro": {"wer": 0.0, "cer": 0.0, "latency": 0.1}},
                        "total_samples": 2,
                        "evaluated_samples": 2,
                        "dataset_total_samples": 2,
                        "is_full_evaluation": True,
                        "outlier_count": 0,
                }
            ),
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                "scripts/generate_leaderboard.py",
                "--results_dir",
                str(root / "results"),
                "--markdown_path",
                str(root / "leaderboard.md"),
                "--data_path",
                str(root / "leaderboard_data.json"),
                "--submitted_rows_path",
                str(root / "submitted_results.json"),
            ],
            check=True,
        )

        markdown = (root / "leaderboard.md").read_text(encoding="utf-8")
        data = json.loads((root / "leaderboard_data.json").read_text(encoding="utf-8"))
        self.assertIn("mock", markdown)
        self.assertEqual(data[0]["model"], "mock")

    def test_merges_submitted_rows(self):
        temp_root = Path.cwd() / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        root = temp_root / f"leaderboard-submitted-{uuid.uuid4().hex}"
        root.mkdir()
        submitted_path = root / "submitted_results.json"
        submitted_path.write_text(
            json.dumps(
                [
                    {
                        "run_id": "submitted-1",
                        "model": "submitted",
                        "dataset": "mock",
                        "metrics": {"macro": {"cer": 0.1}},
                        "evaluated_samples": 2,
                        "dataset_total_samples": 2,
                        "is_full_evaluation": True,
                    }
                ]
            ),
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                "scripts/generate_leaderboard.py",
                "--results_dir",
                str(root / "empty-results"),
                "--markdown_path",
                str(root / "leaderboard.md"),
                "--data_path",
                str(root / "leaderboard_data.json"),
                "--submitted_rows_path",
                str(submitted_path),
            ],
            check=True,
        )

        data = json.loads((root / "leaderboard_data.json").read_text(encoding="utf-8"))
        self.assertEqual(data[0]["model"], "submitted")

    def test_generated_rows_take_precedence_over_legacy_duplicates(self):
        temp_root = Path.cwd() / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        root = temp_root / f"leaderboard-dedupe-{uuid.uuid4().hex}"
        run_dir = root / "results" / "run-2"
        run_dir.mkdir(parents=True)
        generated = {
            "run_id": "run-2",
            "model": "whisper_tiny",
            "dataset": "KsponSpeech",
            "subset": "clean",
            "metrics": {"macro": {"cer": 0.2}},
            "evaluated_samples": 3000,
            "dataset_total_samples": 3000,
            "is_full_evaluation": True,
        }
        submitted = dict(generated)
        submitted["run_id"] = "readme-legacy-whisper-tiny-kspon-clean"
        submitted["metrics"] = {"macro": {"cer": 0.3}}
        (run_dir / "leaderboard_row.json").write_text(json.dumps(generated), encoding="utf-8")
        submitted_path = root / "submitted_results.json"
        submitted_path.write_text(json.dumps([submitted]), encoding="utf-8")

        subprocess.run(
            [
                sys.executable,
                "scripts/generate_leaderboard.py",
                "--results_dir",
                str(root / "results"),
                "--markdown_path",
                str(root / "leaderboard.md"),
                "--data_path",
                str(root / "leaderboard_data.json"),
                "--submitted_rows_path",
                str(submitted_path),
            ],
            check=True,
        )

        data = json.loads((root / "leaderboard_data.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["run_id"], "run-2")
        self.assertEqual(data[0]["metrics"]["macro"]["cer"], 0.2)

    def test_sanitizes_local_paths_in_public_commands(self):
        command = (
            "/tmp/work/open-korean-automatic-speech-recognition/"
            "openkoasr/main.py --dataset_name KsponSpeech "
            "--dataset_rootpath /private/data/KsponSpeech --model_name whisper_tiny"
        )

        sanitized = sanitize_public_command(command, dataset="KsponSpeech")

        self.assertIn("python -m openkoasr.main", sanitized)
        self.assertIn("--dataset_rootpath $KSPON_ROOT", sanitized)
        self.assertNotIn("/tmp/work", sanitized)
        self.assertNotIn("/private/data", sanitized)


if __name__ == "__main__":
    unittest.main()
