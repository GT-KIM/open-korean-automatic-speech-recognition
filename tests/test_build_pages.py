import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from metadata_fixtures import public_metadata


class BuildPagesTest(unittest.TestCase):
    def test_invalid_leaderboard_preserves_existing_site(self):
        temp_root = Path.cwd() / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            data = root / "data.json"
            data.write_text(json.dumps([{"metrics": {"macro": {"cer": float("nan")}}}]),
                            encoding="utf-8")
            output = root / "site"
            output.mkdir()
            marker = output / "index.html"
            marker.write_text("previous site", encoding="utf-8")
            completed = subprocess.run([
                sys.executable, "scripts/build_pages.py", "--data_path", str(data),
                "--output_dir", str(output),
            ], text=True, capture_output=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(marker.read_text(encoding="utf-8"), "previous site")

    def test_builds_static_site_with_leaderboard_data(self):
        temp_root = Path.cwd() / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        root = temp_root / f"pages-{uuid.uuid4().hex}"
        root.mkdir()
        data_path = root / "leaderboard_data.json"
        ondevice_data_path = root / "ondevice_leaderboard_data.json"
        output_dir = root / "_site"
        data_path.write_text(
            json.dumps(
                [
                    {
                        "run_id": "run-1",
                        **public_metadata(),
                        "model": "whisper_tiny",
                        "model_repo": "openai/whisper-tiny",
                        "dataset": "KsponSpeech",
                        "metrics": {"macro": {"wer": 0.0, "cer": 0.0, "mer": 0.0,
                                              "jer": 0.0, "rtf": 0.25}},
                        "total_samples": 1,
                        "evaluated_samples": 1,
                        "dataset_total_samples": 1,
                        "outlier_count": 0,
                        "outlier_policy": {"metric": "cer", "threshold": 1.0},
                        "is_full_evaluation": True,
                    }
                ]
            ),
            encoding="utf-8",
        )
        ondevice_data_path.write_text(
            json.dumps(
                [
                    {
                        "run_id": "device-run-1",
                        "model": "mock-quantized",
                        "dataset": "mock",
                        "device": "Mock Phone",
                        "metrics": {"macro": {"cer": 0.1, "rtfx": 2.0}},
                        "is_full_evaluation": True,
                    }
                ]
            ),
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                "scripts/build_pages.py",
                "--data_path",
                str(data_path),
                "--ondevice_data_path",
                str(ondevice_data_path),
                "--output_dir",
                str(output_dir),
            ],
            check=True,
        )

        self.assertTrue((output_dir / "index.html").is_file())
        self.assertTrue((output_dir / "leaderboard-card.png").is_file())
        self.assertTrue((output_dir / "leaderboard-card.svg").is_file())
        self.assertTrue((output_dir / "styles.css").is_file())
        self.assertTrue((output_dir / "app.js").is_file())
        self.assertTrue((output_dir / "robots.txt").is_file())
        self.assertTrue((output_dir / "sitemap.xml").is_file())
        self.assertTrue((output_dir / "google340a95f996780abe.html").is_file())
        self.assertTrue((output_dir / ".nojekyll").is_file())
        index = (output_dir / "index.html").read_text(encoding="utf-8")
        app = (output_dir / "app.js").read_text(encoding="utf-8")
        data = json.loads((output_dir / "leaderboard_data.json").read_text(encoding="utf-8"))
        ondevice_data = json.loads(
            (output_dir / "ondevice_leaderboard_data.json").read_text(encoding="utf-8")
        )
        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertIn("datasetTabs", index)
        self.assertIn("subsetTabs", index)
        self.assertIn("SoftwareSourceCode", index)
        self.assertIn("GitHub repository", index)
        self.assertIn("leaderboard-card.png", index)
        self.assertIn("Korean ASR 분석 시리즈", index)
        self.assertIn("Main CER", app)
        self.assertIn("Outlier rate", app)
        self.assertIn("All-sample CER", app)
        self.assertIn("Overall Model Leaderboard", app)
        self.assertIn("On-device Leaderboard", app)
        self.assertIn("AIHub", app)
        self.assertIn("All subsets", app)
        self.assertIn("modelRepoUrl", app)
        self.assertIn('href="#evaluation-method"', index)
        self.assertIn('id="evaluation-method"', index)
        self.assertGreater(index.index('id="evaluation-method"'), index.index('id="leaderboardTable"'))
        self.assertNotIn('class="score-policy"', index)
        self.assertNotIn('"Best Run"', app)
        self.assertNotIn('"Run",', app)
        self.assertEqual(data[0]["model"], "whisper_tiny")
        self.assertEqual(ondevice_data[0]["device"], "Mock Phone")
        self.assertNotIn("rtf", data[0]["metrics"]["macro"])
        self.assertEqual(data[0]["metrics"]["macro"]["rtfx"], 4.0)
        self.assertEqual(metadata["row_count"], 1)
        self.assertEqual(metadata["ondevice_row_count"], 1)


if __name__ == "__main__":
    unittest.main()
