import json
import subprocess
import sys
import unittest
import uuid
from pathlib import Path


class BuildPagesTest(unittest.TestCase):
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
                        "model": "mock",
                        "dataset": "mock",
                        "metrics": {"macro": {"cer": 0.0, "rtf": 0.25}},
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
        self.assertIn("Outliers", app)
        self.assertIn("Overall Model Leaderboard", app)
        self.assertIn("On-device Leaderboard", app)
        self.assertIn("AIHub", app)
        self.assertIn("All subsets", app)
        self.assertIn("model card", app)
        self.assertNotIn('"Best Run"', app)
        self.assertNotIn('"Run",', app)
        self.assertEqual(data[0]["model"], "mock")
        self.assertEqual(ondevice_data[0]["device"], "Mock Phone")
        self.assertNotIn("rtf", data[0]["metrics"]["macro"])
        self.assertEqual(data[0]["metrics"]["macro"]["rtfx"], 4.0)
        self.assertEqual(metadata["row_count"], 1)
        self.assertEqual(metadata["ondevice_row_count"], 1)


if __name__ == "__main__":
    unittest.main()
