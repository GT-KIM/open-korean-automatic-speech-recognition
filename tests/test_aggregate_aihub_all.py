import json
import subprocess
import sys
import unittest
import uuid
from pathlib import Path


class AggregateAIHubAllTest(unittest.TestCase):
    def test_aggregates_domain_rows_from_samples(self):
        temp_root = Path.cwd() / ".tmp_tests" / f"aggregate-aihub-{uuid.uuid4().hex}"
        results_dir = temp_root / "results"
        for index, subset in enumerate(("D01", "D02", "D03", "D04"), start=1):
            run_dir = results_dir / subset
            run_dir.mkdir(parents=True)
            row = {
                "run_id": f"run-{subset}",
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
            "rtf": 0.1,
            "latency": 0.2,
        },
        "processing_time": 0.2,
        "is_outlier": False,
    }


if __name__ == "__main__":
    unittest.main()
