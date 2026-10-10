import json
import unittest
import uuid
from pathlib import Path

from openkoasr.evaluation import EvaluationRunner, OutlierPolicy, ResultWriter


class RunnerAndWriterTest(unittest.TestCase):
    def test_mock_runner_writes_artifacts(self):
        runner = EvaluationRunner.from_names(
            dataset_name="mock",
            model_name="mock",
            limit=2,
            outlier_policy=OutlierPolicy(metric="cer", threshold=1.0),
            normalization_preset="strict",
            command="test command",
        )
        result = runner.run()

        self.assertEqual(result.aggregate.total_samples, 2)
        self.assertEqual(result.aggregate.outlier_count, 0)
        self.assertEqual(result.aggregate.macro_average["cer"], 0.0)
        self.assertEqual(result.aggregate.micro_average["wer"], 0.0)

        temp_root = Path.cwd() / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        tmpdir = temp_root / f"runner-{uuid.uuid4().hex}"
        tmpdir.mkdir()
        paths = ResultWriter(tmpdir, save_predictions=True).write(result)
        summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
        row = json.loads(Path(paths["leaderboard_row"]).read_text(encoding="utf-8"))
        self.assertEqual(row["normalization_preset"], "strict")
        self.assertEqual(row["evaluation_protocol"], "v1/strict/cer>1.0")
        self.assertEqual(row["metadata_status"], "recorded")
        provenance = row["reproducibility"]
        self.assertEqual(provenance, summary["metadata"]["reproducibility"])
        self.assertEqual(provenance["execution"]["limit"], 2)
        self.assertIsNone(provenance["model_revision"])
        self.assertRegex(provenance["code"]["source_sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("jiwer", provenance["environment"]["packages"])
        self.assertEqual(row["metrics"]["all_samples_micro"]["cer"], 0.0)
        self.assertEqual(summary["aggregate"]["all_samples_micro_average"]["cer"], 0.0)
        self.assertEqual(summary["aggregate"]["total_samples"], 2)
        self.assertTrue(Path(paths["samples"]).exists())
        self.assertTrue(Path(paths["predictions"]).exists())
        self.assertTrue(Path(paths["error_analysis"]).exists())

    def test_mock_runner_respects_batch_size_without_dropping_samples(self):
        runner = EvaluationRunner.from_names(
            dataset_name="mock",
            model_name="mock",
            batch_size=2,
            outlier_policy=OutlierPolicy(metric="cer", threshold=1.0),
            normalization_preset="strict",
            command="test command",
        )
        result = runner.run()

        self.assertEqual(result.aggregate.total_samples, 3)
        self.assertEqual(result.metadata.evaluated_samples, 3)
        self.assertTrue(result.metadata.is_full_evaluation)


if __name__ == "__main__":
    unittest.main()
