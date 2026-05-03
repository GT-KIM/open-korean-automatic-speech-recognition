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
