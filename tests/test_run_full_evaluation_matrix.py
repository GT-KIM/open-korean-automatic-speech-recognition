import unittest
from types import SimpleNamespace

from scripts.run_full_evaluation_matrix import _command_for_job


class RunFullEvaluationMatrixTest(unittest.TestCase):
    def test_command_includes_log_outliers_when_requested(self):
        args = SimpleNamespace(
            kspon_root="/data/kspon",
            aihub_telephone_root="/data/aihub",
            output_dir="results/full",
            normalization_preset="kspon",
            warmup_samples=1,
            log_interval=500,
            whisper_batch_size=8,
            qwen_batch_size=32,
            num_workers=0,
            log_outliers=True,
        )

        command = _command_for_job(
            args,
            {"model": "qwen3_asr_0_6b", "dataset": "AIHubLowQualityTelephone", "subset": "D01"},
        )

        self.assertIn("--log_outliers", command)
        self.assertIn("/data/aihub", command)
        self.assertIn("32", command)


if __name__ == "__main__":
    unittest.main()
