import json
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class OnDeviceLeaderboardDataTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("transformers"),
                         "QNN scoring dependencies are unavailable")
    def test_qnn_aggregation_keeps_main_score_and_counts_all_samples(self):
        from scripts.run_qnn_whisper_kspon_full import aggregate_scores

        normal = dict(is_outlier=False, wer=0.1, cer=0.1, exact_match=False,
                      word_errors=1, reference_words=10, char_errors=1, reference_chars=10,
                      encoder_ms=100, decoder_ms=100, audio_duration=1)
        outlier = dict(normal, is_outlier=True, wer=2.0, cer=2.0,
                       word_errors=2, reference_words=1, char_errors=2, reference_chars=1)
        result = aggregate_scores([normal, outlier])
        self.assertEqual(result["macro_average"]["cer"], 0.1)
        self.assertAlmostEqual(result["all_samples_micro_average"]["cer"], 3 / 11)

    def assert_metrics_match(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for key, value in expected.items():
            self.assertAlmostEqual(actual[key], value)

    def test_fold7_leaderboard_rows_match_full_benchmark_artifacts(self):
        rows = json.loads(
            (ROOT / "doc" / "ondevice_leaderboard_data.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertTrue(rows)
        for row in rows:
            artifact_name = row["result_url"].rsplit("/", maxsplit=1)[-1]
            artifact = json.loads(
                (ROOT / "doc" / "benchmarks" / artifact_name).read_text(
                    encoding="utf-8"
                )
            )
            aggregate = artifact["aggregate"]
            self.assertTrue(row["is_full_evaluation"])
            self.assertEqual(row["evaluated_samples"], 3000)
            self.assertEqual(aggregate["total_samples"], 3000)
            self.assertEqual(row["valid_samples"], aggregate["valid_samples"])
            self.assertEqual(row["outlier_count"], aggregate["outlier_count"])
            self.assert_metrics_match(
                row["metrics"]["macro"], aggregate["macro_average"]
            )
            self.assert_metrics_match(
                row["metrics"]["micro"], aggregate["micro_average"]
            )
            self.assert_metrics_match(
                row["metrics"]["all_samples_micro"], aggregate["all_samples_micro_average"]
            )
            samples = artifact["samples"]
            self.assertAlmostEqual(
                row["metrics"]["all_samples_micro"]["cer"],
                sum(item["char_errors"] for item in samples) / sum(item["reference_chars"] for item in samples),
            )
            self.assert_metrics_match(
                row["metrics"]["latency_percentiles"],
                aggregate["latency_percentiles"],
            )
            self.assertAlmostEqual(
                row["performance"]["qnn_rtfx_all_samples"],
                aggregate["runtime_all_samples"]["qnn_rtfx"],
            )
