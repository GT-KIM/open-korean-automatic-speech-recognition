import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class OnDeviceLeaderboardDataTest(unittest.TestCase):
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
                row["metrics"]["latency_percentiles"],
                aggregate["latency_percentiles"],
            )
            self.assertAlmostEqual(
                row["performance"]["qnn_rtfx_all_samples"],
                aggregate["runtime_all_samples"]["qnn_rtfx"],
            )
