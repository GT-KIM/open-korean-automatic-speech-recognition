import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_fold7_leaderboard_rows_match_full_benchmark_artifacts():
    rows = json.loads(
        (ROOT / "doc" / "ondevice_leaderboard_data.json").read_text(encoding="utf-8")
    )

    assert rows
    for row in rows:
        artifact_name = row["result_url"].rsplit("/", maxsplit=1)[-1]
        artifact = json.loads(
            (ROOT / "doc" / "benchmarks" / artifact_name).read_text(encoding="utf-8")
        )
        aggregate = artifact["aggregate"]
        assert row["is_full_evaluation"] is True
        assert row["evaluated_samples"] == aggregate["total_samples"] == 3000
        assert row["valid_samples"] == aggregate["valid_samples"]
        assert row["outlier_count"] == aggregate["outlier_count"]
        assert row["metrics"]["macro"] == pytest.approx(aggregate["macro_average"])
        assert row["metrics"]["micro"] == pytest.approx(aggregate["micro_average"])
        assert row["metrics"]["latency_percentiles"] == pytest.approx(
            aggregate["latency_percentiles"]
        )
        assert row["performance"]["qnn_rtfx_all_samples"] == pytest.approx(
            aggregate["runtime_all_samples"]["qnn_rtfx"]
        )
