from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from scripts.leaderboard_nemotron import merge_verified_nemotron, MODEL


class NemotronPublicationTest(unittest.TestCase):
    def setUp(self):
        self.path = Path("doc/benchmarks/nemotron_results_20261010.json")
        self.report = json.loads(self.path.read_bytes())
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def merge(self, rows=(), report=None):
        return merge_verified_nemotron(list(rows), self.report if report is None else report,
                                      self.sha, self.path.as_posix())

    def test_publication_preserves_existing_models_and_is_idempotent(self):
        published = json.loads(Path("doc/leaderboard_data.json").read_bytes())
        previous = [r for r in published if r["model_repo"] != MODEL]
        original = deepcopy(previous)
        merged = self.merge(previous)
        self.assertEqual(previous, original)
        self.assertEqual(merged[:len(previous)], previous)
        self.assertEqual(len(merged), 71)
        self.assertEqual(self.merge(merged), merged)

    def test_only_measured_groups_get_speed_and_rnnt_is_not_token_capped(self):
        rows = self.merge()
        self.assertEqual({r["subset"] for r in rows if "curated_speed" in r}, {"clean", "other", "all"})
        for row in rows:
            if "curated_speed" not in row:
                continue
            self.assertEqual(row["curated_speed"]["transformers"], "5.13.0")
            for name, batch in (("b1", 1), ("b4", 4)):
                track = row["curated_speed"]["tracks"][name]
                self.assertEqual(track["batch_size"], batch)
                self.assertEqual(track["termination"], "encoder_exhaustion")
                self.assertNotIn("token_limit", track)

    def test_partial_or_mismatched_results_cannot_enter_leaderboard(self):
        for mutate in (
            lambda r: r.update(status="partial_accuracy_verified"),
            lambda r: r.update(official_leaderboard_eligible=False),
            lambda r: r["accuracy_rows"].pop(),
            lambda r: r["accuracy_proofs"][0].update(samples=2999),
            lambda r: r["accuracy_rows"][0]["reproducibility"]["nemotron_accuracy"].update(image_id="other"),
            lambda r: r["accuracy_rows"][-1]["reproducibility"]["source_runs"].pop(),
            lambda r: r["speed"]["b1"].update(measured_samples=192),
        ):
            report = deepcopy(self.report)
            mutate(report)
            with self.assertRaises(ValueError):
                self.merge(report=report)

    def test_published_scores_are_the_verified_full_accuracy(self):
        rows = self.merge()
        main = [r for r in rows if r["subset"] in ("clean", "other", "all")]
        self.assertAlmostEqual(sum(r["metrics"]["macro"]["cer"] for r in main) / 3, .31354459611898633)
        self.assertEqual(sum(r["prediction_diagnostics"]["empty_predictions"] for r in main), 6923)
        for row, source in zip(rows, self.report["accuracy_rows"]):
            self.assertEqual(row["metrics"], source["metrics"])


if __name__ == "__main__":
    unittest.main()
