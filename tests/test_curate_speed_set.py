import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.curate_speed_set import SLICES, apportion, identity, input_record, ordered_workload, select, stratify, write_jsonl


def candidates(count=400):
    return [dict(dataset="AIHubLowQualityTelephone", subset="D01", sample_id=f"session-{i // 2}/clip-{i}",
                 parent_group=f"session-{i // 2}", audio_duration=1 + i / 20,
                 reference_characters=10 + (i * 7) % 100, source_index=i)
            for i in range(count)]


class SpeedCurationTest(unittest.TestCase):
    def test_domain_quotas_follow_population_in_full_batch_units(self):
        domains = SLICES[2:]
        populations = [row[2] for row in domains]
        self.assertEqual([n * 4 for n in apportion(64, populations)], [row[3] for row in domains])
        self.assertEqual([n * 4 for n in apportion(16, populations)], [row[4] for row in domains])

    def test_selection_is_input_order_independent_and_group_unique(self):
        population = stratify(candidates())
        first = select(population, 64, "main")
        second = select(list(reversed(population)), 64, "main")
        self.assertEqual(first, second)
        self.assertEqual(len({row["parent_group"] for row in first}), 64)
        self.assertEqual({r["duration_stratum"] for r in first}, set(range(6)))

    def test_scores_predictions_speed_and_outliers_are_not_input_features(self):
        row = {"dataset": "KsponSpeech", "subset": "clean", "run_id": "run"}
        sample = dict(audio_duration=1.5, index=0, sample_id="0", sample_rate=16000,
                      metrics={"cer_hits": 5, "cer_substitutions": 2, "cer_deletions": 1,
                               "cer": .2, "rtfx": 1}, is_outlier=False)
        expected = input_record(sample, row)
        changed = copy.deepcopy(sample)
        changed.update(processing_time=999, prediction="do not use", is_outlier=True)
        changed["metrics"].update(cer=5, rtfx=999, cer_insertions=1000)
        self.assertEqual(input_record(changed, row), expected)
        self.assertEqual(expected["reference_characters"], 8)

    def test_pilot_is_nested_and_never_silently_replaces_infeasible_groups(self):
        population = stratify(candidates())
        main = select(population, 64, "main")
        pilot = select(main, 16, "pilot")
        self.assertEqual(len(pilot), 16)
        self.assertLessEqual({identity(r) for r in pilot}, {identity(r) for r in main})
        for row in population:
            row["parent_group"] = "single-group"
        with self.assertRaisesRegex(ValueError, "unique parent groups"):
            select(population, 64, "main")

    def test_workload_preserves_domain_labels_with_fixed_three_group_order(self):
        rows = [dict(dataset=dataset, subset=subset, sample_id=str(i))
                for i, (dataset, subset) in enumerate([
                    ("AIHubLowQualityTelephone", "D02"), ("KsponSpeech", "other"),
                    ("KsponSpeech", "clean"), ("AIHubLowQualityTelephone", "D01")])]
        ordered = ordered_workload(rows)
        self.assertEqual([r["workload_group"] for r in ordered], ["clean", "other", "telephone", "telephone"])
        self.assertEqual([r["order"] for r in ordered], list(range(4)))
        self.assertEqual(ordered_workload(list(reversed(rows))), ordered)

    def test_public_report_exposes_no_sample_id_or_transcript(self):
        path = Path(__file__).resolve().parents[1] / "doc/benchmarks/speed_curated_set_20261006.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(sum(row["selected"]["samples"] for row in report["input_evidence"]), 768)
        self.assertEqual(sum(row["pilot"]["samples"] for row in report["input_evidence"]), 192)
        self.assertTrue(all(row["cross_checked_runs"] == 8 for row in report["input_evidence"]))
        def keys(value):
            if isinstance(value, dict):
                return set(value) | set().union(*(keys(v) for v in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(v) for v in value))
            return set()
        self.assertFalse({"sample_id", "audio_path", "reference", "prediction", "text"} & keys(report))

    def test_frozen_membership_cannot_be_silently_overwritten(self):
        root = Path(__file__).resolve().parents[1] / ".tmp_tests"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / "selection.jsonl"
            original_hash = write_jsonl(path, [{"sample_id": "one"}])
            self.assertEqual(write_jsonl(path, [{"sample_id": "one"}]), original_hash)
            with self.assertRaisesRegex(ValueError, "Frozen selection differs"):
                write_jsonl(path, [{"sample_id": "two"}])
            self.assertIn("one", path.read_text())


if __name__ == "__main__":
    unittest.main()
