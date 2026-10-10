import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_outlier_rank_sensitivity import aggregate, add_ranks, checked_cer, read_verified
from scripts.verify_benchmark_release import verify


def sample(hits, substitutions=0, deletions=0, insertions=0):
    reference = hits + substitutions + deletions
    value = (substitutions + deletions + insertions) / reference if reference else None
    return {'normalized_reference': '가' * reference, 'is_outlier': value is None or value > 1,
            'metrics': {'cer': value, 'cer_hits': hits, 'cer_substitutions': substitutions,
                        'cer_deletions': deletions, 'cer_insertions': insertions}}


class OutlierSensitivityTest(unittest.TestCase):
    def test_same_macro_definition_and_strict_threshold(self):
        values = [checked_cer(sample(9, 1)), checked_cer(sample(0, 1)),
                  checked_cer(sample(1, insertions=2))]
        result = aggregate(values)
        self.assertEqual((result['samples'], result['retained'], result['outliers']), (3, 2, 1))
        self.assertAlmostEqual(result['filtered_macro_cer'], 0.55)
        self.assertAlmostEqual(result['all_sample_macro_cer'], 3.1 / 3)
        # This is deliberately different from corpus CER (4 / 12).
        self.assertNotAlmostEqual(result['all_sample_macro_cer'], 4 / 12)

    def test_pooling_domains_preserves_utterance_weights(self):
        pooled = aggregate([0, 0, 0] + [1])
        self.assertEqual(pooled['all_sample_macro_cer'], 0.25)

    def test_rank_reversal_and_exact_ties(self):
        rows = add_ranks([{'model': 'a', 'filtered_macro_cer': .1, 'all_sample_macro_cer': .5},
                          {'model': 'b', 'filtered_macro_cer': .2, 'all_sample_macro_cer': .3},
                          {'model': 'c', 'filtered_macro_cer': .2, 'all_sample_macro_cer': .3}])
        self.assertEqual([r['filtered_rank'] for r in rows], [1, 2, 2])
        self.assertEqual([r['all_sample_rank'] for r in rows], [3, 1, 1])
        self.assertEqual(rows[0]['rank_change'], -2)

    def test_undefined_full_macro_is_unranked_and_not_silently_dropped(self):
        result = aggregate([0.1, checked_cer(sample(0, insertions=3))])
        self.assertIsNone(result['all_sample_macro_cer'])
        self.assertEqual(result['undefined_cer_samples'], 1)
        self.assertEqual(result['filtered_macro_cer'], 0.1)
        rows = add_ranks([{'model': 'a', **result}])
        self.assertIsNone(rows[0]['all_sample_rank'])

    def test_reject_inconsistent_saved_metrics(self):
        for row in [sample(1), sample(2)]:
            row['is_outlier'] = True
            with self.assertRaises(ValueError):
                checked_cer(row)

    def test_tampered_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report.json'
            path.write_bytes(b'{"status":"passed"}')
            seal = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(read_verified(path, seal)['status'], 'passed')
            path.write_bytes(b'{"status":"changed"}')
            with self.assertRaises(ValueError):
                read_verified(path, seal)

    def test_source_verifier_rejects_changed_source(self):
        # A fixture keeps future development free to differ from the historical release.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('openkoasr', 'scripts', 'docker', 'doc/benchmarks'):
                (root / name).mkdir(parents=True)
            source = root / 'openkoasr/example.py'
            source.write_bytes(b'pass\r\n')
            harness = root / 'scripts/run_full_accuracy.py'
            harness.write_bytes(b'pass\r\n')
            packages = {'example': '1.0'}
            inventory_hash = hashlib.sha256(json.dumps(packages, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            inventory = {'packages': packages, 'python': '3.12.3', 'image_id': 'fixture',
                         'package_inventory_sha256': inventory_hash}
            (root / 'docker/benchmark-package-inventory.json').write_text(json.dumps(inventory))
            source_hash = hashlib.sha256(b'openkoasr/example.py\0pass\n\0').hexdigest()
            protocol = {'environment': {**inventory, 'source_sha256': source_hash},
                        'harness_sha256': hashlib.sha256(harness.read_bytes()).hexdigest()}
            (root / 'doc/benchmarks/server_accuracy_protocol_20261008.json').write_text(json.dumps(protocol))
            verify(root)
            source.write_bytes(b'changed\n')
            with self.assertRaisesRegex(ValueError, 'source_sha256'):
                verify(root)


if __name__ == '__main__':
    unittest.main()
