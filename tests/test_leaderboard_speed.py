from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.leaderboard_speed import attach_curated_speed, merge_verified_accuracy


class LeaderboardSpeedTest(unittest.TestCase):
    def setUp(self):
        self.report = json.loads(Path('doc/benchmarks/server_speed_results_20261008.json').read_bytes())
        self.artifact = 'doc/benchmarks/server_speed_results_20261008.json'
        source = self.report['rows'][0]
        self.row = dict(model='whisper_tiny', model_repo=source['model'], dataset='KsponSpeech', subset='clean',
                        is_full_evaluation=True, metrics={'macro': {'rtfx': 123, 'cer': 0.2}},
                        accuracy_validation={'status': 'completed_and_verified', 'speed_results_sha256': 'report-hash'},
                        reproducibility=dict(model_revision=source['model_revision'], processor_revision=source['model_revision'],
                            execution={'batch_size': 4},
                            inference={'effective_dtype': 'bfloat16', 'decoding': {'resolved_parameters': {'max_new_tokens': 128}}},
                            full_accuracy={'image_id': self.report['image_id'], 'source_sha256': self.report['source_sha256'],
                                           'speed_result_artifact': self.artifact}))

    def attach(self, row):
        return attach_curated_speed([row], self.report, 'report-hash', self.artifact)[0]

    def test_only_matching_full_accuracy_gets_separate_speed_tracks(self):
        before = deepcopy(self.row)
        actual = self.attach(self.row)
        self.assertEqual(self.row, before)
        self.assertEqual(actual['metrics'], before['metrics'])
        self.assertEqual(actual['curated_speed']['tracks']['b1']['batch_size'], 1)
        self.assertEqual(actual['curated_speed']['tracks']['b4']['measured_generations'], 768)
        for field in ('model_revision', 'processor_revision'):
            changed = deepcopy(self.row)
            changed['reproducibility'][field] = 'different'
            self.assertNotIn('curated_speed', self.attach(changed))
        for change in ({'reproducibility': {}}, {'accuracy_validation': {}}, {'is_full_evaluation': False},
                       {'accuracy_validation': {'status': 'completed_and_verified', 'speed_results_sha256': 'different'}},
                       {'dataset': 'AIHubLowQualityTelephone', 'subset': 'D01'}):
            self.assertNotIn('curated_speed', self.attach({**self.row, **change}))

    def test_telephone_aggregate_requires_every_source_to_match(self):
        row = {**self.row, 'dataset': 'AIHubLowQualityTelephone', 'subset': 'all',
               'reproducibility': {'source_runs': [{'reproducibility': deepcopy(self.row['reproducibility'])} for _ in range(4)]}}
        self.assertEqual(self.attach(row)['curated_speed']['group'], 'telephone')
        row['reproducibility']['source_runs'][3]['reproducibility']['full_accuracy']['image_id'] = 'different'
        self.assertNotIn('curated_speed', self.attach(row))

    def test_incomplete_accuracy_cannot_replace_published_rows(self):
        for status in ('running', 'partial_verified', 'failed'):
            with self.assertRaises(ValueError):
                merge_verified_accuracy([self.row], {'status': status, 'rows': [self.row]})
        with self.assertRaises(ValueError):
            merge_verified_accuracy([], {'status': 'completed_and_verified', 'official_leaderboard_eligible': True,
                                         'rows': [self.row], 'model_dataset_pairs': 1, 'total_measured_generations': 3000})

    def test_verified_matrix_replaces_aliases_but_preserves_other_models_and_protocols(self):
        rows = []
        for index in range(8):
            for subset in ('clean', 'other', 'D01', 'D02', 'D03', 'D04', 'all'):
                rows.append({'model': f'new-name-{index}', 'model_repo': f'org/model-{index}',
                             'dataset': 'KsponSpeech' if subset in ('clean', 'other') else 'AIHubLowQualityTelephone',
                             'subset': subset, 'evaluation_protocol': 'v1/kspon/cer>1.0', 'is_full_evaluation': True})
        previous = {**rows[0], 'model': 'old-alias'}
        reference = {**previous, 'evaluation_protocol': 'different'}
        api = {**previous, 'model_repo': 'provider/api'}
        report = {'id': 'verified-report', 'status': 'completed_and_verified', 'official_leaderboard_eligible': True,
                  'model_dataset_pairs': 48, 'total_measured_generations': 367328, 'rows': rows}
        merged = merge_verified_accuracy([previous, reference, api], report)
        self.assertEqual(len(merged), 58)
        self.assertIn(reference, merged)
        self.assertIn(api, merged)
        self.assertNotIn(previous, merged)
        self.assertEqual(sum('accuracy_validation' in r for r in merged), 56)

    def test_directory_scan_does_not_publish_unverified_new_subsets(self):
        import tempfile
        from scripts.generate_leaderboard import load_rows
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'leaderboard_row.json'
            path.write_text(json.dumps(self.row))
            self.assertEqual(load_rows(Path(name)), [])


if __name__ == '__main__':
    unittest.main()
