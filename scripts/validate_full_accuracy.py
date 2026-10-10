#!/usr/bin/env python3
"""Verify original full-accuracy artifacts before producing publication rows."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openkoasr.evaluation.results import AggregateResult, SampleResult, OutlierPolicy
from scripts.aggregate_aihub_all import _aggregate_runs
from scripts.validate_leaderboard_data import validate_rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_numbers(value):
    if isinstance(value, dict):
        return {k: public_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [public_numbers(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            same(left, right)
    elif type(a) in (int, float) and type(b) in (int, float):
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)
    else:
        assert a == b, (a, b)


def validate(root, partial=False):
    protocol = json.loads((root / 'protocol.json').read_bytes())
    matrix = json.loads((root / 'matrix.json').read_bytes())
    inventory = json.loads((root / 'inputs/inventory.json').read_bytes())
    assert matrix['protocol_id'] == protocol['id']
    expected_pairs = {(m['repo'], d['id']) for m in protocol['models'] for d in protocol['datasets']}
    actual = {(r['model'], r['dataset']) for r in matrix['runs']}
    assert len(actual) == len(matrix['runs']) and actual <= expected_pairs
    if not partial:
        assert (root / 'launcher.exit').read_text().strip() == '0'
        assert matrix['status'] == 'completed' and actual == expected_pairs and len(actual) == 48
    rows, proofs, domains = [], [], {}
    for item in matrix['runs']:
        assert item['status'] == 'passed'
        folder = (root / item['output']).resolve()
        assert folder.is_relative_to(root.resolve())
        completion = json.loads((folder / 'completion.json').read_bytes())
        run = folder / completion['run_id']
        summary_path = run / 'summary.json'
        summary = json.loads(summary_path.read_bytes())
        row_path = run / 'leaderboard_row.json'
        row = json.loads(row_path.read_bytes())
        metadata = summary['metadata']
        evidence = metadata['reproducibility']['full_accuracy']
        model = next(m for m in protocol['models'] if m['repo'] == item['model'])
        dataset = next(d for d in protocol['datasets'] if d['id'] == item['dataset'])
        spec = next(d for d in inventory['datasets'] if d['id'] == item['dataset'])
        manifest = root / 'inputs' / spec['manifest']
        assert sha(manifest) == spec['sha256'] == evidence['input_manifest_sha256']
        assert sha(root / 'inputs/inventory.json') == evidence['input_inventory_sha256']
        assert sha(root / 'protocol.json') == evidence['protocol_sha256']
        assert sha(root / 'preflight.json') == evidence['preflight_sha256']
        assert sha(summary_path) == completion['summary_sha256']
        warmup = json.loads((folder / 'warmup-audit.json').read_bytes())
        assert warmup['sequences'] == metadata['warmup_samples'] == 16
        assert len(warmup['termination_reasons']) == len(warmup['generated_tokens']) == 16
        warmup_counts = Counter(warmup['termination_reasons'])
        assert set(warmup_counts) <= {'eos', 'token_limit'}
        assert dict(warmup_counts) == {k: v for k, v in evidence['termination']['warmup'].items() if v}
        for reason, tokens in zip(warmup['termination_reasons'], warmup['generated_tokens']):
            assert 0 < tokens <= model['max_new_tokens']
            assert reason != 'token_limit' or tokens == model['max_new_tokens']
        assert completion['termination'] == evidence['termination']
        for field in ('image_id', 'source_sha256'):
            assert evidence[field] == protocol['environment'][field]
        assert evidence['harness_sha256'] == protocol['harness_sha256']
        assert row['reproducibility'] == metadata['reproducibility']
        assert row['model_repo'] == metadata['model_repo'] == model['repo']
        assert row['dataset'] == metadata['dataset_name'] == dataset['dataset']
        assert row['subset'] == metadata['dataset_subset'] == dataset['subset']
        assert row['is_full_evaluation'] is metadata['is_full_evaluation'] is True
        assert metadata['limit'] is None and metadata['batch_size'] == 4 and metadata['num_workers'] == 0
        assert row['evaluation_protocol'] == metadata['evaluation_protocol'] == protocol['quality']['evaluation_protocol']
        rep = row['reproducibility']
        assert rep['model_revision'] == model['model_revision'] and rep['processor_revision'] == model['processor_revision']
        assert rep['inference']['effective_dtype'] == 'bfloat16'
        assert rep['inference']['backend_batch_size'] == 4
        assert rep['inference']['decoding']['resolved_parameters']['max_new_tokens'] == model['max_new_tokens']
        expected = [json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines()]
        journal = [json.loads(s) for s in (folder / 'journal.jsonl').read_text(encoding='utf-8').splitlines()]
        saved = [json.loads(s) for s in (run / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(journal) == len(saved) == len(expected) == dataset['samples']
        assert public_numbers(journal) == saved
        reasons = Counter()
        for index, (sample, source) in enumerate(zip(journal, expected)):
            assert sample['index'] == source['index'] == index
            assert sample['sample_id'] == source['sample_id']
            assert hashlib.sha256(sample['reference'].encode('utf-8')).hexdigest() == source['reference_sha256']
            assert sample['sample_rate'] == source['sample_rate']
            assert math.isclose(sample['audio_duration'], source['frames'] / source['sample_rate'], rel_tol=1e-12)
            assert isinstance(sample['prediction'], str)
            assert math.isfinite(sample['processing_time']) and sample['processing_time'] > 0
            assert sample['is_outlier'] == OutlierPolicy('cer', 1.0).is_outlier(sample['metrics'])
            termination = sample['metadata']['generation_termination']
            assert termination['reason'] in ('eos', 'token_limit')
            assert 0 < termination['generated_tokens'] <= model['max_new_tokens']
            if termination['reason'] == 'token_limit':
                assert termination['generated_tokens'] == model['max_new_tokens']
            reasons[termination['reason']] += 1
        assert dict(reasons) == {k: v for k, v in evidence['termination']['measured'].items() if v}
        aggregate = AggregateResult.from_samples([SampleResult(**s) for s in journal]).to_dict()
        same(public_numbers(aggregate), summary['aggregate'])
        same(row['metrics'], {'macro': aggregate['macro_average'], 'micro': aggregate['micro_average'],
                             'all_samples_micro': aggregate['all_samples_micro_average'],
                             'latency_percentiles': aggregate['latency_percentiles']})
        for field in ('total_samples', 'evaluated_samples', 'dataset_total_samples'):
            assert row[field] == dataset['samples']
        assert row['outlier_count'] == aggregate['outlier_count'] and row['valid_samples'] == aggregate['valid_samples']
        artifact = 'results/accuracy-full-20261008/' + row_path.relative_to(root).as_posix()
        row['_artifact'] = artifact
        row['source'] = 'verified full evaluation (native Linux, BF16, batch 4)'
        rows.append(row)
        proofs.append({'model': item['model'], 'dataset': item['dataset'], 'samples': len(journal),
                       'token_limit': reasons['token_limit'],
                       'artifact': artifact, 'summary_sha256': sha(summary_path),
                       'samples_sha256': sha(run / 'samples.jsonl'), 'row_sha256': sha(row_path)})
        if dataset['dataset'] == 'AIHubLowQualityTelephone':
            domains.setdefault((row['model'], row['model_repo']), []).append(
                {'row': row, 'row_path': row_path, 'sample_path': run / 'samples.jsonl'})
    if not partial:
        for key, runs in domains.items():
            row = _aggregate_runs(key, runs)
            row['command'] = 'python scripts/aggregate_aihub_all.py --results_dir $ACCURACY_RESULTS/formal --output_dir $ACCURACY_RESULTS/aggregated'
            row['_artifact'] = 'results/accuracy-full-20261008/aggregated/' + row['run_id'] + '/leaderboard_row.json'
            rows.append(row)
        assert len(rows) == 56 and sum(r['samples'] for r in proofs) == 367328
    problems = []
    validate_rows('full accuracy', rows, problems)
    assert not problems, problems
    return {'id': 'server-full-accuracy-results-20261008',
            'status': 'partial_verified' if partial else 'completed_and_verified',
            'validated_at_utc': datetime.now(timezone.utc).isoformat(),
            'protocol_id': protocol['id'], 'protocol_sha256': sha(root / 'protocol.json'),
            'validator_sha256': sha(Path(__file__)),
            'input_inventory_sha256': sha(root / 'inputs/inventory.json'),
            'speed_results_sha256': protocol['speed_results_sha256'],
            'model_dataset_pairs': len(proofs), 'total_measured_generations': sum(r['samples'] for r in proofs),
            'official_leaderboard_eligible': not partial, 'published': False,
            'checks': 'Full input identity/order/count, immutable model/runtime/harness, all outputs retained, termination counts, journal and saved-output equivalence, all aggregates independently recomputed.',
            'runs': proofs, 'rows': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--partial', action='store_true')
    args = p.parse_args()
    report = validate(args.results_root, args.partial)
    assert not args.output.exists(), 'Do not overwrite an existing validation report'
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f"PASS {report['model_dataset_pairs']} pairs, {report['total_measured_generations']} outputs; {report['status']}")


if __name__ == '__main__':
    main()
