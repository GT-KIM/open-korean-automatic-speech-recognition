#!/usr/bin/env python3
"""Compare filtered and all-sample macro CER using hash-verified saved outputs.

Standard library only; no inference, transcript export, or leaderboard mutation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_verified(path, expected):
    require(sha(path) == expected, f'Artifact hash mismatch: {path.name}')
    return json.loads(path.read_bytes())


def checked_cer(sample):
    metrics = sample['metrics']
    counts = [metrics['cer_' + field] for field in
              ('hits', 'substitutions', 'deletions', 'insertions')]
    require(all(type(x) is int and x >= 0 for x in counts), 'Invalid CER edit counts')
    hits, substitutions, deletions, insertions = counts
    denominator = hits + substitutions + deletions
    require(denominator == len(sample['normalized_reference']), 'CER reference length mismatch')
    value = metrics['cer']
    if value is None and denominator == 0 and insertions > 0:
        require(hits == substitutions == deletions == 0 and sample['is_outlier'] is True,
                'Invalid empty-reference outlier')
        return math.inf
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'Non-finite CER: all-sample macro is undefined; do not silently drop samples')
    require(denominator > 0 or substitutions + deletions + insertions == 0,
            'Empty-reference insertion makes macro CER undefined')
    expected = (substitutions + deletions + insertions) / denominator if denominator else 0.0
    require(math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12), 'CER count mismatch')
    require(type(sample['is_outlier']) is bool and sample['is_outlier'] == (value > 1.0),
            'Outlier flag differs from strict CER > 1.0')
    return float(value)


def aggregate(values):
    require(bool(values), 'Empty comparison population')
    kept = [value for value in values if value <= 1.0]
    require(bool(kept), 'No retained samples')
    filtered = math.fsum(kept) / len(kept)
    undefined = sum(not math.isfinite(value) for value in values)
    all_samples = None if undefined else math.fsum(values) / len(values)
    return {'samples': len(values), 'retained': len(kept), 'outliers': len(values) - len(kept),
            'undefined_cer_samples': undefined,
            'filtered_macro_cer': filtered, 'all_sample_macro_cer': all_samples,
            'delta_pp': None if all_samples is None else 100 * (all_samples - filtered)}


def add_ranks(rows):
    # Competition ranks, exact unrounded values; ties share a rank.
    for metric, field in [('filtered_macro_cer', 'filtered_rank'),
                          ('all_sample_macro_cer', 'all_sample_rank')]:
        for row in rows:
            row[field] = None if any(other[metric] is None for other in rows) else (
                1 + sum(other[metric] < row[metric] for other in rows))
    for row in rows:
        row['rank_change'] = None if row['all_sample_rank'] is None else (
            row['filtered_rank'] - row['all_sample_rank'])
    return sorted(rows, key=lambda row: (row['filtered_rank'], row['model']))


def analyze(root, verified_report):
    proof = json.loads(verified_report.read_bytes())
    require(proof['status'] == 'completed_and_verified', 'Complete verified report required')
    protocol = read_verified(root / 'protocol.json', proof['protocol_sha256'])
    inventory = read_verified(root / 'inputs/inventory.json', proof['input_inventory_sha256'])
    models = [item['repo'] for item in protocol['models']]
    datasets = {item['id']: item for item in protocol['datasets']}
    expected_pairs = {(model, dataset) for model in models for dataset in datasets}
    runs = proof['runs']
    require(len(runs) == len(expected_pairs) and
            {(r['model'], r['dataset']) for r in runs} == expected_pairs, 'Run coverage mismatch')
    manifests = {}
    for spec in inventory['datasets']:
        path = (root / 'inputs' / spec['manifest']).resolve()
        require(path.is_relative_to(root.resolve()), 'Manifest outside result root')
        require(sha(path) == spec['sha256'], 'Input manifest hash mismatch')
        manifests[spec['id']] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    published = {(r['model_repo'], r['dataset'], r['subset']): r for r in proof['rows']}
    values = {model: {} for model in models}
    nonempty_values = {model: {} for model in models}
    empty_references = {}
    identities = {}
    for run in runs:
        relative = Path(run['artifact'])
        require(relative.parts[:2] == ('results', 'accuracy-full-20261008'), 'Unexpected artifact root')
        path = (root / Path(*relative.parts[2:])).with_name('samples.jsonl').resolve()
        require(path.is_relative_to(root.resolve()), 'Artifact outside result root')
        # Hash the same bytes that are parsed, avoiding a separate unchecked read.
        digest, identity = hashlib.sha256(), hashlib.sha256()
        scores, eligible, seen = [], [], set()
        manifest = manifests[run['dataset']]
        with path.open('rb') as stream:
            for index, line in enumerate(stream):
                digest.update(line)
                sample = json.loads(line)
                require(index < len(manifest), 'Unexpected extra sample')
                source = manifest[index]
                require(sample['index'] == source['index'] == index, 'Sample index mismatch')
                require(sample['sample_id'] == source['sample_id'] and sample['sample_id'] not in seen,
                        'Sample identity mismatch or duplicate')
                seen.add(sample['sample_id'])
                require(hashlib.sha256(sample['reference'].encode()).hexdigest() == source['reference_sha256'],
                        'Reference hash mismatch')
                identity.update(json.dumps([sample['sample_id'], sample['normalized_reference']],
                                           ensure_ascii=False, separators=(',', ':')).encode() + b'\n')
                score = checked_cer(sample)
                scores.append(score)
                if sample['normalized_reference']:
                    eligible.append(score)
        require(digest.hexdigest() == run['samples_sha256'], 'Samples artifact hash mismatch')
        require(len(scores) == run['samples'] == datasets[run['dataset']]['samples'] == len(manifest),
                'Sample count mismatch')
        identity_hash = identity.hexdigest()
        require(identities.setdefault(run['dataset'], identity_hash) == identity_hash,
                'Normalized input population differs across models')
        values[run['model']][run['dataset']] = scores
        nonempty_values[run['model']][run['dataset']] = eligible
        empty_references[run['dataset']] = len(scores) - len(eligible)
    groups = {key: [key] for key in datasets}
    groups['aihub-all'] = [key for key, spec in datasets.items()
                           if spec['dataset'] == 'AIHubLowQualityTelephone']
    require(len(groups['aihub-all']) == 4, 'Expected four AIHub domains')
    results = {}
    for group, parts in groups.items():
        rows = []
        for model in models:
            row = {'model': model, **aggregate([v for part in parts for v in values[model][part]])}
            if group == 'aihub-all':
                key = (model, 'AIHubLowQualityTelephone', 'all')
            else:
                spec = datasets[group]
                key = (model, spec['dataset'], spec['subset'])
            baseline = published[key]
            require(math.isclose(row['filtered_macro_cer'], baseline['metrics']['macro']['cer'],
                                 rel_tol=1e-10, abs_tol=1e-10), 'Published macro CER mismatch')
            require(row['outliers'] == baseline['outlier_count'], 'Published outlier count mismatch')
            rows.append(row)
        results[group] = add_ranks(rows)
    common_results = {group: add_ranks([
        {'model': model, **aggregate([v for part in parts for v in nonempty_values[model][part]])}
        for model in models]) for group, parts in groups.items()}
    def overall_rows(sliced):
        overall = []
        for model in models:
            slices = [next(row for row in sliced[group] if row['model'] == model)
                      for group in ('kspon-clean', 'kspon-other', 'aihub-all')]
            filtered = math.fsum(row['filtered_macro_cer'] for row in slices) / 3
            full = None if any(row['all_sample_macro_cer'] is None for row in slices) else (
                math.fsum(row['all_sample_macro_cer'] for row in slices) / 3)
            overall.append({'model': model, 'filtered_macro_cer': filtered, 'all_sample_macro_cer': full,
                            'delta_pp': None if full is None else 100 * (full - filtered),
                            'samples': sum(row['samples'] for row in slices),
                            'undefined_cer_samples': sum(row['undefined_cer_samples'] for row in slices),
                            'outliers': sum(row['outliers'] for row in slices)})
        return add_ranks(overall)
    # This is an explicitly labelled secondary population, fixed across all models.
    for model in models:
        require(sum(len(v) for v in nonempty_values[model].values()) ==
                sum(len(v) for v in values[model].values()) - sum(empty_references.values()),
                'Common population count mismatch')
    return {'id': 'outlier-rank-sensitivity-20261009', 'status': 'verified',
            'analyzed_at_utc': datetime.now(timezone.utc).isoformat(),
            'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
            'verified_report_sha256': sha(verified_report),
            'protocol_sha256': proof['protocol_sha256'],
            'input_inventory_sha256': proof['input_inventory_sha256'],
            'sample_artifact_sha256': [{'model': r['model'], 'dataset': r['dataset'],
                                       'sha256': r['samples_sha256']} for r in runs],
            'model_dataset_pairs': len(runs), 'outputs': sum(r['samples'] for r in runs),
            'method': {'metric': 'utterance macro CER', 'excluded': 'CER > 1.0',
                       'all_sample_policy': 'retain every sample without clipping; undefined macro is null, unranked',
                       'aihub_all': 'pool D01-D04 utterances before macro averaging',
                       'overall': 'equal mean of kspon-clean, kspon-other, aihub-all',
                       'ranking': 'ascending unrounded CER; competition ranks for exact ties',
                       'new_inference': False, 'official_ranking_changed': False,
                       'validation': '48 sample hashes, sealed input identity/order/reference, CER edit counts, '
                                     'normalized population equality, 56 published filtered scores and outlier counts',
                       'limitation': 'Descriptive sensitivity on these saved outputs; no uncertainty estimates '
                                     'or independent edit-distance recomputation.'},
            'overall': overall_rows(results), 'datasets': results,
            'common_nonempty_reference': {
                'policy': 'Exclude normalized empty references equally for every model and both variants; '
                          'then compare CER > 1.0 exclusion versus retention with unchanged macro aggregation.',
                'excluded_by_dataset': empty_references,
                'overall': overall_rows(common_results), 'datasets': common_results}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--verified-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite an existing analysis')
    result = analyze(args.results_root, args.verified_report)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(f"PASS {result['model_dataset_pairs']} pairs; {result['outputs']} saved outputs")


if __name__ == '__main__':
    main()
