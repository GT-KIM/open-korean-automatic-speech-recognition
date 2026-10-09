"""Attach measured speed only to matching, verified full-accuracy executions."""
from copy import deepcopy


def model_identity(row):
    return str(row.get('model_repo') or row.get('model') or '').removeprefix('https://huggingface.co/').strip('/').lower()


def merge_verified_accuracy(rows, report):
    if report.get('status') != 'completed_and_verified' or report.get('official_leaderboard_eligible') is not True:
        raise ValueError('Full accuracy must be completed and verified before publication')
    new = report['rows']
    if len(new) != 56 or report.get('model_dataset_pairs') != 48 or report.get('total_measured_generations') != 367328:
        raise ValueError('Expected all 48 full runs and eight complete telephone aggregates')
    key = lambda r: (model_identity(r), r.get('dataset'), r.get('subset'), r.get('evaluation_protocol'))
    replacements = {key(r) for r in new}
    if len(replacements) != len(new) or any(r.get('is_full_evaluation') is not True for r in new):
        raise ValueError('Duplicate or partial full-accuracy rows')
    verified = deepcopy(new)
    for row in verified:
        row['accuracy_validation'] = {'status': 'completed_and_verified', 'report_id': report.get('id'),
                                      'protocol_sha256': report.get('protocol_sha256'),
                                      'speed_results_sha256': report.get('speed_results_sha256')}
    return [r for r in rows if key(r) not in replacements] + verified


def attach_curated_speed(rows, report, report_sha256, artifact):
    if report.get('status') != 'completed_and_verified' or report.get('model_track_pairs') != 16:
        raise ValueError('A complete verified speed report is required')
    speed_rows = {(r['model'].lower(), r['track']): r for r in report['rows']}
    if len(speed_rows) != 16:
        raise ValueError('Duplicate speed model/track pairs')
    result = []
    for original in rows:
        row = deepcopy(original)
        row.pop('curated_speed', None)
        group = {('KsponSpeech', 'clean'): 'clean', ('KsponSpeech', 'other'): 'other',
                 ('AIHubLowQualityTelephone', 'all'): 'telephone'}.get((row.get('dataset'), row.get('subset')))
        rep = row.get('reproducibility') or {}
        records = [s.get('reproducibility') or {} for s in rep.get('source_runs', [])] if 'source_runs' in rep else [rep]
        b1 = speed_rows.get((model_identity(row), 'latency-b1'))
        b4 = speed_rows.get((model_identity(row), 'throughput-b4'))
        compatible = bool(group and b1 and b4 and records and row.get('is_full_evaluation') is True
                          and row.get('accuracy_validation', {}).get('status') == 'completed_and_verified'
                          and row.get('accuracy_validation', {}).get('speed_results_sha256') == report_sha256)
        for record in records:
            evidence = record.get('full_accuracy') or {}
            inference = record.get('inference') or {}
            compatible = compatible and (
                evidence.get('image_id') == report['image_id']
                and evidence.get('source_sha256') == report['source_sha256']
                and evidence.get('speed_result_artifact') == artifact
                and record.get('model_revision') == b4['model_revision']
                and record.get('processor_revision') == b4['model_revision']
                and inference.get('effective_dtype') == b4['dtype']
                and inference.get('decoding', {}).get('resolved_parameters', {}).get('max_new_tokens') == b4['max_new_tokens']
                and record.get('execution', {}).get('batch_size') == 4)
        if compatible:
            tracks = {}
            for name, source in [('b1', b1), ('b4', b4)]:
                terminations = [r['groups'][group] for r in source['termination']['repeats']]
                total = sum(t['sequences'] for t in terminations)
                capped = sum(t['token_limit'] for t in terminations)
                tracks[name] = {**deepcopy(source['groups'][group]), 'batch_size': source['batch_size'],
                                'token_limit': capped, 'measured_generations': total, 'token_limit_rate': capped / total}
            row['curated_speed'] = dict(status='verified', group=group, samples_per_repeat=256, repetitions=3,
                protocol_id=report['protocol_id'], executed_protocol_sha256=report['executed_protocol_sha256'],
                report_artifact=artifact, report_sha256=report_sha256, environment_id=report['environment_id'],
                gpu=report['gpu'], image_id=report['image_id'], source_sha256=report['source_sha256'], tracks=tracks)
        result.append(row)
    return result
