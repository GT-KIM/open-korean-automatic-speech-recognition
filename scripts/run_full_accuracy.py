#!/usr/bin/env python3
"""Full-corpus accuracy using a sealed runtime and independently sealed inputs.

The existing evaluator, decoder and metrics are reused. The extra audit records
termination outside the evaluator's timer; it never filters or retries outputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def input_identity(sample, index):
    import numpy as np
    from openkoasr.dataset.sample import get_sample_audio, get_sample_id, get_sample_rate, get_sample_text
    audio = np.asarray(get_sample_audio(sample), dtype='<f4')
    assert audio.ndim == 1 and audio.size > 0 and np.isfinite(audio).all()
    return dict(index=index, sample_id=get_sample_id(sample, index),
                sample_rate=get_sample_rate(sample), frames=int(audio.size),
                waveform_sha256=sha(audio.tobytes()),
                reference_sha256=sha(get_sample_text(sample).encode('utf-8')))


def datasets(protocol):
    from openkoasr.configs import get_dataset_config
    from openkoasr.dataset import DatasetFactory
    for entry in protocol['datasets']:
        config = get_dataset_config(entry['dataset'], subset=entry['subset'], rootpath=entry['container_root'])
        dataset = DatasetFactory.load_dataset(config)
        assert len(dataset) == entry['samples'], (entry['id'], len(dataset))
        yield entry, config, dataset


def seal(protocol, output):
    output.mkdir(parents=True, exist_ok=False)
    inventory = []
    for entry, _, dataset in datasets(protocol):
        path = output / (entry['id'] + '.jsonl')
        with path.open('x', encoding='utf-8') as stream:
            for index in range(len(dataset)):
                stream.write(json.dumps(input_identity(dataset[index], index), ensure_ascii=False) + '\n')
        inventory.append({**entry, 'manifest': path.name, 'sha256': sha(path.read_bytes())})
        print(f"SEALED {entry['id']}: {len(dataset)} inputs", flush=True)
    write_json(output / 'inventory.json', {'status': 'passed', 'datasets': inventory,
               'total_samples': sum(e['samples'] for e in inventory)})


def evaluate(args, protocol):
    import torch
    from torch.utils.data import DataLoader
    from unittest.mock import patch
    from openkoasr.configs import get_model_config
    from openkoasr.dataset.sample import identity_collate
    from openkoasr.evaluation.runner import EvaluationRunner
    from openkoasr.evaluation import OutlierPolicy, ResultWriter
    from openkoasr.evaluation.provenance import capture_reproducibility, code_metadata
    from openkoasr.evaluation.speed import GenerationAudit
    from scripts.check_benchmark_environment import package_inventory

    preflight = json.loads(args.preflight.read_bytes())
    assert preflight['status'] == 'passed' and not preflight['failures']
    assert os.environ['ASR_IMAGE_ID'] == preflight['image_id'] == protocol['environment']['image_id']
    assert code_metadata()['source_sha256'] == preflight['code']['source_sha256'] == protocol['environment']['source_sha256']
    assert package_inventory() == preflight['packages']
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    gpu = preflight['checks']['gpu']
    assert torch.get_num_threads() == gpu['torch_threads'] and torch.get_num_interop_threads() == gpu['torch_interop_threads']
    assert sha(Path(__file__).read_bytes()) == protocol['harness_sha256']
    inventory_path = args.inputs / 'inventory.json'
    inventory = json.loads(inventory_path.read_bytes())
    assert inventory['status'] == 'passed' and inventory['total_samples'] == 45916
    entry, config, dataset = next((e, c, d) for e, c, d in datasets(protocol) if e['id'] == args.dataset)
    manifest_entry = next(e for e in inventory['datasets'] if e['id'] == args.dataset)
    manifest = args.inputs / manifest_entry['manifest']
    assert sha(manifest.read_bytes()) == manifest_entry['sha256']
    expected = [json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines()]
    assert len(expected) == entry['samples']

    class VerifiedDataset:
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            sample = dataset[index]
            assert input_identity(sample, index) == expected[index], ('Input changed', index)
            return sample

        def generate_dataloader(self, batch_size, shuffle, num_workers):
            assert not shuffle and num_workers == 0
            return DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=identity_collate)

    model_entry = next(m for m in protocol['models'] if m['repo'] == args.model)
    model_config = get_model_config(args.model)
    for key in ('model_revision', 'processor_revision', 'dtype', 'language', 'max_new_tokens'):
        setattr(model_config, key, model_entry[key])
    for key, value in model_entry['call_overrides'].items():
        setattr(model_config, key, value)
    model_config.device = 'cuda:0'
    if model_config.family == 'qwen3_asr':
        model_config.max_inference_batch_size = 4
    # FLOP profiling is not an accuracy metric and is not part of this run.
    model_config.evaluation.metrics = [m for m in model_config.evaluation.metrics if m != 'flops']
    from openkoasr.model import ModelFactory
    model = ModelFactory.load_model(model_config)
    backend = model.model if hasattr(model.model, 'config') else model.model.model
    backend.eval()
    execution = dict(batch_size=4, num_workers=0, warmup_samples=16, limit=args.limit)
    provenance = capture_reproducibility(model, model_config, execution, {})
    assert provenance['model_revision'] == model_entry['model_revision']
    assert provenance['processor_revision'] == model_entry['processor_revision']
    assert provenance['inference']['effective_dtype'] == 'bfloat16'
    audit = GenerationAudit(backend, model_entry['max_new_tokens'], strict=False)
    counts = {'measured': {'eos': 0, 'token_limit': 0}, 'warmup': {'eos': 0, 'token_limit': 0}}

    with (args.output / 'journal.jsonl').open('x', encoding='utf-8') as journal:
        class AuditedRunner(EvaluationRunner):
            def _warmup(self, model, dataloader):
                super()._warmup(model, dataloader)
                result = audit.check()
                assert result['sequences'] == self.warmup_samples
                for reason in result['termination_reasons']:
                    counts['warmup'][reason] += 1
                write_json(args.output / 'warmup-audit.json', result)

            def record(self, samples):
                result = audit.check()
                assert result['sequences'] == len(samples)
                for sample, reason, tokens in zip(samples, result['termination_reasons'], result['generated_tokens']):
                    assert 0 < tokens <= model_entry['max_new_tokens']
                    sample.metadata['generation_termination'] = {'reason': reason, 'generated_tokens': tokens}
                    counts['measured'][reason] += 1
                    journal.write(json.dumps(sample.to_dict(include_predictions=True), ensure_ascii=False) + '\n')
                journal.flush()
                completed = sum(counts['measured'].values())
                if completed % 500 == 0:
                    print(f"EVALUATED {completed}/{args.limit or len(dataset)}", flush=True)
                return samples

            def _evaluate_batch(self, **kwargs):
                return self.record(super()._evaluate_batch(**kwargs))

            def _evaluate_sample(self, **kwargs):
                return self.record([super()._evaluate_sample(**kwargs)])[0]

            def _log_sample(self, sample, total):
                # Text is retained only in private artifacts, not status output.
                print(f'EVALUATED {sample.index + 1}/{total}', flush=True)

        runner = AuditedRunner(config, model_config, batch_size=4, num_workers=0,
                               limit=args.limit, outlier_policy=OutlierPolicy('cer', 1.0),
                               normalization_preset='kspon', warmup_samples=16, log_interval=500,
                               command=f'python scripts/run_full_accuracy.py --model {args.model} --dataset {args.dataset} --protocol $PROTOCOL --inputs $SEALED_INPUTS --preflight $PREFLIGHT --output $OUTPUT' + (f' --limit {args.limit}' if args.limit else ''))
        try:
            with patch('openkoasr.evaluation.runner.DatasetFactory.load_dataset', return_value=VerifiedDataset()), \
                    patch('openkoasr.evaluation.runner.ModelFactory.load_model', return_value=model), torch.inference_mode():
                result = runner.run()
        finally:
            audit.close()
    assert result.metadata.evaluated_samples == (args.limit or entry['samples'])
    assert result.metadata.is_full_evaluation == (args.limit is None)
    assert result.metadata.reproducibility['inference']['decoding']['resolved_parameters']['max_new_tokens'] == model_entry['max_new_tokens']
    evidence = dict(protocol_id=protocol['id'], protocol_sha256=sha(args.protocol.read_bytes()),
                    image_id=preflight['image_id'], source_sha256=protocol['environment']['source_sha256'],
                    harness_sha256=protocol['harness_sha256'], input_manifest_sha256=sha(manifest.read_bytes()),
                    input_inventory_sha256=sha(inventory_path.read_bytes()), preflight_sha256=sha(args.preflight.read_bytes()),
                    termination=counts, speed_result_artifact=protocol['speed_results_artifact'])
    result.metadata.reproducibility['full_accuracy'] = evidence
    paths = ResultWriter(args.output, save_predictions=True).write(result)
    write_json(args.output / 'completion.json', {'status': 'passed', 'run_id': result.metadata.run_id,
               'model': args.model, 'dataset': args.dataset, 'samples': result.metadata.evaluated_samples,
               'full_evaluation': result.metadata.is_full_evaluation,
               'summary_sha256': sha(paths['summary'].read_bytes()), **evidence})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seal', action='store_true')
    p.add_argument('--inputs', type=Path)
    p.add_argument('--preflight', type=Path)
    p.add_argument('--model')
    p.add_argument('--dataset')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_bytes())
    if args.seal:
        seal(protocol, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', model=args.model, dataset=args.dataset,
                 started_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output / 'status.json', state)
    try:
        evaluate(args, protocol)
        state.update(status='passed', completed_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(args.output / 'status.json', state)


if __name__ == '__main__':
    main()
