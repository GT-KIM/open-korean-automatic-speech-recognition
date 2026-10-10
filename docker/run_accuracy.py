#!/usr/bin/env python3
"""Portable form of the 2026-10-08 full-accuracy host launcher (48 serial runs)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from benchmark import ROOT, run_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'kspon-root', 'telephone-root', 'cache', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--environment', required=True)
    parser.add_argument('--host-kind', choices=('native-linux', 'wsl2'), required=True)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_bytes())
    harness = ROOT / 'scripts/run_full_accuracy.py'
    if hashlib.sha256(harness.read_bytes()).hexdigest() != protocol['harness_sha256']:
        parser.error('Harness bytes differ from the sealed protocol')
    for path in (args.protocol, args.kspon_root, args.telephone_root, args.cache):
        if not path.exists() or ',' in str(path.resolve()):
            parser.error(f'Missing or unsupported mount path: {path}')
    image = protocol['environment']['image_id']
    inspected = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image], text=True))[0]
    if (inspected['Id'], inspected['Os'], inspected['Architecture']) != (image, 'linux', 'amd64'):
        parser.error('The sealed Linux amd64 image is required')
    output = args.output.resolve()
    common = run_command(image, args.environment, args.host_kind, args.gpu, output)
    common[-1:-1] = ['--env', 'OMP_NUM_THREADS=8', '--env', 'MKL_NUM_THREADS=8',
                     '--env', 'PYTHONPATH=/app']
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.protocol, output / 'protocol.json')
    matrix = {'protocol_id': protocol['id'], 'status': 'running', 'runs': []}

    def save_matrix():
        (output / 'matrix.json').write_text(json.dumps(matrix, indent=2) + '\n', encoding='utf-8')

    def call(command, log_name):
        if args.host_kind == 'native-linux':
            active = subprocess.check_output(['nvidia-smi', '-i', args.gpu,
                '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            if active:
                raise RuntimeError('Another GPU compute process is active')
        with (output / log_name).open('x', encoding='utf-8') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    code = 1
    try:
        save_matrix()
        call(common, 'preflight.log')
        shutil.copyfile(output / 'environment.json', output / 'preflight.json')
        mounts = []
        for source, target in ((output / 'protocol.json', '/protocol.json'),
                               (harness, '/stage/run_full_accuracy.py'),
                               (args.kspon_root, '/sources/kspon'),
                               (args.telephone_root, '/sources/telephone'),
                               (args.cache, '/cache/huggingface')):
            mounts.extend(['--mount', f'type=bind,source={source.resolve()},target={target},readonly'])
        common[-1:-1] = mounts
        common += ['python', '/stage/run_full_accuracy.py', '--protocol', '/protocol.json']
        call(common + ['--seal', '--output', '/results/inputs'], 'seal.log')
        base = common + ['--inputs', '/results/inputs', '--preflight', '/results/preflight.json']
        call(base + ['--model', protocol['models'][0]['repo'], '--dataset', 'kspon-clean',
                     '--limit', '8', '--output', '/results/smoke'], 'smoke.log')
        smoke = json.loads((output / 'smoke/completion.json').read_bytes())
        if smoke['samples'] != 8 or smoke['full_evaluation'] or smoke['status'] != 'passed':
            raise RuntimeError('Bounded smoke run did not pass')
        for model in protocol['models']:
            for dataset in protocol['datasets']:
                folder = 'formal/' + model['repo'].replace('/', '--') + '/' + dataset['id']
                item = {'model': model['repo'], 'dataset': dataset['id'], 'output': folder, 'status': 'running'}
                matrix['runs'].append(item)
                save_matrix()
                print(f"Running {model['repo']} {dataset['id']}", flush=True)
                call(base + ['--model', model['repo'], '--dataset', dataset['id'],
                             '--output', '/results/' + folder], folder.replace('/', '--') + '.log')
                completion = json.loads((output / folder / 'completion.json').read_bytes())
                if not completion['full_evaluation'] or completion['samples'] != dataset['samples']:
                    raise RuntimeError('Incomplete formal run')
                item['status'] = 'passed'
                save_matrix()
        matrix['status'] = 'completed'
        code = 0
    finally:
        if code:
            matrix['status'] = 'failed'
        save_matrix()
        (output / 'launcher.exit').write_text(str(code) + '\n', encoding='utf-8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
