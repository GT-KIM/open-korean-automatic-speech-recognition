#!/usr/bin/env python3
"""Check a checkout against the published measurement's immutable source seals."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    protocol = json.loads((root / 'doc/benchmarks/server_accuracy_protocol_20261008.json').read_bytes())
    inventory = json.loads((root / 'docker/benchmark-package-inventory.json').read_bytes())
    digest = hashlib.sha256()
    for path in sorted((root / 'openkoasr').rglob('*.py'),
                       key=lambda path: path.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode() + b'\0')
        digest.update(path.read_bytes().replace(b'\r\n', b'\n') + b'\0')
    actual = {
        'source_sha256': digest.hexdigest(),
        'harness_sha256': hashlib.sha256((root / 'scripts/run_full_accuracy.py').read_bytes()).hexdigest(),
        'package_inventory_sha256': hashlib.sha256(json.dumps(
            inventory['packages'], sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
    }
    expected = {**protocol['environment'], 'harness_sha256': protocol['harness_sha256']}
    for key, value in actual.items():
        if value != expected[key]:
            raise ValueError(f'{key} differs from the measured experiment: {value}')
    for key in ('python', 'image_id', 'package_inventory_sha256'):
        if inventory[key] != expected[key]:
            raise ValueError(f'Archived inventory {key} mismatch')
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps({'status': 'passed', **verify(args.root)}, indent=2))


if __name__ == '__main__':
    main()
