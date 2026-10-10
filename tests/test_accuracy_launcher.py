import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'docker'))
try:
    spec = importlib.util.spec_from_file_location('accuracy_launcher', ROOT / 'docker/run_accuracy.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
finally:
    sys.path.pop(0)


class AccuracyLauncherTest(unittest.TestCase):
    def test_full_matrix_and_readonly_mounts_without_real_inference(self):
        self.run_launcher()

    def test_busy_gpu_stops_before_any_container(self):
        self.run_launcher(busy=True)

    def run_launcher(self, busy=False):
        protocol_path = ROOT / 'doc/benchmarks/server_accuracy_protocol_20261008.json'
        protocol = json.loads(protocol_path.read_bytes())
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            output = parent / 'result'
            args = ['run_accuracy.py', '--protocol', str(protocol_path), '--kspon-root', str(parent),
                    '--telephone-root', str(parent), '--cache', str(parent), '--output', str(output),
                    '--environment', 'test-linux', '--host-kind', 'native-linux']
            commands = []

            def inspect(command, **kwargs):
                if command[0] == 'docker':
                    return json.dumps([{'Id': protocol['environment']['image_id'],
                                        'Os': 'linux', 'Architecture': 'amd64'}])
                return '123\n' if busy else ''

            def run(command, **kwargs):
                commands.append(command)
                self.assertIn('--network=none', command)
                self.assertIn('OMP_NUM_THREADS=8', command)
                if '--output' not in command:
                    (output / 'environment.json').write_text('{}')
                    return
                destination = output / command[command.index('--output') + 1].removeprefix('/results/')
                destination.mkdir(parents=True)
                if '--seal' in command:
                    return
                dataset = command[command.index('--dataset') + 1]
                count = 8 if '--limit' in command else next(d['samples'] for d in protocol['datasets'] if d['id'] == dataset)
                (destination / 'completion.json').write_text(json.dumps({
                    'samples': count, 'full_evaluation': '--limit' not in command, 'status': 'passed'}))

            with patch.object(sys, 'argv', args), patch.object(launcher.subprocess, 'check_output', side_effect=inspect), \
                    patch.object(launcher.subprocess, 'run', side_effect=run), patch('builtins.print'):
                if busy:
                    with self.assertRaisesRegex(RuntimeError, 'GPU compute'):
                        launcher.main()
                else:
                    self.assertEqual(launcher.main(), 0)
            matrix = json.loads((output / 'matrix.json').read_bytes())
            self.assertEqual(matrix['status'], 'failed' if busy else 'completed')
            self.assertEqual((output / 'launcher.exit').read_text().strip(), '1' if busy else '0')
            self.assertEqual(len(commands), 0 if busy else 51)
            if not busy:
                self.assertEqual(len(matrix['runs']), 48)
                for target in ('/sources/kspon', '/sources/telephone', '/cache/huggingface', '/stage/run_full_accuracy.py'):
                    self.assertTrue(any(f'target={target},readonly' in arg for arg in commands[-1]))


if __name__ == '__main__':
    unittest.main()
