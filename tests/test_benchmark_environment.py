import importlib.util
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.check_benchmark_environment import package_inventory, package_mismatches

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark_docker", ROOT / "docker/benchmark.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)
PROVENANCE_SPEC = importlib.util.spec_from_file_location(
    "standalone_provenance", ROOT / "openkoasr/evaluation/provenance.py")
provenance = importlib.util.module_from_spec(PROVENANCE_SPEC)
PROVENANCE_SPEC.loader.exec_module(provenance)
IMAGE = "sha256:" + "a" * 64


class BenchmarkEnvironmentTest(unittest.TestCase):
    def test_source_hash_uses_the_same_case_sensitive_order_on_windows_and_linux(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "openkoasr"
            package.mkdir()
            (package / "Z.py").write_bytes(b"z = 1\r\n")
            (package / "a.py").write_bytes(b"a = 2\n")
            expected = hashlib.sha256(
                b"openkoasr/Z.py\0z = 1\n\0openkoasr/a.py\0a = 2\n\0").hexdigest()
            with patch.object(provenance, "__file__", str(package / "evaluation/provenance.py")):
                self.assertEqual(provenance.code_metadata()["source_sha256"], expected)

    def test_venv_distribution_takes_precedence_over_base_image(self):
        distributions = [
            SimpleNamespace(metadata={"Name": "Qwen_ASR"}, version="0.0.6"),
            SimpleNamespace(metadata={"Name": "qwen-asr"}, version="old-base-version"),
        ]
        with patch("scripts.check_benchmark_environment.metadata.distributions", return_value=distributions):
            self.assertEqual(package_inventory(), {"qwen-asr": "0.0.6"})

    def test_packages_require_exact_cuda_build_and_normalize_names(self):
        expected = "# comment\ntorch==2.13.0+cu130\nqwen-asr==0.0.6\n"
        self.assertEqual(package_mismatches(expected, {
            "torch": "2.13.0+cu130", "qwen_asr": "0.0.6",
        }), [])
        self.assertEqual(len(package_mismatches(expected, {"torch": "2.13.0"})), 2)
        self.assertEqual(package_mismatches("torchcodec==0.14.0", {
            "torchcodec": "0.14.0+cpu",
        }), [])

    def test_wsl_uses_exact_image_and_one_cuda_selector_without_network(self):
        command = benchmark.run_command(IMAGE, "local-wsl2", "wsl2", "0", Path("test output"))
        self.assertEqual(command[-1], IMAGE)
        self.assertEqual(command[command.index("--gpus") + 1], "all")
        self.assertIn("CUDA_VISIBLE_DEVICES=0", command)
        self.assertIn("--network=none", command)
        self.assertIn("--pull=never", command)
        self.assertIn("type=bind,source=test output,target=/results", command)

    def test_linux_selects_one_physical_device(self):
        command = benchmark.run_command(IMAGE, "linux-server", "native-linux", "2", Path("output"))
        self.assertEqual(command[command.index("--gpus") + 1], "device=2")
        self.assertIn("CUDA_VISIBLE_DEVICES=0", command)

    def test_mutable_image_paths_and_multiple_gpus_are_rejected(self):
        for image, environment, gpu, output in (
            ("latest", "local", "0", "output"),
            (IMAGE, "../elsewhere", "0", "output"),
            (IMAGE, "local", "0,1", "output"),
            (IMAGE, "local", "all", "output"),
            (IMAGE, "local", "0", "output,readonly"),
        ):
            with self.subTest(image=image, environment=environment, gpu=gpu, output=output):
                with self.assertRaises(ValueError):
                    benchmark.run_command(image, environment, "wsl2", gpu, Path(output))


if __name__ == "__main__":
    unittest.main()
