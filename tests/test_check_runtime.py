import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from scripts import check_runtime


class RuntimeCheckTest(unittest.TestCase):
    def test_native_and_backend_import_errors_fail_without_skipping_other_checks(self):
        for broken in ("torchaudio", "torchcodec", "openkoasr.model.qwen3_asr"):
            with self.subTest(module=broken):
                def import_module(name):
                    if name == broken:
                        raise OSError("incompatible native library")
                    return SimpleNamespace(__version__="test")

                output, errors = io.StringIO(), io.StringIO()
                with (
                    patch.object(check_runtime.importlib, "import_module", side_effect=import_module),
                    patch.object(check_runtime, "check_audio") as audio,
                    patch.object(check_runtime, "check_cuda") as cuda,
                    redirect_stdout(output),
                    redirect_stderr(errors),
                ):
                    self.assertEqual(check_runtime.main([]), 1)
                self.assertIn(f"FAIL {broken}: OSError: incompatible native library", errors.getvalue())
                self.assertIn("PASS openkoasr.model.hf_ctc", output.getvalue())
                audio.assert_called_once_with()
                cuda.assert_not_called()

    def test_audio_and_required_cuda_failures_are_not_reported_as_success(self):
        for broken in ("check_audio", "check_cuda"):
            with (
                self.subTest(check=broken),
                patch.object(check_runtime.importlib, "import_module"),
                patch.object(check_runtime, "check_audio") as audio,
                patch.object(check_runtime, "check_cuda") as cuda,
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                checks = {"check_audio": audio, "check_cuda": cuda}
                checks[broken].side_effect = RuntimeError("runtime unavailable")
                self.assertEqual(check_runtime.main(["--require-cuda"]), 1)
                audio.assert_called_once_with()
                cuda.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
