import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openkoasr.configs import get_model_config

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    from openkoasr.model.commercial_api import CommercialApiASRInferenceModel


class UnknownValueError(Exception):
    pass


class RequestError(Exception):
    pass


@unittest.skipUnless(HAS_NUMPY, "Commercial API audio conversion requires numpy")
class CommercialApiTest(unittest.TestCase):
    def setUp(self):
        temp_root = Path.cwd() / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        tmpdir = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(tmpdir.cleanup)
        self.cache_dir = Path(tmpdir.name)
        self.env_patch = patch.dict("os.environ", {
            "OPENKOASR_API_CACHE_DIR": str(self.cache_dir),
            "OPENKOASR_API_DELAY_SECONDS": "0",
            "OPENKOASR_API_BUDGET_SECONDS": "",
        })
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        config = get_model_config("google_speech_recognition")
        # Legacy configurations must not turn infrastructure failures into transcripts.
        config.empty_on_error = True
        self.model = CommercialApiASRInferenceModel(config)
        self.sample = {"audio": [0.0] * 16, "metadata": {"id": "sample-1"}}
        self.recognizer = MagicMock()
        speech_recognition = SimpleNamespace(
            Recognizer=MagicMock(return_value=self.recognizer),
            AudioFile=MagicMock(),
            UnknownValueError=UnknownValueError,
            RequestError=RequestError,
        )
        module_patch = patch.dict(sys.modules, {"speech_recognition": speech_recognition})
        module_patch.start()
        self.addCleanup(module_patch.stop)
        sleep_patch = patch("openkoasr.model.commercial_api.time.sleep")
        self.sleep = sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def test_exhausted_requests_raise_without_caching_or_recording_success(self):
        for error in (RequestError("unavailable"), OSError("connection closed")):
            with self.subTest(error=type(error).__name__):
                self.recognizer.recognize_google.reset_mock(side_effect=True)
                self.recognizer.recognize_google.side_effect = error
                self.sleep.reset_mock()
                with self.assertRaisesRegex(RuntimeError, "failed after retries") as caught:
                    self.model.transcribe(self.sample)
                self.assertIs(caught.exception.__cause__, error)
                self.assertEqual(self.recognizer.recognize_google.call_count, 5)
                self.assertEqual(self.sleep.call_count, 4)
                self.assertEqual(list(self.cache_dir.rglob("*.json")), [])

    def test_unrecognized_speech_is_a_valid_cached_empty_transcript(self):
        self.recognizer.recognize_google.side_effect = UnknownValueError()
        self.assertEqual(self.model.transcribe(self.sample), "")
        self.assertEqual(self.model.transcribe(self.sample), "")
        self.assertEqual(self.recognizer.recognize_google.call_count, 1)
        self.assertTrue((self.cache_dir / "usage.json").is_file())
        cache_path = next((self.cache_dir / self.model.provider).rglob("*.json"))
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        self.assertEqual(cached["response"], {"text": ""})
        self.assertEqual(cached["prediction"], "")

    def test_transient_request_failure_can_recover(self):
        self.recognizer.recognize_google.side_effect = [RequestError("unavailable"), "recognized"]
        self.assertEqual(self.model.transcribe(self.sample), "recognized")
        self.assertEqual(self.model.transcribe(self.sample), "recognized")
        self.assertEqual(self.recognizer.recognize_google.call_count, 2)
        self.sleep.assert_called_once_with(5)

    def test_cached_failures_and_invalid_predictions_cannot_be_scored(self):
        invalid_entries = [
            {"prediction": "", "response": {"text": "", "error": "unavailable"}},
            {"prediction": "", "response": {"text": "", "error": ""}},
            {"prediction": None, "response": {"text": None}},
            {"prediction": 42, "response": {"text": "recognized"}},
            {"response": {"text": "recognized"}},
        ]
        cache_path = self.cache_dir / "invalid.json"
        for entry in invalid_entries:
            with self.subTest(entry=entry):
                entry["request_latency"] = 0.5
                cache_path.write_text(json.dumps(entry), encoding="utf-8")
                with patch.object(self.model, "_cache_path", return_value=cache_path):
                    with self.assertRaisesRegex(RuntimeError, "Invalid API cache"):
                        self.model.transcribe(self.sample)
                self.recognizer.recognize_google.assert_not_called()
                self.assertEqual(json.loads(cache_path.read_text(encoding="utf-8")), entry)
                self.assertFalse((self.cache_dir / "usage.json").exists())

    def test_cached_latency_must_be_finite_and_positive(self):
        cache_path = self.cache_dir / "cached.json"
        for value in (None, 0, -1, True, False, float("nan"), float("inf"), "0.5"):
            with self.subTest(value=value):
                entry = {"prediction": "text", "response": {"text": "text"}}
                if value is not None:
                    entry["request_latency"] = value
                cache_path.write_text(json.dumps(entry), encoding="utf-8")
                with patch.object(self.model, "_cache_path", return_value=cache_path):
                    with self.assertRaisesRegex(RuntimeError, "Invalid API cache.*latency"):
                        self.model.transcribe(self.sample)
                self.recognizer.recognize_google.assert_not_called()
                self.assertFalse((self.cache_dir / "usage.json").exists())

    def test_valid_cache_retains_original_request_latency(self):
        cache_path = self.cache_dir / "cached.json"
        cache_path.write_text(json.dumps({"prediction": "text", "response": {"text": "text"},
                                          "request_latency": 0.5}), encoding="utf-8")
        with patch.object(self.model, "_cache_path", return_value=cache_path):
            self.assertEqual(self.model.transcribe(self.sample), "text")
        self.assertEqual(self.model.last_processing_time, 0.5)
        self.recognizer.recognize_google.assert_not_called()


if __name__ == "__main__":
    unittest.main()
