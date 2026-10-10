import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

HAS_QWEN = importlib.util.find_spec("qwen_asr") is not None
if HAS_QWEN:
    from openkoasr.model.qwen3_asr import Qwen3ASRInferenceModel


@unittest.skipUnless(HAS_QWEN, "Qwen adapter requires qwen-asr")
class QwenPredictionTest(unittest.TestCase):
    def test_missing_response_is_not_an_empty_transcript(self):
        model = object.__new__(Qwen3ASRInferenceModel)
        model.model_config = SimpleNamespace(language="Korean")
        model.model = SimpleNamespace(transcribe=Mock(return_value=[None]), max_new_tokens=256)
        with self.assertRaisesRegex(ValueError, "missing.*response"):
            model.transcribe({"audio": [0.0] * 16})

        model.model.transcribe.return_value = [SimpleNamespace(text="")]
        self.assertEqual(model.transcribe({"audio": [0.0] * 16}), "")


if __name__ == "__main__":
    unittest.main()
