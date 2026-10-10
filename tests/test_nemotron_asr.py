import unittest
from types import SimpleNamespace

from openkoasr.configs import get_model_config, infer_model_family
from openkoasr.model import NemotronASRInferenceModel


class NemotronConfigTest(unittest.TestCase):
    def test_only_multilingual_revision_is_selected(self):
        config = get_model_config("nvidia/nemotron-3.5-asr-streaming-0.6b")
        self.assertEqual(config.language, "ko-KR")
        self.assertEqual(config.num_lookahead_tokens, 3)
        self.assertEqual(config.model_revision, config.processor_revision)
        self.assertIsNone(infer_model_family("nvidia/nemotron-asr-streaming-en-0.6b"))


@unittest.skipIf(NemotronASRInferenceModel is None, "requires Nemotron runtime")
class NemotronAuditTest(unittest.TestCase):
    def adapter(self, tokens, advances, lengths=(2,), prompts=(14,)):
        import torch
        model = object.__new__(NemotronASRInferenceModel)
        model.model_config = SimpleNamespace(language="ko-KR")
        model.processor = SimpleNamespace(prompt_dictionary={"ko-KR": 14},
                                          feature_extractor=SimpleNamespace(sampling_rate=16000))
        model.model = SimpleNamespace(config=SimpleNamespace(blank_token_id=9),
                                     _get_subsampling_output_length=lambda mask: torch.tensor(lengths))
        model.last_generation = (SimpleNamespace(sequences=torch.tensor(tokens), durations=torch.tensor(advances)),
                                 torch.ones(len(lengths), 2), torch.tensor(prompts))
        return model

    def test_encoder_exhaustion_with_different_batch_lengths_and_forced_advance(self):
        model = self.adapter([[9, 2, 3, 9, 0], [9, 9, 2, 9, 9]],
                             [[0, 0, 1, 1, 1], [0, 1, 0, 1, 1]], (2, 3), (14, 14))
        rows = model.audit_last_generation()["rows"]
        self.assertEqual([r["consumed_frames"] for r in rows], [2, 3])
        self.assertEqual([r["forced_frame_advances"] for r in rows], [1, 0])
        with self.assertRaisesRegex(ValueError, "No Nemotron"):
            model.audit_last_generation()

    def test_early_stop_and_wrong_language_are_rejected(self):
        for model in (self.adapter([[9, 2, 9]], [[0, 0, 1]]),
                      self.adapter([[9, 9, 9]], [[0, 1, 1]], prompts=(101,))):
            with self.assertRaises(ValueError):
                model.audit_last_generation()

    def test_invalid_batch_rates_are_rejected_before_inference(self):
        model = self.adapter([[9]], [[0]])
        for samples, rates in (([{}], []), ([{}], [8000])):
            with self.assertRaises(ValueError):
                model.transcribe_batch(samples, rates)


if __name__ == "__main__":
    unittest.main()
