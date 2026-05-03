import unittest

from openkoasr.metrics import Evaluator


class EvaluatorTest(unittest.TestCase):
    def test_metric_detail_keys_are_namespaced(self):
        evaluator = Evaluator(["wer", "cer", "ser"])
        result = evaluator.evaluate(
            sentence1="hello world",
            sentence2="hello there",
            total_processing_time=0.5,
            total_audio_length=1.0,
        )

        self.assertIn("wer", result)
        self.assertIn("wer_substitutions", result)
        self.assertIn("cer_substitutions", result)
        self.assertIn("ser_error_sentences", result)
        self.assertNotIn("hits", result)


if __name__ == "__main__":
    unittest.main()
