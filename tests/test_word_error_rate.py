import unittest

from openkoasr.evaluation.results import AggregateResult, SampleResult
from openkoasr.metrics import Evaluator
from openkoasr.metrics.word_error_rate import word_error_rate
from scripts.aggregate_aihub_all import _micro_average


class WordErrorRateTest(unittest.TestCase):
    def test_edit_counts_are_relative_to_reference(self):
        cases = [
            ("a b", "a b", (2, 0, 0, 0), 0.0),
            ("a", "a b", (1, 0, 1, 0), 0.5),
            ("a b", "a", (1, 0, 0, 1), 1.0),
            ("a c", "a b", (1, 1, 0, 0), 0.5),
            ("", "a b", (0, 0, 2, 0), 1.0),
            ("a b", "", (0, 0, 0, 2), float("inf")),
            ("", "", (0, 0, 0, 0), 0.0),
            ("  a\t b  ", "a b", (2, 0, 0, 0), 0.0),
            ("한국어 인식", "한국어 음성 인식", (2, 0, 1, 0), 1 / 3),
        ]
        for prediction, reference, expected_counts, expected_wer in cases:
            with self.subTest(prediction=prediction, reference=reference):
                result = word_error_rate(prediction, reference)
                self.assertEqual(
                    tuple(result[key] for key in ("hits", "substitutions", "deletions", "insertions")),
                    expected_counts,
                )
                self.assertEqual(result["wer"], expected_wer)
                self.assertEqual(
                    result["hits"] + result["substitutions"] + result["deletions"],
                    len(reference.split()),
                )
                self.assertEqual(
                    result["hits"] + result["substitutions"] + result["insertions"],
                    len(prediction.split()),
                )

    def test_live_and_saved_micro_wer_use_reference_word_count(self):
        cases = [
            ([("a", "a b")], 0.5),
            ([("a b", "a")], 1.0),
            ([("", "a b")], 1.0),
            ([("a", "a b"), ("a b", "a"), ("", "a b"), ("x", "")], 1.0),
        ]
        for pairs, expected in cases:
            with self.subTest(pairs=pairs):
                samples = [
                    SampleResult(
                        index=index,
                        sample_id=str(index),
                        reference=reference,
                        prediction=prediction,
                        normalized_reference=reference,
                        normalized_prediction=prediction,
                        metrics=Evaluator(["wer"]).evaluate(
                            sentence1=prediction, sentence2=reference
                        ),
                        processing_time=0.1,
                        audio_duration=1.0,
                        sample_rate=16000,
                        is_outlier=False,
                    )
                    for index, (prediction, reference) in enumerate(pairs)
                ]
                live = AggregateResult.from_samples(samples).micro_average
                saved = _micro_average([sample.to_dict() for sample in samples])
                self.assertAlmostEqual(live["wer"], expected)
                self.assertAlmostEqual(saved["wer"], expected)


if __name__ == "__main__":
    unittest.main()
