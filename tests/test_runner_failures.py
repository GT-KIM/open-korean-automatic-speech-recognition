import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from openkoasr.evaluation import EvaluationRunner


@patch("openkoasr.evaluation.runner.logger")
class RunnerFailureTest(unittest.TestCase):
    def setUp(self):
        self.runner = EvaluationRunner.from_names("mock", "mock", normalization_preset="strict")
        self.samples = [
            {"audio": [0.0] * 16, "sample_rate": 16, "text": "one two",
             "metadata": {"id": f"sample-{index}"}}
            for index in range(3)
        ]

    def test_batch_count_mismatch_fails_before_scoring(self, _logger):
        for predictions in ([], ["one"], ["one", "two", "three"]):
            with self.subTest(predictions=predictions):
                evaluator = Mock()
                model = SimpleNamespace(transcribe_batch=Mock(return_value=predictions))
                with self.assertRaisesRegex(ValueError, "prediction count mismatch"):
                    self.runner._evaluate_batch(model, evaluator, self.samples[:2], 0)
                evaluator.evaluate.assert_not_called()

    def test_batch_requires_a_sequence_of_predictions(self, _logger):
        for predictions in ("ab", b"ab", None, {"a": "one", "b": "two"}):
            with self.subTest(predictions=predictions):
                evaluator = Mock()
                model = SimpleNamespace(transcribe_batch=Mock(return_value=predictions))
                with self.assertRaisesRegex(ValueError, "sequence of strings"):
                    self.runner._evaluate_batch(model, evaluator, self.samples[:2], 0)
                evaluator.evaluate.assert_not_called()

    def test_invalid_batch_item_fails_before_any_sample_is_scored(self, _logger):
        for prediction in (None, False, 0, [], {}):
            with self.subTest(prediction=prediction):
                evaluator = Mock()
                model = SimpleNamespace(transcribe_batch=Mock(return_value=["one", prediction]))
                with self.assertRaisesRegex(ValueError, "Prediction.*string"):
                    self.runner._evaluate_batch(model, evaluator, self.samples[:2], 0)
                evaluator.evaluate.assert_not_called()

    def test_single_prediction_must_be_a_string(self, _logger):
        for prediction in (None, False, 0, [], {}):
            with self.subTest(prediction=prediction):
                evaluator = Mock()
                model = SimpleNamespace(transcribe=Mock(return_value=prediction))
                with self.assertRaisesRegex(ValueError, "Prediction.*string"):
                    self.runner._evaluate_sample(model, evaluator, self.samples[0], 0)
                evaluator.evaluate.assert_not_called()

    def test_warmup_rejects_invalid_predictions(self, _logger):
        self.runner.warmup_samples = 1
        model = SimpleNamespace(transcribe=Mock(return_value=None))
        with self.assertRaisesRegex(ValueError, "Prediction.*string"):
            self.runner._warmup(model, [self.samples[:2]])
        model.transcribe.assert_called_once()

    def test_invalid_reported_processing_time_fails_before_scoring(self, _logger):
        for value in (0, -1, True, False, float("nan"), float("inf"), "0.5"):
            with self.subTest(value=value):
                evaluator = Mock()
                model = SimpleNamespace(transcribe=Mock(return_value="text"),
                                        last_processing_time=value)
                with self.assertRaisesRegex(ValueError, "Processing time"):
                    self.runner._evaluate_sample(model, evaluator, self.samples[0], 0)
                evaluator.evaluate.assert_not_called()

    def test_invalid_audio_fails_before_inference(self, _logger):
        for override in ({"audio": []}, {"sample_rate": 0}):
            for batch in (False, True):
                with self.subTest(override=override, batch=batch):
                    evaluator = Mock()
                    model = SimpleNamespace(transcribe=Mock(), transcribe_batch=Mock())
                    sample = dict(self.samples[0], **override)
                    with self.assertRaises(ValueError):
                        if batch:
                            self.runner._evaluate_batch(model, evaluator,
                                                        [self.samples[1], sample], 0)
                        else:
                            self.runner._evaluate_sample(model, evaluator, sample, 0)
                    model.transcribe.assert_not_called()
                    model.transcribe_batch.assert_not_called()
                    evaluator.evaluate.assert_not_called()

    def test_reported_processing_time_is_used_instead_of_cache_read_time(self, _logger):
        evaluator = Mock()
        evaluator.evaluate.return_value = {}
        for reported, expected in ((None, 0.25), (0.5, 0.5)):
            with self.subTest(reported=reported):
                model = SimpleNamespace(transcribe=Mock(return_value="text"),
                                        last_processing_time=reported)
                with patch("openkoasr.evaluation.runner.time.perf_counter",
                           side_effect=[10.0, 10.25]):
                    result = self.runner._evaluate_sample(model, evaluator, self.samples[0], 0)
                self.assertEqual(result.processing_time, expected)
                self.assertEqual(evaluator.evaluate.call_args.kwargs["total_processing_time"],
                                 expected)

    def test_zero_measured_single_and_batch_times_fail_before_scoring(self, _logger):
        for batch in (False, True):
            with self.subTest(batch=batch):
                evaluator = Mock()
                model = SimpleNamespace(transcribe=Mock(return_value="text"),
                                        transcribe_batch=Mock(return_value=["one", "two"]))
                with patch("openkoasr.evaluation.runner.time.perf_counter", return_value=10.0):
                    with self.assertRaisesRegex(ValueError, "Processing time"):
                        if batch:
                            self.runner._evaluate_batch(model, evaluator, self.samples[:2], 0)
                        else:
                            self.runner._evaluate_sample(model, evaluator, self.samples[0], 0)
                evaluator.evaluate.assert_not_called()

    def test_empty_transcripts_remain_valid_in_full_single_and_batch_runs(self, _logger):
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                self.runner.batch_size = batch_size
                dataset = MagicMock()
                dataset.__len__.return_value = len(self.samples)
                dataset.generate_dataloader.return_value = [
                    self.samples[index:index + batch_size]
                    for index in range(0, len(self.samples), batch_size)
                ]
                model = SimpleNamespace(
                    supports_batch_transcribe=True,
                    transcribe=Mock(return_value=""),
                    transcribe_batch=Mock(return_value=("", "")),
                )
                with patch("openkoasr.evaluation.runner.DatasetFactory.load_dataset",
                           return_value=dataset), patch(
                    "openkoasr.evaluation.runner.ModelFactory.load_model", return_value=model
                ):
                    result = self.runner.run()
                self.assertTrue(result.metadata.is_full_evaluation)
                self.assertEqual(result.aggregate.total_samples, 3)
                self.assertEqual([sample.sample_id for sample in result.samples],
                                 ["sample-0", "sample-1", "sample-2"])
                self.assertTrue(all(sample.prediction == "" for sample in result.samples))
                self.assertEqual(result.aggregate.micro_average["wer"], 1.0)
                self.assertEqual(model.transcribe_batch.call_count, batch_size - 1)


if __name__ == "__main__":
    unittest.main()
