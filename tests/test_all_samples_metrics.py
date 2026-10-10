import unittest

from openkoasr.evaluation.results import AggregateResult, SampleResult


def sample(index, hits, substitutions=0, deletions=0, insertions=0, outlier=False):
    denominator = hits + substitutions + deletions
    cer = (substitutions + deletions + insertions) / denominator if denominator else float("inf")
    return SampleResult(
        index=index, sample_id=str(index), reference="", prediction="",
        normalized_reference="", normalized_prediction="",
        metrics={"cer": cer, "cer_hits": hits, "cer_substitutions": substitutions,
                 "cer_deletions": deletions, "cer_insertions": insertions,
                 "latency": 0.2},
        processing_time=0.2, audio_duration=1.0, sample_rate=16000,
        is_outlier=outlier,
    )


class AllSamplesMetricsTest(unittest.TestCase):
    def test_outliers_only_affect_the_separate_all_samples_score(self):
        normal = sample(0, 9, substitutions=1)
        # Different reference lengths ensure this is corpus CER, not mean CER.
        outlier = sample(1, 1, insertions=2, outlier=True)
        baseline = AggregateResult.from_samples([normal])
        result = AggregateResult.from_samples([normal, outlier])
        self.assertEqual(result.macro_average, baseline.macro_average)
        self.assertEqual(result.micro_average, baseline.micro_average)
        self.assertEqual(result.latency_percentiles, baseline.latency_percentiles)
        self.assertEqual((result.total_samples, result.valid_samples, result.outlier_count), (2, 1, 1))
        self.assertAlmostEqual(result.all_samples_micro_average["cer"], 3 / 11)

    def test_empty_reference_insertions_remain_in_corpus_score(self):
        result = AggregateResult.from_samples([
            sample(0, 10), sample(1, 0, insertions=3, outlier=True),
        ])
        self.assertEqual(result.macro_average["cer"], 0.0)
        self.assertEqual(result.all_samples_micro_average["cer"], 0.3)

    def test_all_outliers_and_zero_reference_denominator(self):
        result = AggregateResult.from_samples([sample(0, 1, insertions=2, outlier=True)])
        self.assertEqual(result.macro_average, {})
        self.assertEqual(result.all_samples_micro_average["cer"], 2.0)
        for samples in ([], [sample(0, 0, insertions=3, outlier=True)]):
            with self.subTest(samples=samples):
                self.assertNotIn("cer", AggregateResult.from_samples(samples).all_samples_micro_average)
