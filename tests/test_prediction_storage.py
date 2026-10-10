import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openkoasr.dataset.manifest import ManifestSpeechDataset
from openkoasr.evaluation import EvaluationRunner, ResultWriter
from openkoasr.evaluation.results import SampleResult


class PredictionStorageTest(unittest.TestCase):
    def test_serialization_omits_text_without_mutating_metadata(self):
        metadata = {
            key: f"private_{key}"
            for key in (
                "text", "transcript", "sentence", "reference", "prediction",
                "normalized_reference", "normalized_prediction",
            )
        }
        metadata.update(id="sample-1", speaker="speaker-1", sample_rate=16000)
        sample = SampleResult(
            index=0,
            sample_id="sample-1",
            reference="private_reference",
            prediction="private_prediction",
            normalized_reference="private_normalized_reference",
            normalized_prediction="private_normalized_prediction",
            metrics={"wer": 0.5},
            processing_time=0.1,
            audio_duration=1.0,
            sample_rate=16000,
            is_outlier=False,
            metadata=metadata,
        )
        before = copy.deepcopy(sample.to_dict(include_predictions=True))

        sanitized = sample.to_dict(include_predictions=False)

        self.assertNotIn("private_", json.dumps(sanitized))
        self.assertEqual(
            sanitized["metadata"],
            {"id": "sample-1", "speaker": "speaker-1", "sample_rate": 16000},
        )
        self.assertEqual(sanitized["metrics"], {"wer": 0.5})
        self.assertEqual(sample.to_dict(include_predictions=True), before)

    @patch("openkoasr.evaluation.runner.logger")
    @patch.object(ManifestSpeechDataset, "_load_audio", return_value=([0.0] * 160, 16000))
    def test_manifest_artifacts_respect_save_predictions(self, _load_audio, _logger):
        temp_root = Path.cwd() / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmpdir:
            root = Path(tmpdir)
            for suffix in (".jsonl", ".csv"):
                for text_key in ("text", "transcript", "sentence"):
                    with self.subTest(suffix=suffix, text_key=text_key):
                        reference = "PRIVATE_REFERENCE"
                        prediction = "PRIVATE_PREDICTION_" * 4
                        rows = [
                            {"id": "valid", "audio_path": "valid.wav", text_key: reference,
                             "prediction": reference, "speaker": "speaker-1"},
                            {"id": "outlier", "audio_path": "outlier.wav", text_key: reference,
                             "prediction": prediction, "speaker": "speaker-2"},
                        ]
                        manifest = root / f"{text_key}{suffix}"
                        with manifest.open("w", encoding="utf-8", newline="") as output:
                            if suffix == ".jsonl":
                                for row in rows:
                                    output.write(json.dumps(row) + "\n")
                            else:
                                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                                writer.writeheader()
                                writer.writerows(rows)
                        result = EvaluationRunner.from_names(
                            dataset_name="manifest", model_name="mock",
                            manifest_path=str(manifest), normalization_preset="strict",
                        ).run()
                        self.assertEqual(result.aggregate.total_samples, 2)
                        self.assertEqual(result.aggregate.outlier_count, 1)
                        before = copy.deepcopy(result.to_dict(include_samples=True))

                        for save_predictions in (False, True):
                            paths = ResultWriter(
                                root / f"{text_key}-{suffix[1:]}-{save_predictions}",
                                save_predictions=save_predictions,
                            ).write(result)
                            for name, count in (("samples", 2), ("outliers", 1)):
                                serialized = [
                                    json.loads(line)
                                    for line in paths[name].read_text(encoding="utf-8").splitlines()
                                ]
                                self.assertEqual(len(serialized), count)
                                for sample in serialized:
                                    original = result.samples[sample["index"]]
                                    self.assertEqual(sample["metrics"], original.metrics)
                                    self.assertEqual(sample["metadata"]["speaker"], original.metadata["speaker"])
                                    if save_predictions:
                                        self.assertEqual(sample["reference"], reference)
                                        self.assertEqual(sample["prediction"], original.prediction)
                                        self.assertEqual(sample["metadata"], original.metadata)
                                    else:
                                        self.assertNotIn("PRIVATE_", json.dumps(sample))
                            self.assertEqual((paths["run_dir"] / "predictions.csv").exists(), save_predictions)
                            self.assertEqual((paths["run_dir"] / "error_analysis.jsonl").exists(), save_predictions)
                            if not save_predictions:
                                for path in paths.values():
                                    if path.is_file():
                                        self.assertNotIn("PRIVATE_", path.read_text(encoding="utf-8"))
                            self.assertEqual(result.to_dict(include_samples=True), before)


if __name__ == "__main__":
    unittest.main()
