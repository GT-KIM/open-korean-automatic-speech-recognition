import csv
import importlib.util
import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openkoasr.dataset.manifest import ManifestSpeechDataset
from openkoasr.dataset.sample import get_audio_duration_seconds, get_sample_rate


class AudioMetadataTest(unittest.TestCase):
    def setUp(self):
        temp_root = Path.cwd() / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        tmpdir = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(tmpdir.cleanup)
        self.root = Path(tmpdir.name)

    def dataset(self, row, suffix):
        path = self.root / f"manifest{suffix}"
        with path.open("w", encoding="utf-8", newline="") as handle:
            if suffix == ".jsonl":
                handle.write(json.dumps(row) + "\n")
            else:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
        return ManifestSpeechDataset(SimpleNamespace(
            manifest_path=str(path), rootpath=str(self.root), sample_rate=16000,
        ))

    @patch.object(ManifestSpeechDataset, "_load_audio", return_value=([0.0] * 800, 8000))
    def test_manifest_cannot_relabel_audio_without_resampling(self, _load_audio):
        for suffix in (".jsonl", ".csv"):
            with self.subTest(suffix=suffix):
                dataset = self.dataset({"audio_path": "clip.wav", "text": "test",
                                        "sample_rate": 16000}, suffix)
                with self.assertRaisesRegex(ValueError, "sample_rate.*16000.*8000"):
                    dataset[0]

    @patch.object(ManifestSpeechDataset, "_load_audio", return_value=([0.0] * 800, 8000))
    def test_matching_and_optional_rates_preserve_decoded_duration(self, _load_audio):
        for suffix in (".jsonl", ".csv"):
            for declared in (8000, None, ""):
                with self.subTest(suffix=suffix, declared=declared):
                    dataset = self.dataset({"audio_path": "clip.wav", "text": "test",
                                            "sample_rate": declared}, suffix)
                    sample = dataset[0]
                    self.assertEqual(sample["sample_rate"], 8000)
                    self.assertEqual(len(sample["audio"]), 800)
                    self.assertEqual(get_audio_duration_seconds(
                        sample["audio"], sample["sample_rate"]), 0.1)

    @patch.object(ManifestSpeechDataset, "_load_audio", return_value=([0.0] * 800, 8000))
    def test_manifest_rejects_invalid_declared_rates(self, _load_audio):
        for suffix in (".jsonl", ".csv"):
            for declared in (0, -8000, True, False, 8000.5, "invalid"):
                with self.subTest(suffix=suffix, declared=declared):
                    dataset = self.dataset({"audio_path": "clip.wav", "text": "test",
                                            "sample_rate": declared}, suffix)
                    with self.assertRaisesRegex(ValueError, "sample_rate"):
                        dataset[0]

    def test_sample_rate_does_not_hide_invalid_values_with_defaults(self):
        for value in (None, 0, -1, True, False, 8000.5, float("nan"), float("inf"), ""):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "sample_rate"):
                    get_sample_rate({"sample_rate": value})
        self.assertEqual(get_sample_rate({}), 16000)
        self.assertEqual(get_sample_rate(([0.0], "text")), 16000)
        for value in (8000, 8000.0, "8000", [8000]):
            self.assertEqual(get_sample_rate({"sample_rate": value}), 8000)

    def test_duration_rejects_empty_audio_and_invalid_rates(self):
        with self.assertRaisesRegex(ValueError, "Audio.*sample"):
            get_audio_duration_seconds([], 16000)
        for value in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "sample_rate"):
                    get_audio_duration_seconds([0.0] * 800, value)

    @unittest.skipUnless(importlib.util.find_spec("soundfile"), "WAV decoder requires soundfile")
    def test_real_wav_header_controls_duration(self):
        path = self.root / "clip.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(b"\x00\x00" * 800)
        row = {"audio_path": path.name, "text": "test"}
        sample = self.dataset(row, ".jsonl")[0]
        self.assertEqual(sample["sample_rate"], 8000)
        self.assertEqual(get_audio_duration_seconds(sample["audio"], sample["sample_rate"]), 0.1)
        with self.assertRaisesRegex(ValueError, "sample_rate.*16000.*8000"):
            self.dataset(dict(row, sample_rate=16000), ".jsonl")[0]


if __name__ == "__main__":
    unittest.main()
