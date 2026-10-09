import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError as error:
    raise unittest.SkipTest('NumPy is required for waveform sealing tests') from error

from scripts.seal_speed_inputs import read_selection, verify_sample


class SpeedInputsTest(unittest.TestCase):
    def setUp(self):
        self.row = {"sample_id": "source/one", "source_index": 3, "order": 0,
                    "sample_rate": 16000, "audio_duration": .01, "reference_characters": 5}
        self.sample = {"audio": np.zeros(160, dtype=np.float32), "sample_rate": 16000,
                       "text": "안녕 하세요", "metadata": {"id": "source/one"}}
        self.row["reference_characters"] = len(self.sample["text"])

    def test_preserves_loader_waveform_and_reference(self):
        audio, rate, text, normalized = verify_sample(self.row, self.sample)
        np.testing.assert_array_equal(audio, self.sample["audio"])
        self.assertEqual((rate, text, normalized), (16000, "안녕 하세요", "안녕 하세요"))

    def test_rejects_source_identity_duration_and_reference_changes(self):
        for change in ({"sample_id": "wrong"}, {"audio_duration": .1},
                       {"reference_characters": 10}, {"sample_rate": 8000}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify_sample({**self.row, **change}, self.sample)

    def test_rejects_nonfinite_or_nonmono_audio(self):
        for audio in (np.array([np.nan]), np.zeros((2, 160)), np.array([])):
            with self.assertRaises(ValueError):
                verify_sample(self.row, {**self.sample, "audio": audio})

    def test_selection_hash_and_order_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "selection.jsonl"
            row = {"dataset": "d", "subset": "s", "sample_id": "x", "order": 0}
            raw = (json.dumps(row) + "\n").encode()
            p.write_bytes(raw)
            self.assertEqual(read_selection(p, hashlib.sha256(raw).hexdigest()), [row])
            with self.assertRaisesRegex(ValueError, "hash"):
                read_selection(p, "0" * 64)
            raw = (json.dumps({**row, "order": 1}) + "\n").encode()
            p.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "order"):
                read_selection(p, hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    unittest.main()
