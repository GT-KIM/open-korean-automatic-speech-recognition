import io
import wave
import zipfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
except Exception:  # pragma: no cover - optional in no-dependency smoke CI.
    np = None

from openkoasr.dataset.aihub_telephone import AIHubLowQualityTelephoneDataset


@unittest.skipIf(np is None, "numpy is required for audio decoding checks")
class AIHubTelephoneDatasetTest(unittest.TestCase):
    def test_indexes_matching_label_and_audio_paths_and_resamples(self):
        temp_root = Path.cwd() / ".tmp_tests" / f"aihub-{uuid.uuid4().hex}"
        label_dir = temp_root / "01.데이터" / "2.Validation" / "라벨링데이터_230316"
        audio_dir = temp_root / "01.데이터" / "2.Validation" / "원천데이터_230316"
        label_dir.mkdir(parents=True)
        audio_dir.mkdir(parents=True)

        with zipfile.ZipFile(label_dir / "VL_D03.zip", "w") as labels:
            labels.writestr("D03/J13/S000001/0001.txt", "o/ 여보세요.")
        with zipfile.ZipFile(audio_dir / "VS_D03.zip", "w") as audio:
            audio.writestr("D03/J13/S000001/0001.wav", _wav_bytes(sample_rate=8000))

        dataset = AIHubLowQualityTelephoneDataset(
            SimpleNamespace(rootpath=str(temp_root), subset="D03", sample_rate=16000)
        )
        sample = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(sample["text"], "o/ 여보세요.")
        self.assertEqual(sample["sample_rate"], 16000)
        self.assertEqual(sample["metadata"]["id"], "D03/J13/S000001/0001")
        self.assertGreater(len(sample["audio"]), 10)

    def test_rejects_label_without_matching_audio(self):
        temp_root = Path.cwd() / ".tmp_tests" / f"aihub-missing-{uuid.uuid4().hex}"
        label_dir = temp_root / "01.데이터" / "2.Validation" / "라벨링데이터_230316"
        audio_dir = temp_root / "01.데이터" / "2.Validation" / "원천데이터_230316"
        label_dir.mkdir(parents=True)
        audio_dir.mkdir(parents=True)

        with zipfile.ZipFile(label_dir / "VL_D03.zip", "w") as labels:
            labels.writestr("D03/J13/S000001/0001.txt", "o/ 여보세요.")
        with zipfile.ZipFile(audio_dir / "VS_D03.zip", "w") as audio:
            audio.writestr("D03/J13/S000001/0002.wav", _wav_bytes(sample_rate=8000))

        with self.assertRaises(FileNotFoundError):
            AIHubLowQualityTelephoneDataset(
                SimpleNamespace(rootpath=str(temp_root), subset="D03", sample_rate=16000)
            )


def _wav_bytes(sample_rate):
    samples = (np.sin(np.linspace(0, 0.1, sample_rate // 10)) * 12000).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
