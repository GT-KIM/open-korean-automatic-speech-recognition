import csv
import json
from pathlib import Path


class ManifestSpeechDataset:
    def __init__(self, config):
        self.config = config
        self.manifest_path = Path(getattr(config, "manifest_path", ""))
        if not self.manifest_path:
            raise ValueError("ManifestSpeechDataset requires config.manifest_path.")
        self.default_sample_rate = int(getattr(config, "sample_rate", 16000))
        self.rootpath = Path(getattr(config, "rootpath", "") or ".")
        self.data = self._load_manifest()

    def generate_dataloader(self, batch_size=1, shuffle=False, num_workers=0):
        try:
            from torch.utils.data import DataLoader
        except Exception:
            return [self[index] for index in range(len(self))]

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def _load_manifest(self):
        suffix = self.manifest_path.suffix.lower()
        if suffix == ".jsonl":
            rows = []
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        if suffix == ".csv":
            with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
        raise ValueError("Manifest must be a .jsonl or .csv file.")

    def _resolve_audio_path(self, row):
        value = row.get("audio_path") or row.get("audio_filepath") or row.get("path")
        if not value:
            raise ValueError("Manifest row requires audio_path, audio_filepath, or path.")
        path = Path(value)
        return path if path.is_absolute() else self.rootpath / path

    def _load_audio(self, path):
        try:
            import soundfile as sf

            audio, sample_rate = sf.read(str(path), dtype="float32")
            if getattr(audio, "ndim", 1) > 1:
                audio = audio.mean(axis=1)
            return audio, int(sample_rate)
        except Exception as soundfile_error:
            try:
                import librosa

                audio, sample_rate = librosa.load(str(path), sr=None, mono=True)
                return audio, int(sample_rate)
            except Exception as librosa_error:
                raise RuntimeError(
                    f"Could not load audio file {path}. Install soundfile or librosa."
                ) from librosa_error or soundfile_error

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio_path = self._resolve_audio_path(row)
        audio, sample_rate = self._load_audio(audio_path)
        text = row.get("text") or row.get("transcript") or row.get("sentence") or ""
        metadata = dict(row)
        metadata.setdefault("id", row.get("id") or audio_path.stem)
        metadata["audio_path"] = str(audio_path)
        return {
            "audio": audio,
            "text": text,
            "sample_rate": int(row.get("sample_rate") or sample_rate or self.default_sample_rate),
            "metadata": metadata,
        }
