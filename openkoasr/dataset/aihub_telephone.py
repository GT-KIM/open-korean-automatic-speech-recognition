import io
import zipfile
from pathlib import Path


class AIHubLowQualityTelephoneDataset:
    """AIHub low-quality telephone ASR validation split reader.

    The public package stores validation labels and wav files in paired zip
    files named VL_Dxx.zip and VS_Dxx.zip. This dataset reads the zips lazily
    so the multi-GB validation archives do not need to be extracted first.
    """

    def __init__(self, config):
        self.config = config
        self.rootpath = Path(getattr(config, "rootpath", ""))
        self.subset = str(getattr(config, "subset", "D03"))
        self.sample_rate = int(getattr(config, "sample_rate", 16000))
        self.label_zip_dir = self.rootpath / "01.데이터" / "2.Validation" / "라벨링데이터_230316"
        self.audio_zip_dir = self.rootpath / "01.데이터" / "2.Validation" / "원천데이터_230316"
        self.domain_ids = self._domain_ids()
        self.data = self._index_data()
        self._zip_cache = {}

    def _domain_ids(self):
        if self.subset.lower() in {"all", "validation", "valid"}:
            return ["D01", "D02", "D03", "D04"]
        return [self.subset.upper()]

    def _index_data(self):
        rows = []
        for domain_id in self.domain_ids:
            label_zip = self.label_zip_dir / f"VL_{domain_id}.zip"
            audio_zip = self.audio_zip_dir / f"VS_{domain_id}.zip"
            if not label_zip.exists():
                raise FileNotFoundError(f"Missing label zip: {label_zip}")
            if not audio_zip.exists():
                raise FileNotFoundError(f"Missing audio zip: {audio_zip}")

            with zipfile.ZipFile(label_zip) as labels:
                label_names = sorted(name for name in labels.namelist() if name.endswith(".txt"))
                for label_name in label_names:
                    audio_name = label_name[:-4] + ".wav"
                    rows.append(
                        {
                            "domain_id": domain_id,
                            "label_zip": str(label_zip),
                            "audio_zip": str(audio_zip),
                            "label_name": label_name,
                            "audio_name": audio_name,
                            "id": f"{domain_id}/{audio_name[:-4]}",
                        }
                    )
        return rows

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        labels = self._zip(row["label_zip"])
        text = labels.read(row["label_name"]).decode("utf-8", errors="replace").strip()
        audio_zip = self._zip(row["audio_zip"])
        audio_bytes = audio_zip.read(row["audio_name"])
        audio, sample_rate = self._load_audio(audio_bytes)

        return {
            "audio": audio,
            "text": text,
            "sample_rate": sample_rate or self.sample_rate,
            "metadata": {
                "id": row["id"],
                "domain_id": row["domain_id"],
                "label_name": row["label_name"],
                "audio_name": row["audio_name"],
                "dataset": "AIHubLowQualityTelephone",
            },
        }

    def _zip(self, path):
        if path not in self._zip_cache:
            self._zip_cache[path] = zipfile.ZipFile(path)
        return self._zip_cache[path]

    def _load_audio(self, audio_bytes):
        try:
            import soundfile as sf

            audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if getattr(audio, "ndim", 1) > 1:
                audio = audio.mean(axis=1)
            return self._resample_if_needed(audio, int(sample_rate))
        except Exception as soundfile_error:
            try:
                import librosa

                audio, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
                return self._resample_if_needed(audio, int(sample_rate))
            except Exception as librosa_error:
                raise RuntimeError("Could not decode wav bytes from telephone dataset.") from (
                    librosa_error or soundfile_error
                )

    def _resample_if_needed(self, audio, sample_rate):
        if sample_rate == self.sample_rate:
            return audio, sample_rate
        import librosa

        return librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate), self.sample_rate
