import math

from openkoasr.dataset.sample import identity_collate


class MockSpeechDataset:
    def __init__(self, config):
        self.config = config
        self.sample_rate = int(getattr(config, "sample_rate", 16000))
        self.duration_seconds = float(getattr(config, "duration_seconds", 1.0))
        self.data = list(getattr(config, "samples", []) or [])
        if not self.data:
            self.data = [
                {"id": "mock-0001", "text": "안녕하세요", "prediction": "안녕하세요"},
                {"id": "mock-0002", "text": "한국어 음성 인식", "prediction": "한국어 음성 인식"},
                {"id": "mock-0003", "text": "오픈소스 리더보드", "prediction": "오픈소스 리더보드"},
            ]

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
            collate_fn=identity_collate,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        length = max(1, int(math.ceil(self.sample_rate * self.duration_seconds)))
        return {
            "audio": [0.0] * length,
            "text": row.get("text", ""),
            "sample_rate": self.sample_rate,
            "metadata": {
                "id": row.get("id", str(index)),
                "prediction": row.get("prediction", row.get("text", "")),
                "dataset": "mock",
            },
        }
