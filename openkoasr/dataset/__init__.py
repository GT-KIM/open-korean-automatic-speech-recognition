from openkoasr.registry import Registry
from openkoasr.dataset.aihub_telephone import AIHubLowQualityTelephoneDataset
from openkoasr.dataset.mock import MockSpeechDataset
from openkoasr.dataset.manifest import ManifestSpeechDataset

try:
    from openkoasr.dataset.KsponSpeech.dataloader import KsponSpeechDataset
except Exception:  # pragma: no cover - optional heavy deps may be absent in smoke tests.
    KsponSpeechDataset = None


dataset_registry = Registry("dataset")
dataset_registry.register("AIHubLowQualityTelephone", AIHubLowQualityTelephoneDataset)
dataset_registry.register("mock", MockSpeechDataset)
dataset_registry.register("manifest", ManifestSpeechDataset)

if KsponSpeechDataset is not None:
    dataset_registry.register("KsponSpeech", KsponSpeechDataset)

dataset_factory = dataset_registry.as_dict()

class DatasetFactory:
    @staticmethod
    def load_dataset(config):
        dataset_class = dataset_registry.get(config.name, None)
        if not dataset_class:
            if config.name == "KsponSpeech":
                raise ValueError(
                    "Dataset KsponSpeech is requested but unavailable. "
                    "Please install torch, librosa, and related audio dependencies."
                )
            raise ValueError(
                f"Dataset {config.name} is not supported. "
                f"Available datasets: {', '.join(dataset_registry.keys())}"
            )
        return dataset_class(config)
