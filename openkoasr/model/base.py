from abc import ABC, abstractmethod

try:
    import torch
except Exception:  # pragma: no cover - exercised in lightweight test envs.
    torch = None


class BaseASRInferenceModel(ABC):
    supports_batch_transcribe = False

    def __init__(self):
        if torch is None:
            self.TORCH_DTYPE = {
                'float32': 'float32',
                'float16': 'float16',
                'bfloat16': 'bfloat16',
            }
        else:
            self.TORCH_DTYPE = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
            }

    @abstractmethod
    def initialize_model(self):
        raise NotImplementedError

    @abstractmethod
    def initialize_processor(self):
        raise NotImplementedError

    def inference_sample(self, sample, sampling_rate):
        raise NotImplementedError

    def transcribe(self, sample, sampling_rate=16000):
        return self.inference_sample(sample, sampling_rate=sampling_rate)

    def transcribe_batch(self, samples, sampling_rates=None):
        if sampling_rates is None:
            sampling_rates = [16000] * len(samples)
        return [
            self.transcribe(sample, sampling_rate=sampling_rate)
            for sample, sampling_rate in zip(samples, sampling_rates)
        ]

    def metadata(self):
        config = getattr(self, "model_config", None)
        if config is None:
            return {}
        return {
            "name": getattr(config, "name", None),
            "family": getattr(config, "family", None),
            "repo_name": getattr(config, "repo_name", None),
            "dtype": getattr(config, "dtype", None),
            "device": getattr(config, "device", None),
        }
