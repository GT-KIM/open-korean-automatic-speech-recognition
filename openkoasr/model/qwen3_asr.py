import numpy as np
import torch

from qwen_asr import Qwen3ASRModel

from openkoasr.dataset.sample import get_sample_audio
from openkoasr.model.base import BaseASRInferenceModel


class Qwen3ASRInferenceModel(BaseASRInferenceModel):
    supports_batch_transcribe = True

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.model = self.initialize_model()
        self.processor = self.initialize_processor()

    def initialize_model(self):
        max_inference_batch_size = getattr(self.model_config, "max_inference_batch_size", 1)
        max_new_tokens = getattr(self.model_config, "max_new_tokens", 256)

        model = Qwen3ASRModel.from_pretrained(
            self.model_config.repo_name,
            dtype=self.TORCH_DTYPE[self.model_config.dtype],
            device_map=self.model_config.device,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )
        return model

    def initialize_processor(self):
        # qwen-asr 패키지는 모델 객체가 내부적으로 processor를 관리합니다.
        return None

    def extract_input_features(self, sample, sample_rate):
        # qwen-asr backend는 transcribe API를 직접 사용하므로 feature 추출을 외부에 노출하지 않습니다.
        raise NotImplementedError("Qwen3-ASR does not expose raw input_features in this backend.")

    def inference_sample(self, sample, sampling_rate):
        return self.transcribe_batch([sample], [sampling_rate])[0]

    def transcribe_batch(self, samples, sampling_rates=None):
        sampling_rates = sampling_rates or [16000] * len(samples)
        audio_inputs = [
            (_as_float32_numpy(get_sample_audio(sample)), int(sampling_rate))
            for sample, sampling_rate in zip(samples, sampling_rates)
        ]

        language = getattr(self.model_config, "language", None)
        languages = [language] * len(audio_inputs) if language else None
        results = self.model.transcribe(
            audio=audio_inputs,
            language=languages,
            return_time_stamps=False,
        )

        return [result.text if result else "" for result in results]


def _as_float32_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)
