import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from openkoasr.dataset.sample import get_sample_audio
from openkoasr.model.base import BaseASRInferenceModel

class WhisperASRInferenceModel(BaseASRInferenceModel):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.model = self.initialize_model()
        self.processor = self.initialize_processor()

    def initialize_model(self):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_config.repo_name,
                                                  dtype=self.TORCH_DTYPE[self.model_config.dtype],
                                                  low_cpu_mem_usage=True,
                                                  use_safetensors=True,
                                                  device_map='cpu')
        model.to(self.model_config.device)
        return model

    def initialize_processor(self):
        processor = AutoProcessor.from_pretrained(self.model_config.repo_name)
        return processor

    def extract_input_features(self, sample, sample_rate):
        input_features = self.processor(get_sample_audio(sample), sampling_rate=sample_rate, return_tensors="pt").input_features
        input_features = input_features.to(self.model_config.device, dtype=self.TORCH_DTYPE[self.model_config.dtype])
        return input_features


    def inference_sample(self, sample, sampling_rate):
        input_features = self.extract_input_features(sample, sampling_rate)

        predicted_ids = self.model.generate(input_features, attention_mask=torch.ones_like(input_features), language="ko", task="transcribe")
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0]
