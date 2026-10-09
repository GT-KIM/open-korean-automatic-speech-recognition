import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from openkoasr.dataset.sample import get_sample_audio
from openkoasr.model.base import BaseASRInferenceModel
from openkoasr.model.revisions import pinned_revisions, verify_loaded_revision, verify_processor_revision

class WhisperASRInferenceModel(BaseASRInferenceModel):
    supports_batch_transcribe = True

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.model = self.initialize_model()
        self.processor = self.initialize_processor()

    def initialize_model(self):
        revision, _ = pinned_revisions(self.model_config)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_config.repo_name,
                                                  **({"revision": revision} if revision else {}),
                                                  dtype=self.TORCH_DTYPE[self.model_config.dtype],
                                                  low_cpu_mem_usage=True,
                                                  use_safetensors=True,
                                                  device_map='cpu')
        verify_loaded_revision(model, revision)
        model.to(self.model_config.device)
        return model

    def initialize_processor(self):
        _, revision = pinned_revisions(self.model_config)
        processor = AutoProcessor.from_pretrained(
            self.model_config.repo_name, **({"revision": revision} if revision else {}))
        self.processor_revision = revision
        self.processor_revision_source = verify_processor_revision(processor, revision)
        return processor

    def extract_input_features(self, sample, sample_rate):
        input_features = self.processor(get_sample_audio(sample), sampling_rate=sample_rate, return_tensors="pt").input_features
        input_features = input_features.to(self.model_config.device, dtype=self.TORCH_DTYPE[self.model_config.dtype])
        return input_features


    def inference_sample(self, sample, sampling_rate):
        return self.transcribe_batch([sample], [sampling_rate])[0]

    def transcribe_batch(self, samples, sampling_rates=None):
        sampling_rates = sampling_rates or [16000] * len(samples)
        if len(set(int(rate) for rate in sampling_rates)) != 1:
            return super().transcribe_batch(samples, sampling_rates=sampling_rates)

        audios = [_as_processor_audio(get_sample_audio(sample)) for sample in samples]
        input_features = self.processor(
            audios,
            sampling_rate=int(sampling_rates[0]),
            return_tensors="pt",
        ).input_features
        input_features = input_features.to(
            self.model_config.device,
            dtype=self.TORCH_DTYPE[self.model_config.dtype],
        )

        self.generation_overrides = self._generation_kwargs()
        predicted_ids = self.model.generate(
            input_features,
            attention_mask=torch.ones_like(input_features),
            **self.generation_overrides,
        )
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

    def _generation_kwargs(self):
        keys = (
            "language",
            "task",
            "max_new_tokens",
            "num_beams",
            "temperature",
            "condition_on_prev_tokens",
        )
        return {
            key: getattr(self.model_config, key)
            for key in keys
            if hasattr(self.model_config, key)
        }


def _as_processor_audio(value):
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy()
    return value
