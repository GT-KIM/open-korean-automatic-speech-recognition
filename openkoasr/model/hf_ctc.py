import torch
from transformers import AutoModelForCTC, AutoProcessor

from openkoasr.dataset.sample import get_sample_audio
from openkoasr.model.base import BaseASRInferenceModel
from openkoasr.model.revisions import pinned_revisions, verify_loaded_revision, verify_processor_revision


class HfCtcASRInferenceModel(BaseASRInferenceModel):
    supports_batch_transcribe = True

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.processor = self.initialize_processor()
        self.model = self.initialize_model()

    def initialize_model(self):
        revision, _ = pinned_revisions(self.model_config)
        model = AutoModelForCTC.from_pretrained(
            self.model_config.repo_name,
            **({"revision": revision} if revision else {}),
            torch_dtype=self.TORCH_DTYPE[self.model_config.dtype],
        )
        verify_loaded_revision(model, revision)
        model.to(self.model_config.device)
        model.eval()
        return model

    def initialize_processor(self):
        _, revision = pinned_revisions(self.model_config)
        processor = AutoProcessor.from_pretrained(
            self.model_config.repo_name, **({"revision": revision} if revision else {}))
        self.processor_revision = revision
        self.processor_revision_source = verify_processor_revision(processor, revision)
        return processor

    def inference_sample(self, sample, sampling_rate):
        return self.transcribe_batch([sample], [sampling_rate])[0]

    def transcribe_batch(self, samples, sampling_rates=None):
        sampling_rates = sampling_rates or [16000] * len(samples)
        if len(set(int(rate) for rate in sampling_rates)) != 1:
            return super().transcribe_batch(samples, sampling_rates=sampling_rates)

        audios = [_as_processor_audio(get_sample_audio(sample)) for sample in samples]
        inputs = self.processor(
            audios,
            sampling_rate=int(sampling_rates[0]),
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(self.model_config.device)
            for key, value in inputs.items()
            if torch.is_tensor(value)
        }

        with torch.inference_mode():
            logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)


def _as_processor_audio(value):
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy()
    return value
