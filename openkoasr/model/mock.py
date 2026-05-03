from openkoasr.dataset.sample import get_sample_metadata, get_sample_text
from openkoasr.model.base import BaseASRInferenceModel


class MockASRInferenceModel(BaseASRInferenceModel):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.model = self.initialize_model()
        self.processor = self.initialize_processor()

    def initialize_model(self):
        return self

    def initialize_processor(self):
        return None

    def inference_sample(self, sample, sampling_rate=16000):
        metadata = get_sample_metadata(sample)
        mode = getattr(self.model_config, "mode", "metadata_prediction")
        if mode == "echo":
            return get_sample_text(sample)
        return str(metadata.get("prediction", get_sample_text(sample)))
