from openkoasr.model.base import BaseASRInferenceModel
from openkoasr.model.whisper import WhisperASRInferenceModel
try:
    from openkoasr.model.qwen3_asr import Qwen3ASRInferenceModel
except Exception:
    Qwen3ASRInferenceModel = None

__all__ = [
    "BaseASRInferenceModel",
    "WhisperASRInferenceModel",
]

if Qwen3ASRInferenceModel is not None:
    __all__.append("Qwen3ASRInferenceModel")

model_factory = {
    "whisper": WhisperASRInferenceModel,
    "whisper_tiny": WhisperASRInferenceModel,
    "whisper_base": WhisperASRInferenceModel,
    "whisper_small": WhisperASRInferenceModel,
    "whisper_medium": WhisperASRInferenceModel,
    "whisper_large": WhisperASRInferenceModel,
}

if Qwen3ASRInferenceModel is not None:
    model_factory.update({
        "qwen3_asr": Qwen3ASRInferenceModel,
        "qwen3_asr_0_6b": Qwen3ASRInferenceModel,
        "qwen3_asr_1_7b": Qwen3ASRInferenceModel,
    })

class ModelFactory:
    @staticmethod
    def load_model(config):
        family = getattr(config, "family", None)
        model_class = None

        if family is not None:
            model_class = model_factory.get(family, None)

        # Backward compatibility
        if model_class is None:
            model_class = model_factory.get(config.name, None)

        if model_class is None:
            if family == "qwen3_asr" or str(getattr(config, "name", "")).startswith("qwen3_asr"):
                raise ValueError(
                    "Qwen3-ASR model is requested but unavailable. "
                    "Please install dependency: `pip install -U qwen-asr`."
                )
            raise ValueError(
                f"Model {config.name} is not supported. "
                f"(family={family})"
            )
        return model_class(config)
