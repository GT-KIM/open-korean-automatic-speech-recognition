from openkoasr.model.base import BaseASRInferenceModel
from openkoasr.model.mock import MockASRInferenceModel
from openkoasr.registry import Registry

try:
    from openkoasr.model.whisper import WhisperASRInferenceModel
except Exception:
    WhisperASRInferenceModel = None

try:
    from openkoasr.model.qwen3_asr import Qwen3ASRInferenceModel
except Exception:
    Qwen3ASRInferenceModel = None

__all__ = [
    "BaseASRInferenceModel",
    "MockASRInferenceModel",
]

if WhisperASRInferenceModel is not None:
    __all__.append("WhisperASRInferenceModel")
if Qwen3ASRInferenceModel is not None:
    __all__.append("Qwen3ASRInferenceModel")

model_registry = Registry("model")
model_registry.register("mock", MockASRInferenceModel)

if WhisperASRInferenceModel is not None:
    model_registry.register("whisper", WhisperASRInferenceModel)
    model_registry.register("whisper_tiny", WhisperASRInferenceModel)
    model_registry.register("whisper_base", WhisperASRInferenceModel)
    model_registry.register("whisper_small", WhisperASRInferenceModel)
    model_registry.register("whisper_medium", WhisperASRInferenceModel)
    model_registry.register("whisper_large", WhisperASRInferenceModel)

if Qwen3ASRInferenceModel is not None:
    model_registry.register("qwen3_asr", Qwen3ASRInferenceModel)
    model_registry.register("qwen3_asr_0_6b", Qwen3ASRInferenceModel)
    model_registry.register("qwen3_asr_1_7b", Qwen3ASRInferenceModel)

model_factory = model_registry.as_dict()

class ModelFactory:
    @staticmethod
    def load_model(config):
        family = getattr(config, "family", None)
        model_class = None

        if family is not None:
            model_class = model_registry.get(family, None)

        # Backward compatibility
        if model_class is None:
            model_class = model_registry.get(config.name, None)

        if model_class is None:
            if family == "qwen3_asr" or str(getattr(config, "name", "")).startswith("qwen3_asr"):
                raise ValueError(
                    "Qwen3-ASR model is requested but unavailable. "
                    "Please install dependency: `pip install -U qwen-asr`."
                )
            if family == "whisper" or str(getattr(config, "name", "")).startswith("whisper"):
                raise ValueError(
                    "Whisper model is requested but unavailable. "
                    "Please install torch and transformers dependencies."
                )
            raise ValueError(
                f"Model {config.name} is not supported. "
                f"(family={family}; available={', '.join(model_registry.keys())})"
            )
        return model_class(config)
