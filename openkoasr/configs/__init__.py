from openkoasr.configs.config_parser import ConfigParser

# Dataset Config
from openkoasr.configs.dataset.KsponSpeech import kspon_speech_config
from openkoasr.configs.dataset.aihub_telephone import aihub_telephone_config
from openkoasr.configs.dataset.mock import mock_dataset_config
from openkoasr.configs.dataset.manifest import manifest_dataset_config

# Model Config
from openkoasr.configs.model.whisper_tiny import whisper_config as whisper_tiny_config
from openkoasr.configs.model.whisper_base import whisper_config as whisper_base_config
from openkoasr.configs.model.whisper_small import whisper_config as whisper_small_config
from openkoasr.configs.model.whisper_medium import whisper_config as whisper_medium_config
from openkoasr.configs.model.qwen3_asr_0_6b import qwen3_asr_config as qwen3_asr_0_6b_config
from openkoasr.configs.model.qwen3_asr_1_7b import qwen3_asr_config as qwen3_asr_1_7b_config
from openkoasr.configs.model.mock import mock_asr_config
from copy import deepcopy

dataset_config_dict = {
    'KsponSpeech': ConfigParser(kspon_speech_config),
    'AIHubLowQualityTelephone': ConfigParser(aihub_telephone_config),
    'mock': ConfigParser(mock_dataset_config),
    'manifest': ConfigParser(manifest_dataset_config),
}

model_config_dict = {
    'whisper_tiny': ConfigParser(whisper_tiny_config),
    'whisper_base': ConfigParser(whisper_base_config),
    'whisper_small': ConfigParser(whisper_small_config),
    'whisper_medium': ConfigParser(whisper_medium_config),
    'qwen3_asr_0_6b': ConfigParser(qwen3_asr_0_6b_config),
    'qwen3_asr_1_7b': ConfigParser(qwen3_asr_1_7b_config),
    'mock': ConfigParser(mock_asr_config),
}


def infer_model_family(model_name: str):
    normalized = model_name.lower()
    if "whisper" in normalized:
        return "whisper"
    if "qwen3-asr" in normalized or "qwen3_asr" in normalized:
        return "qwen3_asr"
    if normalized == "mock" or normalized.startswith("mock_"):
        return "mock"
    return None


def _to_repo_name(model_name: str):
    # If a full HF repo id is passed, use it as-is.
    if "/" in model_name:
        return model_name

    normalized = model_name.lower().replace("_", "-")
    if normalized.startswith("whisper-"):
        return f"openai/{normalized}"
    if normalized.startswith("whisper"):
        suffix = normalized[len("whisper"):].lstrip("-")
        if suffix:
            return f"openai/whisper-{suffix}"

    if normalized.startswith("qwen3-asr-"):
        size = normalized[len("qwen3-asr-"):].replace("-", ".").upper()
        return f"Qwen/Qwen3-ASR-{size}"
    if normalized.startswith("qwen3-asr"):
        suffix = normalized[len("qwen3-asr"):].lstrip("-")
        if suffix:
            return f"Qwen/Qwen3-ASR-{suffix.replace('-', '.').upper()}"
    return model_name


def _create_whisper_config(model_name: str):
    # Reuse existing defaults for dtype/device/metrics.
    config = deepcopy(whisper_tiny_config)
    config["name"] = model_name
    config["family"] = "whisper"
    config["repo_name"] = _to_repo_name(model_name)
    return ConfigParser(config)


def _create_qwen3_asr_config(model_name: str):
    # Reuse existing defaults for dtype/device/metrics.
    config = deepcopy(qwen3_asr_0_6b_config)
    config["name"] = model_name
    config["family"] = "qwen3_asr"
    config["repo_name"] = _to_repo_name(model_name)
    return ConfigParser(config)


def get_model_config(model_name: str):
    # Backward-compatible path for predefined configs.
    if model_name in model_config_dict:
        config = model_config_dict[model_name]
        if not hasattr(config, "family"):
            config.family = infer_model_family(getattr(config, "name", model_name))
        return config

    family = infer_model_family(model_name)
    if family == "whisper":
        return _create_whisper_config(model_name)
    if family == "qwen3_asr":
        return _create_qwen3_asr_config(model_name)
    if family == "mock":
        return ConfigParser(deepcopy(mock_asr_config))

    raise ValueError(f"Model '{model_name}' is not supported (could not infer model family).")


def get_dataset_config(dataset_name: str, **overrides):
    if dataset_name not in dataset_config_dict:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")
    config = dataset_config_dict[dataset_name].to_dict()
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return ConfigParser(deepcopy(config))
