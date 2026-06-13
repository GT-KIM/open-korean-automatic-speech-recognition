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
    if (
        normalized.startswith("deepgram_")
        or normalized.startswith("assemblyai_")
        or normalized in {"google_speech_recognition", "google_web_speech"}
    ):
        return "commercial_api"
    if "whisper" in normalized:
        return "whisper"
    if "qwen3-asr" in normalized or "qwen3_asr" in normalized:
        return "qwen3_asr"
    if "wav2vec2" in normalized or "xlsr" in normalized or "ctc" in normalized:
        return "hf_ctc"
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


def _create_hf_ctc_config(model_name: str):
    config = {
        "name": model_name,
        "family": "hf_ctc",
        "repo_name": _to_repo_name(model_name),
        "dtype": "float32",
        "device": "cuda:0",
        "evaluation": {
            "metrics": [
                "wer",
                "cer",
                "mer",
                "jer",
                "ser",
                "rtfx",
                "latency",
            ]
        },
    }
    return ConfigParser(config)


def _create_commercial_api_config(model_name: str):
    normalized = model_name.lower()
    metrics = ["wer", "cer", "mer", "jer", "ser", "rtfx", "latency"]
    if normalized.startswith("deepgram_"):
        api_model = _commercial_api_model_name(normalized[len("deepgram_"):])
        config = {
            "name": model_name,
            "family": "commercial_api",
            "provider": "deepgram",
            "repo_name": f"deepgram/{api_model}",
            "api_model": api_model,
            "language": "ko",
            "env_key": "DEEPGRAM_API_KEY",
            "cache_dir": ".openkoasr_cache/api_asr",
            "min_interval_seconds": 1.0,
            "timeout_seconds": 120,
            "evaluation": {"metrics": metrics},
        }
        return ConfigParser(config)

    if normalized.startswith("assemblyai_"):
        api_model = _commercial_api_model_name(normalized[len("assemblyai_"):])
        config = {
            "name": model_name,
            "family": "commercial_api",
            "provider": "assemblyai",
            "repo_name": f"assemblyai/{api_model}",
            "api_model": api_model,
            "language": "ko",
            "env_key": "ASSEMBLYAI_API_KEY",
            "cache_dir": ".openkoasr_cache/api_asr",
            "min_interval_seconds": 1.0,
            "timeout_seconds": 120,
            "poll_interval_seconds": 3,
            "poll_timeout_seconds": 900,
            "evaluation": {"metrics": metrics},
        }
        return ConfigParser(config)

    if normalized in {"google_speech_recognition", "google_web_speech"}:
        config = {
            "name": model_name,
            "family": "commercial_api",
            "provider": "speechrecognition_google",
            "repo_name": "https://pypi.org/project/SpeechRecognition/",
            "api_model": "recognize_google",
            "language": "ko-KR",
            "cache_dir": ".openkoasr_cache/api_asr",
            "min_interval_seconds": 1.0,
            "timeout_seconds": 120,
            "empty_on_error": True,
            "evaluation": {"metrics": metrics},
        }
        return ConfigParser(config)

    raise ValueError(f"Commercial API model '{model_name}' is not supported.")


def _commercial_api_model_name(value: str):
    aliases = {
        "nova3": "nova-3",
        "nova_3": "nova-3",
        "nova2": "nova-2",
        "nova_2": "nova-2",
        "universal2": "universal-2",
        "universal_2": "universal-2",
        "universal3_pro": "universal-3-pro",
        "universal_3_pro": "universal-3-pro",
    }
    return aliases.get(value, value.replace("_", "-"))


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
    if family == "hf_ctc":
        return _create_hf_ctc_config(model_name)
    if family == "commercial_api":
        return _create_commercial_api_config(model_name)
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
