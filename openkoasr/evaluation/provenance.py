"""Capture only reproducibility settings, never credentials or dataset paths."""
import hashlib
import subprocess
from importlib import metadata
from pathlib import Path


MODEL_SETTINGS = (
    "family", "repo_name", "dtype", "device", "language", "task", "max_new_tokens",
    "num_beams", "temperature", "condition_on_prev_tokens", "max_inference_batch_size",
    "provider", "api_model", "model_revision", "processor_revision",
    "num_lookahead_tokens",
)
GENERATION_SETTINGS = (
    "language", "task", "max_new_tokens", "max_length", "num_beams", "do_sample",
    "temperature", "top_k", "top_p", "use_cache", "condition_on_prev_tokens",
    "repetition_penalty", "length_penalty", "no_repeat_ngram_size", "forced_decoder_ids",
    "suppress_tokens", "begin_suppress_tokens", "eos_token_id", "pad_token_id",
    "decoder_start_token_id", "return_timestamps",
)
PACKAGES = (
    "openkoasr", "torch", "torchaudio", "transformers", "qwen-asr", "SpeechRecognition",
    "numpy", "soundfile", "jiwer", "kiwipiepy", "jamo",
)


def code_metadata():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((root / "openkoasr").rglob("*.py"),
                       key=lambda path: path.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    commit, dirty = None, None
    if (root / ".git").exists():
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
            dirty = bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--", "openkoasr"], cwd=root, text=True,
                stderr=subprocess.DEVNULL, timeout=5,
            ).strip())
        except (OSError, subprocess.SubprocessError):
            commit, dirty = None, None
    return {"commit": commit, "dirty": dirty, "source_sha256": digest.hexdigest()}


def capture_reproducibility(model, config, execution, environment):
    backend = getattr(model, "model", None)
    # Qwen's wrapper owns the Transformers model one level deeper.
    if not hasattr(backend, "config"):
        backend = getattr(backend, "model", None)
    revision = getattr(getattr(backend, "config", None), "_commit_hash", None)
    generation = getattr(backend, "generation_config", None)
    settings = {key: getattr(config, key) for key in MODEL_SETTINGS if hasattr(config, key)}
    generation_settings = {key: getattr(generation, key) for key in GENERATION_SETTINGS
                           if hasattr(generation, key)}
    overrides = {key: value for key, value in getattr(model, "generation_overrides", {}).items()
                 if key in GENERATION_SETTINGS}
    resolved = {**generation_settings, **overrides}
    inactive = []
    if resolved.get("do_sample") is False:
        inactive = [key for key in ("temperature", "top_k", "top_p") if key in resolved]
        resolved = {key: value for key, value in resolved.items() if key not in inactive}
    dtypes = {}
    if callable(getattr(backend, "parameters", None)):
        for parameter in backend.parameters():
            dtype = str(parameter.dtype).removeprefix("torch.")
            dtypes[dtype] = dtypes.get(dtype, 0) + parameter.numel()
    effective_dtype = next(iter(dtypes)) if len(dtypes) == 1 else (
        "mixed:" + "+".join(sorted(dtypes)) if dtypes else None)
    family = getattr(config, "family", None)
    local = family in {"whisper", "qwen3_asr", "hf_ctc", "nemotron_asr"}
    scope = "asr_transcribe_call" if local else (
        "api_request_reported" if family == "commercial_api" else None)
    wrapper = getattr(model, "model", None)
    inner_batch = getattr(wrapper, "max_inference_batch_size", None) if family == "qwen3_asr" else (
        execution.get("batch_size") if local and getattr(model, "supports_batch_transcribe", False) else 1)
    processor_revision = getattr(model, "processor_revision", None)
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "model_revision": revision if isinstance(revision, str) else None,
        "processor_revision": processor_revision,
        "processor_revision_source": getattr(model, "processor_revision_source", None),
        "code": code_metadata(),
        "model_config": settings,
        "generation_config": generation_settings,
        "inference": {
            "effective_dtype": effective_dtype,
            "parameter_dtype_numel": dtypes,
            "backend": "transformers" if local else getattr(config, "provider", None),
            "performance_scope": scope,
            "backend_batch_size": inner_batch,
            "batching": "backend_chunked" if family == "qwen3_asr" else (
                "native" if local and getattr(model, "supports_batch_transcribe", False) else "sequential"),
            "decoding": {
                "call_overrides": overrides, "resolved_parameters": resolved,
                "inactive_sampling_parameters": inactive,
                "transcription_options": {key: value for key, value in
                    getattr(model, "transcription_options", {}).items()
                    if key in {"language", "return_time_stamps", "mode",
                               "num_lookahead_tokens", "max_symbols_per_step"}},
            },
        },
        "execution": execution,
        "environment": {**environment, "packages": versions},
    }
