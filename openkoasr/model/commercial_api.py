import hashlib
import io
import json
import os
import time
import wave
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from openkoasr.dataset.sample import get_audio_duration_seconds, get_sample_audio, get_sample_id
from openkoasr.model.base import BaseASRInferenceModel


class CommercialApiASRInferenceModel(BaseASRInferenceModel):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.provider = str(getattr(model_config, "provider", "")).lower()
        self.api_model = getattr(model_config, "api_model", None)
        self.language = getattr(model_config, "language", "ko")
        self.env_key = getattr(model_config, "env_key", None)
        self.api_key = os.environ.get(self.env_key or "")
        self.cache_dir = Path(
            os.environ.get(
                "OPENKOASR_API_CACHE_DIR",
                getattr(model_config, "cache_dir", ".openkoasr_cache/api_asr"),
            )
        )
        self.min_interval_seconds = float(
            os.environ.get(
                "OPENKOASR_API_DELAY_SECONDS",
                getattr(model_config, "min_interval_seconds", 0),
            )
            or 0
        )
        self.timeout_seconds = float(getattr(model_config, "timeout_seconds", 120))
        self.poll_interval_seconds = float(getattr(model_config, "poll_interval_seconds", 3))
        self.poll_timeout_seconds = float(getattr(model_config, "poll_timeout_seconds", 900))
        self.last_request_time = 0.0
        self.last_processing_time = None
        self.initialize_model()
        self.initialize_processor()

    def initialize_model(self):
        if self.provider not in {"deepgram", "assemblyai", "speechrecognition_google"}:
            raise ValueError(f"Unsupported commercial API provider: {self.provider}")
        if self.env_key and not self.api_key:
            raise RuntimeError(
                f"{self.env_key} is required to evaluate {self.model_config.name}."
            )

    def initialize_processor(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def inference_sample(self, sample, sampling_rate):
        audio = get_sample_audio(sample)
        duration = get_audio_duration_seconds(audio, sampling_rate)
        wav_bytes = _to_wav_bytes(audio, sampling_rate)
        cache_path = self._cache_path(sample, wav_bytes)
        cached = _read_json(cache_path)
        if cached is not None:
            cached_latency = cached.get("request_latency")
            if isinstance(cached_latency, (int, float)) and cached_latency > 0:
                self.last_processing_time = cached_latency
            else:
                self.last_processing_time = None
            return str(cached.get("prediction", ""))

        _assert_budget(self.cache_dir, duration)
        self._throttle()
        request_start_time = time.perf_counter()
        if self.provider == "deepgram":
            response = self._transcribe_deepgram(wav_bytes)
            prediction = _deepgram_text(response)
        elif self.provider == "assemblyai":
            response = self._transcribe_assemblyai(wav_bytes)
            prediction = str(response.get("text") or "")
        else:
            response = self._transcribe_speechrecognition_google(wav_bytes)
            prediction = str(response.get("text") or "")
        request_latency = time.perf_counter() - request_start_time
        self.last_processing_time = request_latency

        _write_json(
            cache_path,
            {
                "provider": self.provider,
                "model": self.api_model,
                "language": self.language,
                "sample_id": get_sample_id(sample, 0),
                "audio_sha256": hashlib.sha256(wav_bytes).hexdigest(),
                "audio_duration": duration,
                "request_latency": request_latency,
                "prediction": prediction,
                "response": response,
            },
        )
        _record_usage(self.cache_dir, duration)
        return prediction

    def _cache_path(self, sample, wav_bytes):
        sample_id = get_sample_id(sample, 0)
        digest = hashlib.sha256(
            f"{self.provider}:{self.api_model}:{self.language}:{sample_id}:".encode("utf-8")
            + hashlib.sha256(wav_bytes).digest()
        ).hexdigest()
        model_name = str(self.model_config.name).replace("/", "_")
        return self.cache_dir / self.provider / model_name / f"{digest}.json"

    def _throttle(self):
        if self.min_interval_seconds <= 0:
            return
        elapsed = time.monotonic() - self.last_request_time
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self.last_request_time = time.monotonic()

    def _transcribe_deepgram(self, wav_bytes):
        params = {
            "model": self.api_model or "nova-3",
            "language": self.language,
            "smart_format": str(bool(getattr(self.model_config, "smart_format", False))).lower(),
        }
        url = "https://api.deepgram.com/v1/listen?" + urlencode(params)
        return _json_request(
            url=url,
            method="POST",
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/wav",
            },
            body=wav_bytes,
            timeout=self.timeout_seconds,
        )

    def _transcribe_assemblyai(self, wav_bytes):
        upload = _json_request(
            url="https://api.assemblyai.com/v2/upload",
            method="POST",
            headers={
                "Authorization": self.api_key,
                "Content-Type": "application/octet-stream",
            },
            body=wav_bytes,
            timeout=self.timeout_seconds,
        )
        payload = {
            "audio_url": upload["upload_url"],
            "language_code": self.language,
            "speech_models": [self.api_model or "universal-2"],
            "punctuate": bool(getattr(self.model_config, "punctuate", False)),
            "format_text": bool(getattr(self.model_config, "format_text", False)),
        }
        transcript = _json_request(
            url="https://api.assemblyai.com/v2/transcript",
            method="POST",
            headers={
                "Authorization": self.api_key,
                "Content-Type": "application/json",
            },
            body=json.dumps(payload).encode("utf-8"),
            timeout=self.timeout_seconds,
        )
        transcript_id = transcript["id"]
        deadline = time.monotonic() + self.poll_timeout_seconds
        while time.monotonic() < deadline:
            result = _json_request(
                url=f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                method="GET",
                headers={"Authorization": self.api_key},
                timeout=self.timeout_seconds,
            )
            if result.get("status") == "completed":
                return result
            if result.get("status") == "error":
                raise RuntimeError(f"AssemblyAI transcript failed: {result.get('error')}")
            time.sleep(self.poll_interval_seconds)
        raise TimeoutError(f"AssemblyAI transcript did not complete: {transcript_id}")

    def _transcribe_speechrecognition_google(self, wav_bytes):
        try:
            import speech_recognition as sr
        except ImportError as error:
            raise RuntimeError(
                "SpeechRecognition is required for google_speech_recognition. "
                "Install it with `pip install SpeechRecognition`."
            ) from error

        recognizer = sr.Recognizer()
        recognizer.operation_timeout = self.timeout_seconds
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio_data = recognizer.record(source)
        errors = []
        retry_delays = (5, 15, 30, 60)
        for attempt in range(len(retry_delays) + 1):
            try:
                return {
                    "text": recognizer.recognize_google(audio_data, language=self.language)
                }
            except sr.UnknownValueError:
                return {"text": ""}
            except (OSError, sr.RequestError) as error:
                errors.append(error)
                if attempt < len(retry_delays):
                    time.sleep(retry_delays[attempt])
        if bool(getattr(self.model_config, "empty_on_error", False)):
            return {"text": "", "error": str(errors[-1])}
        raise RuntimeError(
            f"SpeechRecognition Google request failed after retries: {errors[-1]}"
        ) from errors[-1]


def _to_wav_bytes(audio, sampling_rate):
    array = _as_numpy(audio)
    if array.ndim > 1:
        array = array.mean(axis=0)
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    pcm = (np.clip(array, -1.0, 1.0) * 32767.0).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sampling_rate))
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _as_numpy(audio):
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio)


def _json_request(url, method, headers, body=None, timeout=120):
    request = Request(url=url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            data = response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request failed with HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"API request failed: {error}") from error
    return json.loads(data.decode("utf-8"))


def _deepgram_text(response):
    channels = response.get("results", {}).get("channels", [])
    if not channels:
        return ""
    alternatives = channels[0].get("alternatives", [])
    if not alternatives:
        return ""
    return str(alternatives[0].get("transcript") or "")


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _usage_path(cache_dir):
    return Path(cache_dir) / "usage.json"


def _assert_budget(cache_dir, next_duration):
    budget = os.environ.get("OPENKOASR_API_BUDGET_SECONDS")
    if not budget:
        return
    budget_seconds = float(budget)
    usage = _read_json(_usage_path(cache_dir)) or {}
    used_seconds = float(usage.get("used_seconds", 0.0))
    if used_seconds + next_duration > budget_seconds:
        raise RuntimeError(
            "OPENKOASR_API_BUDGET_SECONDS would be exceeded "
            f"({used_seconds + next_duration:.2f}s > {budget_seconds:.2f}s)."
        )


def _record_usage(cache_dir, duration):
    path = _usage_path(cache_dir)
    usage = _read_json(path) or {}
    usage["used_seconds"] = float(usage.get("used_seconds", 0.0)) + float(duration)
    usage["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(path, usage)
