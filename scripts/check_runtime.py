#!/usr/bin/env python3
"""Exercise native audio libraries and model imports without downloading weights."""

import argparse
import importlib
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_audio():
    import soundfile
    import torch
    import torchaudio

    audio = torch.sin(torch.arange(800, dtype=torch.float32) * (2 * torch.pi * 440 / 8000))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tone.wav"
        soundfile.write(path, audio.numpy(), 8000, subtype="PCM_16")
        decoded, sample_rate = torchaudio.load(path)
    if sample_rate != 8000 or decoded.shape != (1, 800):
        raise RuntimeError(f"Unexpected decoded audio: {sample_rate} Hz, {decoded.shape}")
    torch.testing.assert_close(decoded[0], audio, atol=1 / 32768, rtol=0)
    resampled = torchaudio.functional.resample(decoded, sample_rate, 16000)
    if resampled.shape != (1, 1600) or not torch.isfinite(resampled).all():
        raise RuntimeError("Audio resampling produced invalid samples")


def check_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    value = torch.ones((16, 16), device="cuda")
    torch.testing.assert_close(value @ value, torch.full_like(value, 16))
    torch.cuda.synchronize()
    print(f"GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true", help="Also require a CUDA operation")
    args = parser.parse_args(argv)
    failures = []
    modules = (
        "torch", "torchvision", "torchaudio", "torchcodec", "librosa", "soundfile",
        "datasets", "accelerate", "whisper", "qwen_asr",
        "openkoasr.model.whisper", "openkoasr.model.qwen3_asr", "openkoasr.model.hf_ctc",
    )
    checks = [(name, lambda name=name: importlib.import_module(name)) for name in modules]
    checks.append(("audio decode and resample", check_audio))
    if args.require_cuda:
        checks.append(("CUDA computation", check_cuda))

    for name, check in checks:
        try:
            result = check()
            version = getattr(result, "__version__", "")
            print(f"PASS {name} {version}".rstrip(), flush=True)
        except Exception as error:
            failures.append(name)
            print(f"FAIL {name}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    if failures:
        print(f"Runtime check failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("Runtime check passed; no model weights were downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
