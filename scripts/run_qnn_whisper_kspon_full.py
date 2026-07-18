import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from transformers import WhisperFeatureExtractor, WhisperTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openkoasr.metrics.character_error_rate import character_error_rate
from openkoasr.metrics.word_error_rate import word_error_rate
from openkoasr.normalization import normalize_text


REMOTE_ROOT = "/data/local/tmp/openkoasr-qnn-kspon-full"
RESULT_PREFIX = "RESULT "


def run(command, *, capture=False):
    return subprocess.run(
        [str(part) for part in command],
        check=True,
        text=True,
        capture_output=capture,
    )


def adb_shell(adb, command, *, capture=False):
    return run([adb, "shell", command], capture=capture)


def push(adb, *sources, destination):
    run([adb, "push", *sources, destination], capture=True)


def load_samples(dataset_root, subset, limit=None):
    transcript_path = dataset_root / "KsponSpeech_scripts" / f"eval_{subset}.trn"
    lines = transcript_path.read_text(encoding="utf-8").splitlines()
    if limit is not None:
        lines = lines[:limit]
    samples = []
    for index, line in enumerate(lines, start=1):
        audio_rel, reference = line.split(" :: ", 1)
        audio_path = dataset_root / audio_rel
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        samples.append((index, audio_rel, audio_path, reference))
    return transcript_path, samples


def quantize(values, tensor_spec):
    params = tensor_spec["quantization_parameters"]
    dtype = np.dtype(tensor_spec["dtype"])
    limits = np.iinfo(dtype)
    return np.clip(
        np.rint(values / params["scale"] + params["zero_point"]),
        limits.min,
        limits.max,
    ).astype(dtype)


def prepare_features(samples, feature_cache, tensor_spec):
    record_bytes = int(np.prod(tensor_spec["shape"])) * np.dtype(
        tensor_spec["dtype"]
    ).itemsize
    completed = 0
    if feature_cache.exists():
        size = feature_cache.stat().st_size
        if size % record_bytes != 0:
            raise ValueError(f"Invalid partial feature cache size: {size}")
        completed = size // record_bytes
        if completed > len(samples):
            raise ValueError("Feature cache contains more records than requested")
    if completed == len(samples):
        return record_bytes

    feature_cache.parent.mkdir(parents=True, exist_ok=True)
    extractor = WhisperFeatureExtractor.from_pretrained(
        "openai/whisper-small", local_files_only=True
    )
    with feature_cache.open("ab") as output:
        for offset, (_, _, audio_path, _) in enumerate(
            samples[completed:], start=completed + 1
        ):
            audio = np.fromfile(audio_path, dtype=np.int16).astype(np.float32) / 32768.0
            features = extractor(
                audio, sampling_rate=16000, return_tensors="np"
            )["input_features"]
            native = quantize(features, tensor_spec)
            if native.nbytes != record_bytes:
                raise ValueError(f"Unexpected feature size for {audio_path}")
            output.write(native.tobytes())
            if offset % 100 == 0 or offset == len(samples):
                output.flush()
                os.fsync(output.fileno())
                print(f"feature progress {offset}/{len(samples)}", flush=True)
    return record_bytes


def deploy(adb, qairt_dir, model_dir, runner, feature_cache):
    lib_dir = qairt_dir / "lib" / "aarch64-android"
    dsp_dir = qairt_dir / "lib" / "hexagon-v79" / "unsigned"
    assets = [
        runner,
        lib_dir / "libQnnHtp.so",
        lib_dir / "libQnnHtpV79Stub.so",
        lib_dir / "libQnnSystem.so",
        dsp_dir / "libQnnHtpV79Skel.so",
        model_dir / "encoder.bin",
        model_dir / "decoder.bin",
        feature_cache,
    ]
    missing = [str(path) for path in assets if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing deployment assets:\n" + "\n".join(missing))

    remote_dirs = " ".join(
        f"{REMOTE_ROOT}/{name}" for name in ("bin", "lib", "dsp", "model", "input")
    )
    adb_shell(adb, f"rm -rf {REMOTE_ROOT}; mkdir -p {remote_dirs}")
    push(adb, runner, destination=f"{REMOTE_ROOT}/bin/qnn-whisper-runner")
    push(
        adb,
        lib_dir / "libQnnHtp.so",
        lib_dir / "libQnnHtpV79Stub.so",
        lib_dir / "libQnnSystem.so",
        destination=f"{REMOTE_ROOT}/lib/",
    )
    push(adb, dsp_dir / "libQnnHtpV79Skel.so", destination=f"{REMOTE_ROOT}/dsp/")
    push(
        adb,
        model_dir / "encoder.bin",
        model_dir / "decoder.bin",
        destination=f"{REMOTE_ROOT}/model/",
    )
    push(adb, feature_cache, destination=f"{REMOTE_ROOT}/input/features.raw")
    adb_shell(adb, f"chmod 755 {REMOTE_ROOT}/bin/qnn-whisper-runner")


def read_cached_results(result_cache):
    results = {}
    if not result_cache.exists():
        return results
    for line in result_cache.read_text(encoding="ascii").splitlines():
        item = json.loads(line)
        results[item["index"]] = item
    expected = list(range(1, len(results) + 1))
    if sorted(results) != expected:
        raise ValueError("Result cache is not a contiguous prefix")
    return results


def run_device(adb, count, start, result_cache):
    command = (
        f"cd {REMOTE_ROOT}; export LD_LIBRARY_PATH={REMOTE_ROOT}/lib; "
        f"export ADSP_LIBRARY_PATH={REMOTE_ROOT}/dsp; "
        f"./bin/qnn-whisper-runner "
        f"--backend ./lib/libQnnHtp.so --system ./lib/libQnnSystem.so "
        f"--encoder ./model/encoder.bin --decoder ./model/decoder.bin "
        f"--features ./input/features.raw --start {start} --count {count}"
    )
    result_cache.parent.mkdir(parents=True, exist_ok=True)
    with result_cache.open("a", encoding="ascii", buffering=1) as output:
        process = subprocess.Popen(
            [str(adb), "shell", command],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if line.startswith(RESULT_PREFIX):
                item = json.loads(line[len(RESULT_PREFIX) :])
                output.write(json.dumps(item, separators=(",", ":")) + "\n")
            else:
                print(line, flush=True)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, process.args)


def score(samples, device_results, tokenizer):
    scored = []
    for index, audio_rel, audio_path, reference in samples:
        item = device_results[index]
        prediction = tokenizer.decode(item["tokens"], skip_special_tokens=True).strip()
        normalized_reference = normalize_text(reference, preset="kspon")
        normalized_prediction = normalize_text(prediction, preset="kspon")
        wer = word_error_rate(normalized_prediction, normalized_reference)
        cer = character_error_rate(normalized_prediction, normalized_reference)
        duration = audio_path.stat().st_size / 2 / 16000
        word_error_count = sum(wer[key] for key in ("substitutions", "deletions", "insertions"))
        char_error_count = sum(cer[key] for key in ("substitutions", "deletions", "insertions"))
        words = len(normalized_reference.split())
        chars = len(normalized_reference)
        is_exact = normalized_reference == normalized_prediction
        is_outlier = cer["cer"] > 1.0
        scored.append(
            {
                "index": index,
                "sample_id": audio_rel,
                "audio_duration": duration,
                "exact_match": is_exact,
                "is_outlier": is_outlier,
                "wer": wer["wer"],
                "word_errors": word_error_count,
                "reference_words": words,
                "cer": cer["cer"],
                "char_errors": char_error_count,
                "reference_chars": chars,
                "encoder_ms": item["encoder_ms"],
                "decoder_ms": item["decoder_ms"],
                "tokens_generated": len(item["tokens"]) - 1,
            }
        )
    return scored, aggregate_scores(scored)


def aggregate_scores(scored):
    valid = [item for item in scored if not item["is_outlier"]]
    if not valid:
        raise ValueError("No valid samples after outlier filtering")

    def mean(key):
        return sum(item[key] for item in valid) / len(valid)

    def percentile(values, percentage):
        return float(np.percentile(np.asarray(values, dtype=np.float64), percentage))

    valid_latencies = [
        (item["encoder_ms"] + item["decoder_ms"]) / 1000 for item in valid
    ]
    valid_rtfx = [
        item["audio_duration"] / latency
        for item, latency in zip(valid, valid_latencies, strict=True)
    ]
    word_errors = sum(item["word_errors"] for item in valid)
    reference_words = sum(item["reference_words"] for item in valid)
    char_errors = sum(item["char_errors"] for item in valid)
    reference_chars = sum(item["reference_chars"] for item in valid)

    audio_seconds = sum(item["audio_duration"] for item in scored)
    encoder_seconds = sum(item["encoder_ms"] for item in scored) / 1000
    decoder_seconds = sum(item["decoder_ms"] for item in scored) / 1000
    qnn_seconds = encoder_seconds + decoder_seconds
    return {
        "total_samples": len(scored),
        "valid_samples": len(valid),
        "outlier_count": len(scored) - len(valid),
        "exact_match_samples_all": sum(item["exact_match"] for item in scored),
        "macro_average": {
            "wer": mean("wer"),
            "cer": mean("cer"),
            "ser": 1 - sum(item["exact_match"] for item in valid) / len(valid),
            "latency": sum(valid_latencies) / len(valid_latencies),
            "rtfx": sum(valid_rtfx) / len(valid_rtfx),
        },
        "micro_average": {
            "wer": word_errors / reference_words,
            "cer": char_errors / reference_chars,
            "ser": 1 - sum(item["exact_match"] for item in valid) / len(valid),
        },
        "latency_percentiles": {
            "p50": percentile(valid_latencies, 50),
            "p90": percentile(valid_latencies, 90),
            "p95": percentile(valid_latencies, 95),
            "p99": percentile(valid_latencies, 99),
        },
        "runtime_all_samples": {
            "audio_seconds": audio_seconds,
            "encoder_qnn_seconds": encoder_seconds,
            "decoder_qnn_seconds": decoder_seconds,
            "qnn_seconds": qnn_seconds,
            "qnn_rtfx": audio_seconds / qnn_seconds,
        },
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--subset", choices=("clean", "other"), default="clean")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--qairt-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--adb", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--result-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-remote", action="store_true")
    parser.add_argument("--keep-result-cache", action="store_true")
    args = parser.parse_args()

    transcript_path, samples = load_samples(args.dataset_root, args.subset, args.limit)
    metadata = json.loads((args.model_dir / "metadata.json").read_text(encoding="utf-8"))
    tensor_spec = metadata["model_files"]["encoder.bin"]["inputs"]["input_features"]
    record_bytes = prepare_features(samples, args.feature_cache, tensor_spec)
    expected_bytes = record_bytes * len(samples)
    if args.feature_cache.stat().st_size != expected_bytes:
        raise ValueError("Feature cache did not reach expected size")

    cached = read_cached_results(args.result_cache)
    success = False
    try:
        if len(cached) < len(samples):
            deploy(args.adb, args.qairt_dir, args.model_dir, args.runner, args.feature_cache)
            run_device(args.adb, len(samples), len(cached) + 1, args.result_cache)
        device_results = read_cached_results(args.result_cache)
        if len(device_results) != len(samples):
            raise ValueError(f"Expected {len(samples)} device results, got {len(device_results)}")

        tokenizer = WhisperTokenizer.from_pretrained(
            "openai/whisper-small", local_files_only=True
        )
        scored, aggregate = score(samples, device_results, tokenizer)
        output = {
            "scope": "full_dataset" if args.limit is None else "validation_prefix",
            "dataset": "KsponSpeech",
            "subset": args.subset,
            "transcript_sha256": sha256(transcript_path),
            "model": metadata["model_name"],
            "precision": metadata["precision"],
            "language": "korean",
            "forced_decoder_ids": [50264, 50359, 50363],
            "device": "Samsung SM-F966N / Qualcomm SM8750",
            "qairt_version": "2.47.0.260601",
            "predictions_saved": False,
            "outlier_policy": {"metric": "cer", "threshold": 1.0},
            "samples": scored,
            "aggregate": aggregate,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(aggregate, ensure_ascii=False), flush=True)
        success = True
    finally:
        if success and not args.keep_remote:
            adb_shell(args.adb, f"rm -rf {REMOTE_ROOT}")
        if success and not args.keep_result_cache:
            args.result_cache.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
