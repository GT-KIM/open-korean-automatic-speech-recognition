import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openkoasr.metrics.character_error_rate import character_error_rate
from openkoasr.metrics.word_error_rate import word_error_rate
from openkoasr.normalization import normalize_text


REMOTE_ROOT = "/data/local/tmp/openkoasr-qnn-kspon"


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


def quantize(values, tensor_spec):
    params = tensor_spec["quantization_parameters"]
    scale = params["scale"]
    zero_point = params["zero_point"]
    dtype = np.dtype(tensor_spec["dtype"])
    limits = np.iinfo(dtype)
    return np.clip(np.rint(values / scale + zero_point), limits.min, limits.max).astype(
        dtype
    )


def write_native(path, values, tensor_spec):
    dtype = tensor_spec["dtype"]
    if "quantization_parameters" in tensor_spec:
        values = quantize(np.asarray(values, dtype=np.float32), tensor_spec)
    else:
        values = np.asarray(values, dtype=dtype)
    values.tofile(path)


def deploy_runtime(adb, qairt_dir, model_dir):
    bin_dir = qairt_dir / "bin" / "aarch64-android"
    lib_dir = qairt_dir / "lib" / "aarch64-android"
    dsp_dir = qairt_dir / "lib" / "hexagon-v79" / "unsigned"
    required = [
        bin_dir / "qnn-net-run",
        lib_dir / "libQnnHtp.so",
        lib_dir / "libQnnHtpV79Stub.so",
        lib_dir / "libQnnSystem.so",
        dsp_dir / "libQnnHtpV79Skel.so",
        model_dir / "encoder.bin",
        model_dir / "decoder.bin",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing runtime assets:\n" + "\n".join(missing))

    remote_dirs = " ".join(
        f"{REMOTE_ROOT}/{name}"
        for name in ("bin", "lib", "dsp", "model", "input", "decoder")
    )
    adb_shell(adb, f"rm -rf {REMOTE_ROOT}; mkdir -p {remote_dirs}")
    push(adb, bin_dir / "qnn-net-run", destination=f"{REMOTE_ROOT}/bin/")
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
    adb_shell(adb, f"chmod 755 {REMOTE_ROOT}/bin/qnn-net-run")


def qnn_command(graph, input_list, output_dir):
    return (
        f"cd {REMOTE_ROOT}; export LD_LIBRARY_PATH={REMOTE_ROOT}/lib; "
        f"export ADSP_LIBRARY_PATH={REMOTE_ROOT}/dsp; rm -rf {output_dir}; "
        f"./bin/qnn-net-run --backend ./lib/libQnnHtp.so "
        f"--retrieve_context ./model/{graph}.bin --input_list {input_list} "
        f"--use_native_input_files --use_native_output_files --output_dir {output_dir} "
        f"--perf_profile burst --num_inferences 1 --keep_num_outputs 1 --log_level error"
    )


def transcribe_sample(
    adb,
    audio,
    feature_extractor,
    tokenizer,
    config,
    metadata,
    workdir,
    forced_decoder_ids,
):
    encoder = metadata["model_files"]["encoder.bin"]
    decoder = metadata["model_files"]["decoder.bin"]
    decoder_inputs = decoder["inputs"]

    features = feature_extractor(audio, sampling_rate=16000, return_tensors="np")[
        "input_features"
    ]
    feature_path = workdir / "input_features.raw"
    write_native(feature_path, features, encoder["inputs"]["input_features"])
    encoder_list = workdir / "encoder_input_list.txt"
    encoder_list.write_text(
        f"input_features:={REMOTE_ROOT}/input/input_features.raw\n", encoding="ascii"
    )
    push(adb, feature_path, encoder_list, destination=f"{REMOTE_ROOT}/input/")
    adb_shell(
        adb,
        qnn_command(
            "encoder",
            f"{REMOTE_ROOT}/input/encoder_input_list.txt",
            f"{REMOTE_ROOT}/encoder-output",
        ),
        capture=True,
    )

    self_sources = {}
    initial_cache_paths = []
    for layer in range(config.decoder_layers):
        for prefix in ("k", "v"):
            name = f"{prefix}_cache_self_{layer}_in"
            spec = decoder_inputs[name]
            cache_path = workdir / f"{name}.raw"
            write_native(cache_path, np.zeros(spec["shape"], dtype=np.float32), spec)
            initial_cache_paths.append(cache_path)
            self_sources[name] = f"{REMOTE_ROOT}/input/{cache_path.name}"
    push(adb, *initial_cache_paths, destination=f"{REMOTE_ROOT}/input/")

    output_ids = [config.decoder_start_token_id]
    attention = np.full(decoder_inputs["attention_mask"]["shape"], -100.0, dtype=np.float32)
    position_path = workdir / "position_ids.raw"
    token_path = workdir / "input_ids.raw"
    attention_path = workdir / "attention_mask.raw"
    input_list_path = workdir / "decoder_input_list.txt"
    logits_path = workdir / "logits_native.raw"

    for step in range(199):
        np.asarray([[output_ids[-1]]], dtype=np.int32).tofile(token_path)
        np.asarray([step], dtype=np.int32).tofile(position_path)
        attention[..., 199 - step] = 0.0
        write_native(attention_path, attention, decoder_inputs["attention_mask"])

        mappings = [
            f"input_ids:={REMOTE_ROOT}/input/input_ids.raw",
            f"position_ids:={REMOTE_ROOT}/input/position_ids.raw",
        ]
        for layer in range(config.decoder_layers):
            for prefix in ("k", "v"):
                name = f"{prefix}_cache_self_{layer}_in"
                mappings.append(f"{name}:={self_sources[name]}")
            mappings.append(
                f"k_cache_cross_{layer}:={REMOTE_ROOT}/encoder-output/Result_0/"
                f"k_cache_cross_{layer}_native.raw"
            )
            mappings.append(
                f"v_cache_cross_{layer}:={REMOTE_ROOT}/encoder-output/Result_0/"
                f"v_cache_cross_{layer}_native.raw"
            )
        mappings.append(f"attention_mask:={REMOTE_ROOT}/input/attention_mask.raw")
        input_list_path.write_text(" ".join(mappings) + "\n", encoding="ascii")
        push(
            adb,
            token_path,
            position_path,
            attention_path,
            input_list_path,
            destination=f"{REMOTE_ROOT}/input/",
        )

        slot = "a" if step % 2 == 0 else "b"
        remote_output = f"{REMOTE_ROOT}/decoder/{slot}"
        adb_shell(
            adb,
            qnn_command(
                "decoder",
                f"{REMOTE_ROOT}/input/decoder_input_list.txt",
                remote_output,
            ),
            capture=True,
        )
        run(
            [
                adb,
                "pull",
                f"{remote_output}/Result_0/logits_native.raw",
                logits_path,
            ],
            capture=True,
        )
        if step < len(forced_decoder_ids):
            next_token = forced_decoder_ids[step]
        else:
            next_token = int(np.fromfile(logits_path, dtype=np.uint16).argmax())
        output_ids.append(next_token)
        print(f"decoder step={step + 1} token={next_token}", flush=True)
        if next_token == config.eos_token_id:
            break
        for layer in range(config.decoder_layers):
            for prefix in ("k", "v"):
                name = f"{prefix}_cache_self_{layer}_in"
                self_sources[name] = (
                    f"{remote_output}/Result_0/{prefix}_cache_self_{layer}_out_native.raw"
                )

    return tokenizer.decode(output_ids, skip_special_tokens=True).strip(), output_ids


def parse_samples(dataset_root, subset, indices):
    lines = (dataset_root / "KsponSpeech_scripts" / f"eval_{subset}.trn").read_text(
        encoding="utf-8"
    ).splitlines()
    samples = []
    for index in indices:
        audio_rel, reference = lines[index - 1].split(" :: ", 1)
        audio_path = dataset_root / audio_rel
        audio = np.fromfile(audio_path, dtype=np.int16).astype(np.float32) / 32768.0
        samples.append((index, audio_rel, audio, reference))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--subset", choices=("clean", "other"), default="clean")
    parser.add_argument("--indices", type=int, nargs="+", default=[2, 3, 8])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--qairt-dir", type=Path, required=True)
    parser.add_argument("--adb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="korean")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--keep-remote", action="store_true")
    args = parser.parse_args()

    metadata = json.loads((args.model_dir / "metadata.json").read_text(encoding="utf-8"))
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        "openai/whisper-small", local_files_only=True
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-small", local_files_only=True
    )
    config = WhisperConfig.from_pretrained("openai/whisper-small", local_files_only=True)
    forced_decoder_ids = [
        token_id
        for _, token_id in tokenizer.get_decoder_prompt_ids(
            language=args.language,
            task="transcribe",
            no_timestamps=True,
        )
    ]
    samples = parse_samples(args.dataset_root, args.subset, args.indices)
    deploy_runtime(args.adb, args.qairt_dir, args.model_dir)

    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="openkoasr-qnn-") as temp_dir:
            workdir = Path(temp_dir)
            for index, audio_rel, audio, reference in samples:
                started = time.perf_counter()
                prediction, token_ids = transcribe_sample(
                    args.adb,
                    audio,
                    feature_extractor,
                    tokenizer,
                    config,
                    metadata,
                    workdir,
                    forced_decoder_ids,
                )
                elapsed = time.perf_counter() - started
                normalized_reference = normalize_text(reference, preset="kspon")
                normalized_prediction = normalize_text(prediction, preset="kspon")
                wer = word_error_rate(normalized_prediction, normalized_reference)
                cer = character_error_rate(normalized_prediction, normalized_reference)
                result = {
                    "index": index,
                    "sample_id": audio_rel,
                    "audio_duration": len(audio) / 16000,
                    "processing_time": elapsed,
                    "reference": reference,
                    "prediction": prediction,
                    "normalized_reference": normalized_reference,
                    "normalized_prediction": normalized_prediction,
                    "token_ids": token_ids,
                    "wer": wer,
                    "cer": cer,
                }
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        if not args.keep_remote:
            adb_shell(args.adb, f"rm -rf {REMOTE_ROOT}")

    word_errors = sum(
        item["wer"][key]
        for item in results
        for key in ("substitutions", "deletions", "insertions")
    )
    char_errors = sum(
        item["cer"][key]
        for item in results
        for key in ("substitutions", "deletions", "insertions")
    )
    total_audio = sum(item["audio_duration"] for item in results)
    total_time = sum(item["processing_time"] for item in results)
    output_samples = results
    if not args.save_predictions:
        private_keys = {
            "reference",
            "prediction",
            "normalized_reference",
            "normalized_prediction",
            "token_ids",
        }
        output_samples = [
            {key: value for key, value in item.items() if key not in private_keys}
            for item in results
        ]
    output = {
        "scope": "validation_subset_not_leaderboard",
        "dataset": "KsponSpeech",
        "subset": args.subset,
        "indices": args.indices,
        "model": metadata["model_name"],
        "precision": metadata["precision"],
        "language": args.language,
        "forced_decoder_ids": forced_decoder_ids,
        "device": "Samsung SM-F966N / Qualcomm SM8750",
        "predictions_saved": args.save_predictions,
        "samples": output_samples,
        "aggregate": {
            "samples": len(results),
            "exact_match_samples": sum(
                item["normalized_reference"] == item["normalized_prediction"]
                for item in results
            ),
            "wer_micro": word_errors
            / sum(len(item["normalized_reference"].split()) for item in results),
            "cer_micro": char_errors
            / sum(len(item["normalized_reference"]) for item in results),
            "audio_seconds": total_audio,
            "wall_seconds": total_time,
            "wall_rtfx": total_audio / total_time,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
