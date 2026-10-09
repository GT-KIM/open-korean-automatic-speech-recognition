#!/usr/bin/env python3
"""Bind frozen private selections to original inputs without changing their order."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def identity(row):
    return row["dataset"], row["subset"], row["sample_id"]


def read_selection(path, expected_hash):
    raw = path.read_bytes()
    if sha(raw) != expected_hash:
        raise ValueError("Frozen selection hash mismatch")
    rows = [json.loads(line) for line in raw.splitlines() if line]
    if [r["order"] for r in rows] != list(range(len(rows))):
        raise ValueError("Selection order must be contiguous")
    if len({identity(r) for r in rows}) != len(rows):
        raise ValueError("Duplicate selection identity")
    return rows


def verify_sample(row, sample):
    import numpy as np
    from openkoasr.dataset.sample import get_sample_id, get_sample_rate, get_sample_text
    from openkoasr.normalization import normalize_text

    # Inspect the loader's raw shape before the general batching helper can squeeze it.
    audio = sample["audio"] if isinstance(sample, dict) else sample[0]
    if hasattr(audio, "numpy"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype="<f4")
    rate = get_sample_rate(sample)
    text = get_sample_text(sample)
    if get_sample_id(sample, row["source_index"]) != row["sample_id"]:
        raise ValueError(f"Source ID mismatch at selection order {row['order']}")
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Expected finite, nonempty mono audio")
    if rate != row["sample_rate"] or rate != 16000:
        raise ValueError("Source sample rate mismatch")
    duration = audio.size / rate
    if abs(duration - row["audio_duration"]) > 1 / rate + 1e-12:
        raise ValueError(f"Source duration mismatch at selection order {row['order']}")
    normalized = normalize_text(text, preset="kspon")
    if len(normalized) != row["reference_characters"]:
        raise ValueError(f"Source reference length mismatch at selection order {row['order']}")
    return audio, rate, text, normalized


def seal(args):
    import numpy as np
    from openkoasr.configs import get_dataset_config
    from openkoasr.dataset import DatasetFactory
    from openkoasr.evaluation.provenance import code_metadata

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    contract = protocol["input_contract"]
    selected = read_selection(args.selection, contract["selection_manifest_sha256"])
    pilots = read_selection(args.pilot_selection, contract["pilot"]["selection_sha256"])
    if len(selected) != 768 or len(pilots) != 192:
        raise ValueError("Expected the frozen 768/192 workload")
    if Counter(r["workload_group"] for r in selected) != dict.fromkeys(("clean", "other", "telephone"), 256):
        raise ValueError("Main workload group counts changed")
    if Counter(r["workload_group"] for r in pilots) != dict.fromkeys(("clean", "other", "telephone"), 64):
        raise ValueError("Pilot workload group counts changed")
    lookup = {identity(r): r for r in selected}
    for r in pilots:
        original = lookup.get(identity(r))
        if original is None or {k: v for k, v in r.items() if k != "order"} != {
                k: v for k, v in original.items() if k != "order"}:
            raise ValueError("Pilot must be nested with identical source metadata")

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audio").mkdir()
    datasets, sources, rows = {}, {}, []
    for row in selected:
        key = row["dataset"], row["subset"]
        if key not in datasets:
            root = args.kspon_root if key[0] == "KsponSpeech" else args.telephone_root
            datasets[key] = DatasetFactory.load_dataset(get_dataset_config(
                key[0], rootpath=str(root), subset=key[1]))
            if key[0] == "KsponSpeech":
                trn = root / "KsponSpeech_scripts" / f"eval_{key[1]}.trn"
                sources["/".join(key)] = {"trn_sha256": sha(trn.read_bytes()),
                                          "source_samples": len(datasets[key])}
            else:
                sources["/".join(key)] = {"source_samples": len(datasets[key])}
        dataset = datasets[key]
        index = row["source_index"]
        sample = dataset[index]
        audio, rate, text, normalized = verify_sample(row, sample)
        if key[0] == "KsponSpeech":
            relative, loader_text = dataset.data[index]
            raw_audio = (args.kspon_root / relative).read_bytes()
            label_line = (args.kspon_root / "KsponSpeech_scripts" / f"eval_{key[1]}.trn").read_bytes().splitlines()[index]
            source = {"audio_file": relative, "label_line_index": index,
                      "audio_sha256": sha(raw_audio), "label_sha256": sha(label_line),
                      "odd_pcm_byte_trimmed": bool(len(raw_audio) % 2)}
            if loader_text != text:
                raise ValueError("Kspon loader text mismatch")
        else:
            entry = dataset.data[index]
            source = {"audio_member": entry["audio_name"], "label_member": entry["label_name"],
                      "audio_sha256": sha(dataset._zip(entry["audio_zip"]).read(entry["audio_name"])),
                      "label_sha256": sha(dataset._zip(entry["label_zip"]).read(entry["label_name"]))}
        audio_path = f"audio/{row['order']:04d}.npy"
        with (args.output / audio_path).open("xb") as stream:
            np.save(stream, audio, allow_pickle=False)
        rows.append({"selection": row, "audio_path": audio_path,
                     "audio_file_sha256": sha((args.output / audio_path).read_bytes()),
                     "waveform_sha256": sha(audio.tobytes()), "sample_rate": rate,
                     "frames": int(audio.size), "text": text,
                     "reference_sha256": sha(text.encode()),
                     "normalized_reference_sha256": sha(normalized.encode()), "source": source})
        if len(rows) % 64 == 0:
            print(f"Verified {len(rows)}/768 input identities, durations and reference lengths", flush=True)
    by_id = {identity(r["selection"]): r for r in rows}
    pilot_rows = [by_id[identity(r)] for r in pilots]
    manifests = {}
    for name, records in (("manifest.jsonl", rows), ("pilot_manifest.jsonl", pilot_rows)):
        raw = b"".join(encode(r) + b"\n" for r in records)
        (args.output / name).write_bytes(raw)
        manifests[name] = {"sha256": sha(raw), "samples": len(records)}
    report = {"schema_version": 1, "dataset_id": contract["dataset_id"], "status": "sealed",
              "selection_sha256": contract["selection_manifest_sha256"],
              "pilot_selection_sha256": contract["pilot"]["selection_sha256"],
              "manifests": manifests, "sources": sources,
              "source_code_sha256": code_metadata()["source_sha256"],
              "sealer_sha256": sha(Path(__file__).read_bytes()),
              "audio_representation": "Original loader output as lossless little-endian float32 NPY; file reading and resampling excluded from timing",
              "audio_seconds": sum(r["frames"] / r["sample_rate"] for r in rows)}
    (args.output / "seal.json").write_bytes(encode(report) + b"\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "selection", "pilot-selection", "kspon-root", "telephone-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    seal(p.parse_args())


if __name__ == "__main__":
    main()
