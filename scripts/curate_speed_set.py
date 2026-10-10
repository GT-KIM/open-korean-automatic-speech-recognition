"""Freeze a model-independent speed workload from saved input metadata; no inference.

Private selection records are written under results/. Public output contains only
aggregate distributions, rules and hashes. Original audio/text must be verified
before these selections can become an executable dataset manifest.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


SET_ID = "korean-asr-speed-768-v1"
SEED = "openkoasr-speed-20261006"
BOUNDS = (0, .25, .5, .75, .9, .95, 1)
SLICES = (
    ("KsponSpeech", "clean", 3000, 256, 64),
    ("KsponSpeech", "other", 3000, 256, 64),
    ("AIHubLowQualityTelephone", "D01", 8664, 56, 12),
    ("AIHubLowQualityTelephone", "D02", 15211, 96, 24),
    ("AIHubLowQualityTelephone", "D03", 1936, 12, 4),
    ("AIHubLowQualityTelephone", "D04", 14105, 92, 24),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def identity(item):
    return f"{item['dataset']}/{item['subset']}/{item['sample_id']}"


def priority(item, purpose):
    return digest(f"{SEED}/{purpose}/{identity(item)}".encode())


def apportion(total, populations, minimum=0):
    """Largest remainder with explicit nonempty-cell minimum and capacity checks."""
    if total > sum(populations) or total < minimum * sum(p > 0 for p in populations):
        raise ValueError("Infeasible allocation")
    targets = [total * count / sum(populations) for count in populations]
    quotas = [min(count, max(minimum if count else 0, math.floor(target)))
              for count, target in zip(populations, targets)]
    while sum(quotas) < total:
        candidates = [i for i, count in enumerate(populations) if quotas[i] < count]
        i = max(candidates, key=lambda i: (targets[i] - quotas[i], -i))
        quotas[i] += 1
    while sum(quotas) > total:
        candidates = [i for i, count in enumerate(populations) if quotas[i] > (minimum if count else 0)]
        i = max(candidates, key=lambda i: (quotas[i] - targets[i], -i))
        quotas[i] -= 1
    return quotas


def input_record(sample, row):
    duration = sample["audio_duration"]
    chars = sum(sample["metrics"]["cer_" + key] for key in ("hits", "substitutions", "deletions"))
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid input duration")
    if not isinstance(chars, (int, float)) or not math.isfinite(chars) or chars < 0 or int(chars) != chars:
        raise ValueError("Invalid reference length")
    sample_id = str(sample["sample_id"])
    return {
        "dataset": row["dataset"], "subset": row["subset"],
        "source_run_id": row["run_id"], "source_index": sample["index"],
        "sample_id": sample_id, "audio_duration": duration, "reference_characters": int(chars),
        "sample_rate": sample["sample_rate"],
        "parent_group": sample_id.rsplit("/", 1)[0] if row["dataset"].startswith("AIHub") else sample_id,
    }


def load_population(root, row):
    path = (root / Path(row["_artifact"].replace("\\", "/"))).with_name("samples.jsonl")
    records = [input_record(json.loads(line), row) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len({identity(item) for item in records}) != len(records):
        raise ValueError("Duplicate source IDs")
    return records, digest(path.read_bytes())


def stratify(records):
    ordered = sorted(records, key=lambda item: (item["audio_duration"], identity(item)))
    for band, (start, end) in enumerate(zip(BOUNDS, BOUNDS[1:])):
        band_items = ordered[math.floor(start * len(ordered)):math.floor(end * len(ordered))]
        by_density = sorted(band_items, key=lambda item: (
            item["reference_characters"] / item["audio_duration"], identity(item)))
        for index, item in enumerate(by_density):
            item["duration_stratum"] = band
            item["reference_density_half"] = int(index >= len(by_density) // 2)
    return records


def select(records, size, purpose):
    bands = [[r for r in records if r["duration_stratum"] == i] for i in range(6)]
    band_quota = apportion(size, [len(band) for band in bands], minimum=1 if size >= 6 else 0)
    cells = []
    for band, quota in zip(bands, band_quota):
        halves = [[r for r in band if r["reference_density_half"] == i] for i in range(2)]
        quotas = apportion(quota, [len(half) for half in halves]) if quota else [0, 0]
        cells.extend((half, count) for half, count in zip(halves, quotas) if count)
    chosen, groups = [], set()
    # Process scarce cells first; never relax group diversity or silently change quotas.
    for candidates, count in sorted(cells, key=lambda cell: (
            len(cell[0]) / cell[1], cell[0][0]["duration_stratum"], cell[0][0]["reference_density_half"])):
        available = sorted(candidates, key=lambda item: priority(item, purpose))
        added = 0
        for item in available:
            if item["parent_group"] in groups:
                continue
            chosen.append(dict(item))
            groups.add(item["parent_group"])
            added += 1
            if added == count:
                break
        if added != count:
            raise ValueError("Cannot meet stratum quota with unique parent groups")
    if len(chosen) != size:
        raise ValueError("Selection count mismatch")
    return sorted(chosen, key=lambda item: priority(item, "order"))


def quantile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def distribution(records):
    durations = [item["audio_duration"] for item in records]
    return {
        "samples": len(records), "audio_seconds": sum(durations),
        "mean_duration_seconds": sum(durations) / len(durations),
        "duration_quantiles_seconds": {str(p): quantile(durations, p) for p in (0, .25, .5, .75, .9, .95, 1)},
        "duration_strata_counts": [sum(r["duration_stratum"] == i for r in records) for i in range(6)],
        "reference_density_half_counts": [sum(r["reference_density_half"] == i for r in records) for i in range(2)],
        "mean_reference_characters": sum(r["reference_characters"] for r in records) / len(records),
        "unique_parent_groups": len({r["parent_group"] for r in records}),
    }


def ordered_workload(records):
    groups = defaultdict(list)
    for item in records:
        group = item["subset"] if item["dataset"] == "KsponSpeech" else "telephone"
        groups[group].append(item)
    result = []
    for group in ("clean", "other", "telephone"):
        for item in sorted(groups[group], key=lambda item: priority(item, "order")):
            result.append({**item, "workload_group": group, "order": len(result)})
    return result


def write_jsonl(path, records):
    raw = b"".join(canonical_bytes(item) + b"\n" for item in records)
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("Frozen selection differs: use a new set ID and output directory")
    path.write_bytes(raw)
    return digest(raw)


def curate(root, private_dir, report_path):
    rows = json.loads((root / "doc/leaderboard_data.json").read_text(encoding="utf-8"))
    measured, pilots, evidence, checks = [], [], [], []
    for dataset, subset, expected, size, pilot_size in SLICES:
        sources = [r for r in rows if r["dataset"] == dataset and r["subset"] == subset
                   and r.get("metadata_status") == "recovered" and r.get("evaluation_protocol") == "v1/kspon/cer>1.0"
                   and not r["model"].startswith("google_")]
        if len(sources) != 8 or len({r["model_repo"] for r in sources}) != 8:
            raise ValueError("Expected eight distinct full model runs per source slice")
        row = next(r for r in sources if r["model"] == "whisper_large_v3")
        population, source_hash = load_population(root, row)
        if len(population) != expected:
            raise ValueError("Source sample count changed")
        baseline = {r["sample_id"]: r for r in population}
        for other in sources:
            observed, _ = load_population(root, other)
            if {r["sample_id"] for r in observed} != set(baseline):
                raise ValueError("Cross-model input IDs differ")
            for item in observed:
                original = baseline[item["sample_id"]]
                if (abs(item["audio_duration"] - original["audio_duration"]) > 1 / 16000
                        or item["reference_characters"] != original["reference_characters"]
                        or item["source_index"] != original["source_index"]):
                    raise ValueError("Cross-model input metadata differs")
        stratify(population)
        chosen = select(population, size, "main")
        pilot = select(chosen, pilot_size, "pilot")
        measured.extend(chosen)
        pilots.extend(pilot)
        evidence.append({"dataset": dataset, "subset": subset, "run_id": row["run_id"],
                         "samples_sha256": source_hash, "cross_checked_runs": len(sources),
                         "population": distribution(population), "selected": distribution(chosen),
                         "pilot": distribution(pilot)})
        checks.append({"dataset": dataset, "subset": subset, "unique_ids": len(chosen),
                       "pilot_nested": {identity(r) for r in pilot} <= {identity(r) for r in chosen}})
    measured, pilots = ordered_workload(measured), ordered_workload(pilots)
    private_dir.mkdir(parents=True, exist_ok=True)
    main_hash = write_jsonl(private_dir / "selection.jsonl", measured)
    pilot_hash = write_jsonl(private_dir / "pilot_selection.jsonl", pilots)
    report = {
        "id": SET_ID, "date": "2026-10-06", "seed": SEED,
        "selection_script_sha256": digest(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        "eligible_for_accuracy_leaderboard": False,
        "status": "selection_fixed_original_content_verification_pending",
        "selection_sha256": main_hash, "pilot_selection_sha256": pilot_hash,
        "audio_reference_content_sha256": None,
        "sampling": {"main_samples": 768, "pilot_samples": 192,
            "duration_quantile_boundaries": list(BOUNDS),
            "rules": [
                "Equal 256-sample weight for clean, other and telephone; 64 each in nested pilot.",
                "Telephone domain quotas follow corpus sample proportions, rounded by largest remainder in units of four.",
                "Within each domain: duration-rank strata, then lower/upper reference-character-density halves.",
                "At least one per duration stratum when quota >= 6; SHA-256 selection priority, no seed search.",
                "At most one sample per AIHub ID parent path; this is not a verified speaker-identity constraint.",
                "No CER, outlier, prediction, model rank or measured speed is used in selection.",
                "Reference length is CER hits+substitutions+deletions, cross-checked across all eight models.",
                "Hash-shuffled order within each workload group, no length bucketing; same ordered samples for batch 1 and 4.",
                "Pilot selection is nested but has its own fixed order; it is diagnostic, not the formal reported workload.",
            ]},
        "total_audio_seconds": sum(r["audio_duration"] for r in measured),
        "pilot_audio_seconds": sum(r["audio_duration"] for r in pilots),
        "main_batches_per_pass": {"batch_1": 768, "batch_4": 192},
        "pilot_batches_per_pass": {"batch_1": 192, "batch_4": 48},
        "input_evidence": evidence, "checks": checks,
        "measurement_policy": {
            "repetitions": 3, "warmup_samples_per_group": 16,
            "pilot_repetitions": 1,
            "warmup": "Use first 16 samples again before each measured group/pass; exclude warmup times, not those samples from the measured population.",
            "summary": "Per-group latency p50/p95 and throughput; median and min/max across repeats. Overall uses equal mean of the three group throughputs.",
            "coverage": "Short-utterance workload only; no per-telephone-domain p95 claim, no whole-corpus representativeness claim.",
            "process_policy": "Load a model once per track, then reuse it for all three workload groups and repeats; measure warmed inference, not cold starts.",
            "quality": "Full-corpus accuracy once under the fixed batch-4 condition, separate from curated speed measurements; retain curated CER as a diagnostic only.",
            "expansion": "Report repeat spread. If model ordering is unresolved at that spread, report unresolved; do not silently grow or reselect this set.",
        },
        "limitations": [
            "Historical dataset roots were absent at selection time. Membership and order are fixed from saved metadata, not yet audio-content-verified.",
            "Kspon IDs are historical zero-based row indices; bind to the original eval TRN before execution. Speaker diversity is unverified.",
            "Before execution, verify source membership, audio duration and normalized reference length; seal audio/reference SHA-256 and do not silently replace missing samples.",
            "Public report contains no original transcript, audio, private sample IDs or local filesystem paths.",
            "Selection files are private selection records, not directly executable ManifestSpeechDataset files.",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private_dir", type=Path, default=Path("results/speed-curation-20261006"))
    parser.add_argument("--report_path", type=Path, default=Path("doc/benchmarks/speed_curated_set_20261006.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    private_dir = args.private_dir.resolve()
    if not private_dir.is_relative_to(root / "results"):
        raise ValueError("Private selection must stay under ignored results/")
    report = curate(root, private_dir, args.report_path)
    print(json.dumps({key: report[key] for key in (
        "id", "selection_sha256", "pilot_selection_sha256", "total_audio_seconds", "pilot_audio_seconds")}, indent=2))


if __name__ == "__main__":
    main()
