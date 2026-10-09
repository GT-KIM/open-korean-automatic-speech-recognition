"""Recover metadata from matching artifacts without modifying scores or source files."""
import hashlib
import json
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openkoasr.protocol import evaluation_protocol_id


LEGACY_RUN_IDS = {
    f"readme-legacy-{model}-kspon-clean" for model in (
        "whisper-tiny", "whisper-base", "whisper-small", "whisper-medium",
        "whisper-large-v3", "qwen3-asr-0-6b", "qwen3-asr-1-7b",
    )
}


def recover_run_metadata(row, summary_path):
    row = dict(row)
    if row.get("reproducibility"):
        return row
    if not summary_path.is_file():
        return row
    raw = summary_path.read_bytes()
    metadata = json.loads(raw).get("metadata")
    if not isinstance(metadata, dict):
        return row
    for field, source in (
        ("run_id", "run_id"), ("model", "model_name"), ("dataset", "dataset_name"),
        ("subset", "dataset_subset"), ("evaluated_samples", "evaluated_samples"),
        ("outlier_policy", "outlier_policy"),
    ):
        if metadata.get(source) != row.get(field):
            raise ValueError(f"{summary_path}: {field} does not match leaderboard row")
    normalization = metadata.get("normalization_preset")
    if row.get("normalization_preset") not in (None, normalization):
        raise ValueError(f"{summary_path}: conflicting normalization_preset")
    row["normalization_preset"] = normalization
    row["evaluation_protocol"] = evaluation_protocol_id(normalization, row.get("outlier_policy"))
    row["metadata_status"] = "recovered"
    row["reproducibility"] = {
        "model_revision": None,
        "code": {"commit": None, "dirty": None, "source_sha256": None},
        "model_config": {},
        "execution": {key: metadata.get(key) for key in (
            "batch_size", "num_workers", "warmup_samples", "limit",
        )},
        "environment": metadata.get("environment", {}),
        "evidence": {"run_id": row["run_id"], "summary_sha256": hashlib.sha256(raw).hexdigest()},
    }
    return row


def aggregate_metadata(rows):
    base = rows[0]
    for row in rows:
        for field in ("normalization_preset", "outlier_policy", "evaluation_protocol"):
            if row.get(field) != base.get(field):
                raise ValueError(f"Mixed {field} in aggregate metadata")
        if not row.get("reproducibility"):
            raise ValueError(f"{row.get('run_id')}: missing reproducibility metadata")
    return {
        "normalization_preset": base["normalization_preset"],
        "evaluation_protocol": evaluation_protocol_id(base["normalization_preset"], base["outlier_policy"]),
        "metadata_status": "aggregate",
        "reproducibility": {"source_runs": [
            {"run_id": row["run_id"], "metadata_status": row["metadata_status"],
             "reproducibility": row["reproducibility"]} for row in rows
        ]},
    }


def enrich_rows(rows):
    enriched = []
    for row in rows:
        artifact = Path(row.get("_artifact", ""))
        if artifact.name == "leaderboard_row.json" and not row.get("aggregation"):
            row = recover_run_metadata(row, artifact.parent / "summary.json")
        enriched.append(dict(row))
    by_id = {row["run_id"]: row for row in enriched}
    for row in enriched:
        if row.get("metadata_status") == "aggregate":
            continue
        sources = row.get("aggregation", {}).get("source_run_ids")
        if sources:
            try:
                source_rows = [by_id[run_id] for run_id in sources]
            except KeyError as error:
                raise ValueError(f"{row['run_id']}: missing source run {error}") from error
            if len(sources) != 4 or len(set(sources)) != 4 or {
                source.get("subset") for source in source_rows
            } != {"D01", "D02", "D03", "D04"}:
                raise ValueError(f"{row['run_id']}: expected four distinct AIHub domains")
            for source in source_rows:
                if source.get("model") != row.get("model") or source.get("dataset") != row.get("dataset"):
                    raise ValueError(f"{row['run_id']}: incompatible aggregate source")
            row.update(aggregate_metadata(source_rows))
    return enriched
