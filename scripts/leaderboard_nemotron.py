"""Admit the completed Nemotron batch without pooling its runtime cohort."""
from copy import deepcopy

from .leaderboard_speed import model_identity


MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
COUNTS = {"clean": 3000, "other": 3000, "D01": 8664, "D02": 15211,
          "D03": 1936, "D04": 14105, "all": 39916}
GROUPS = {"clean": "clean", "other": "other", "all": "telephone"}


def require(value, message):
    if not value:
        raise ValueError(message)


def merge_verified_nemotron(rows, report, report_sha256, artifact):
    require(report.get("status") == "completed_and_verified"
            and report.get("official_leaderboard_eligible") is True,
            "Nemotron requires completed validation and publication admission")
    new = deepcopy(report["accuracy_rows"])
    require(len(new) == 7 and {r["subset"] for r in new} == set(COUNTS),
            "Expected six full Nemotron runs and one telephone aggregate")
    proofs = report["accuracy_proofs"]
    expected = {("kspon-" + k if k in ("clean", "other") else "telephone-" + k): n
                for k, n in COUNTS.items() if k != "all"}
    require(len(proofs) == 6 and {p["dataset"]: p["samples"] for p in proofs} == expected,
            "Incomplete Nemotron accuracy proofs")
    runtime = report["runtime"]
    require(runtime["transformers"] == "5.13.0", "Nemotron speed cohort requires Transformers 5.13.0")
    by_id = {r["run_id"]: r for r in new}
    require(len(by_id) == 7, "Duplicate Nemotron run IDs")
    for row in new:
        subset = row["subset"]
        require(model_identity(row) == MODEL and row.get("is_full_evaluation") is True
                and row["dataset"] == ("KsponSpeech" if subset in ("clean", "other") else "AIHubLowQualityTelephone")
                and all(row.get(k) == COUNTS[subset] for k in ("total_samples", "evaluated_samples", "dataset_total_samples"))
                and row.get("evaluation_protocol") == "v1/kspon/cer>1.0", "Wrong Nemotron evaluation population")
        rep = row["reproducibility"]
        if subset == "all":
            sources = rep["source_runs"]
            require(len(sources) == 4 and {by_id[s["run_id"]]["subset"] for s in sources} == {"D01", "D02", "D03", "D04"},
                    "Incomplete Nemotron aggregate")
            require(all(s["reproducibility"] == by_id[s["run_id"]]["reproducibility"] for s in sources),
                    "Aggregate source provenance differs from its full runs")
            records = [s["reproducibility"] for s in sources]
        else:
            records = [rep]
        for record in records:
            evidence = record["nemotron_accuracy"]
            require(evidence["image_id"] == runtime["image_id"]
                    and evidence["source_sha256"] == runtime["source_sha256"]
                    and record["model_revision"] == record["processor_revision"] == report["model_revision"]
                    and record["inference"]["effective_dtype"] == "bfloat16"
                    and record["execution"]["batch_size"] == 4, "Nemotron runtime identity mismatch")
        row["accuracy_validation"] = {"status": "completed_and_verified", "report_id": report["id"],
                                       "report_sha256": report_sha256, "protocol_sha256": report["protocol_sha256"]}
        row["source"] = "verified Nemotron full evaluation (native Linux, BF16, B4)"
        row["prediction_diagnostics"] = deepcopy(report["prediction_diagnostics"][subset])
        row["evaluation_mode"] = "offline_full_utterance"
        row["batch_dependent_outputs"] = True
        row.pop("curated_speed", None)
        if subset not in GROUPS:
            continue
        group = GROUPS[subset]
        tracks = {}
        for name, batch in (("b1", 1), ("b4", 4)):
            speed = report["speed"][name]
            require(speed["measured_samples"] == 2304 and speed["warmups"] == 144
                    and [r["repeat"] for r in speed["repeats"]] == [1, 2, 3]
                    and all(r["groups"][group]["samples"] == 256 for r in speed["repeats"]),
                    "Incomplete Nemotron speed track")
            tracks[name] = {**deepcopy(speed["groups"][group]), "batch_size": batch,
                            "measured_generations": 768,
                            "frame_guard_advances": report["speed_frame_guard_advances"][name][group],
                            "termination": "encoder_exhaustion"}
        row["curated_speed"] = dict(status="verified", group=group, samples_per_repeat=256, repetitions=3,
            protocol_id="nemotron35-offline-bf16-20261010", executed_protocol_sha256=report["protocol_sha256"],
            report_artifact=artifact, report_sha256=report_sha256,
            environment_id="linux-3090ti-nemotron-tf5.13.0", gpu=runtime["gpu"],
            transformers=runtime["transformers"], image_id=runtime["image_id"],
            source_sha256=runtime["source_sha256"], tracks=tracks)
    key = lambda r: (model_identity(r), r.get("dataset"), r.get("subset"), r.get("evaluation_protocol"))
    replacements = {key(r) for r in new}
    return [r for r in rows if key(r) not in replacements] + new
