#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

if __package__:
    from .validate_leaderboard_data import validate_rows
    from .leaderboard_metadata import enrich_rows
    from .leaderboard_speed import attach_curated_speed, merge_verified_accuracy
else:
    from validate_leaderboard_data import validate_rows
    from leaderboard_metadata import enrich_rows
    from leaderboard_speed import attach_curated_speed, merge_verified_accuracy


COLUMNS = [
    ("model", "Model"),
    ("dataset", "Dataset"),
    ("subset", "Subset"),
    ("cer", "Main CER"),
    ("outliers", "Outlier rate"),
    ("all_samples_cer", "All-sample CER"),
    ("wer", "WER"),
    ("mer", "MER"),
    ("jer", "JER"),
    ("ser", "SER"),
    ("rtfx", "RTFx"),
    ("latency", "Latency"),
    ("gpu", "GPU"),
    ("run_id", "Run"),
]

LIVE_LEADERBOARD_URL = "https://gt-kim.github.io/open-korean-automatic-speech-recognition/"
STANDARD_PROTOCOL = "v1/kspon/cer>1.0"
LEADERBOARD_JSON_URL = f"{LIVE_LEADERBOARD_URL}leaderboard_data.json"
RESULT_SUBMISSION_URL = (
    "https://github.com/GT-KIM/open-korean-automatic-speech-recognition/issues/new"
    "?template=result_submission.md"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--markdown_path", default="leaderboard.md")
    parser.add_argument("--data_path", default="doc/leaderboard_data.json")
    parser.add_argument("--verified_accuracy_path", help="Completed full-accuracy validation report; replaces matching older model aliases.")
    parser.add_argument("--speed_results_path", help="Verified curated speed report to attach to compatible full-accuracy rows.")
    parser.add_argument(
        "--submitted_rows_path",
        default="doc/submitted_results.json",
        help="Optional JSON list of curated full-evaluation rows to merge into the leaderboard.",
    )
    parser.add_argument(
        "--include_partial",
        action="store_true",
        help="Include runs produced with --limit or otherwise not covering the full evaluation set.",
    )
    args = parser.parse_args()

    rows = load_rows(Path(args.results_dir), include_partial=args.include_partial)
    rows.extend(load_submitted_rows(Path(args.submitted_rows_path), include_partial=args.include_partial))
    rows = enrich_rows(rows)
    if args.verified_accuracy_path:
        # Resolve legacy fallbacks before replacing documented model aliases.
        rows = dedupe_rows(rows)
        rows = merge_verified_accuracy(rows, json.loads(Path(args.verified_accuracy_path).read_bytes()))
    if args.speed_results_path:
        import hashlib
        speed_path = Path(args.speed_results_path)
        raw = speed_path.read_bytes()
        rows = attach_curated_speed(rows, json.loads(raw), hashlib.sha256(raw).hexdigest(), speed_path.as_posix())
    rows = [sanitize_public_row(row) for row in rows]
    rows = dedupe_rows(rows)
    if not args.include_partial:
        problems = []
        validate_rows("generated leaderboard", rows, problems)
        if problems:
            raise ValueError("Leaderboard validation failed:\n" + "\n".join(problems))
    rows.sort(key=lambda row: _metric(row, "cer", default=float("inf")))

    data = json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    markdown = render_markdown(rows)
    Path(args.markdown_path).write_text(markdown, encoding="utf-8")

    data_path = Path(args.data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(data, encoding="utf-8")

    print(f"Wrote {args.markdown_path} and {args.data_path} from {len(rows)} full run(s).")


def load_rows(results_dir, include_partial=False):
    rows = []
    for path in results_dir.glob("**/leaderboard_row.json"):
        with path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        # Audited full-corpus runs enter through their completed validation report.
        # A finished subset alone must not bypass the matrix verification gate.
        if row.get("reproducibility", {}).get("full_accuracy"):
            continue
        if not include_partial and not row.get("is_full_evaluation", False):
            continue
        row["_artifact"] = str(path)
        rows.append(row)
    return rows


def load_submitted_rows(path, include_partial=False):
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Submitted rows must be a JSON list: {path}")

    filtered = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Submitted row #{index + 1} must be a JSON object: {path}")
        if not include_partial and not row.get("is_full_evaluation", False):
            continue
        row = dict(row)
        row.setdefault("_artifact", str(path))
        filtered.append(row)
    return filtered


def sanitize_public_row(row):
    row = dict(row)
    row = normalize_metric_schema(row)
    command = row.get("command")
    if isinstance(command, str) and command:
        row["command"] = sanitize_public_command(command, row.get("dataset"))
    return row


def normalize_metric_schema(row):
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return row
    metrics = dict(metrics)
    macro = metrics.get("macro")
    if isinstance(macro, dict):
        macro = dict(macro)
        rtf = macro.pop("rtf", None)
        if type(rtf) in (int, float) and math.isfinite(rtf) and rtf > 0:
            macro.setdefault("rtfx", 1 / rtf)
        elif rtf is not None:
            macro["rtf"] = rtf
        metrics["macro"] = macro
    row["metrics"] = metrics
    return row


def dedupe_rows(rows):
    selected = {}
    documented_slices = {
        (row.get("model"), row.get("dataset"), row.get("subset"))
        for row in rows if row.get("metadata_status") != "legacy"
    }
    for row in rows:
        key = (row.get("model"), row.get("dataset"), row.get("subset"))
        if row.get("metadata_status") == "legacy" and key in documented_slices:
            continue
        key += (row.get("evaluation_protocol"),)
        if None in key[:2]:
            key = (row.get("run_id"),) + key
        current = selected.get(key)
        if current is None or _row_priority(row) > _row_priority(current):
            selected[key] = row
    return list(selected.values())


def _row_priority(row):
    artifact = str(row.get("_artifact", "")).replace("\\", "/")
    generated_artifact = artifact.endswith("/leaderboard_row.json") or artifact.startswith("results/")
    return (1 if generated_artifact else 0, str(row.get("run_id", "")))


def sanitize_public_command(command, dataset=None):
    command = command.replace("\\", "/")
    command = re.sub(
        r"(?:python\s+)?\S*open-korean-automatic-speech-recognition/openkoasr/main\.py",
        "python -m openkoasr.main",
        command,
    )
    command = re.sub(
        r"(?:python\s+)?\S*openkoasr/main\.py",
        "python -m openkoasr.main",
        command,
    )

    dataset_root_placeholder = {
        "KsponSpeech": "$KSPON_ROOT",
        "AIHubLowQualityTelephone": "$AIHUB_TELEPHONE_ROOT",
    }.get(dataset, "$DATASET_ROOT")
    return re.sub(
        r"--dataset_rootpath\s+(?:(?!\s--).)+",
        f"--dataset_rootpath {dataset_root_placeholder}",
        command,
    )


def render_markdown(rows):
    lines = [
        "# OpenKoASR Leaderboard",
        "",
        (
            "These tables include only full evaluation runs generated from "
            "`results/**/leaderboard_row.json` or curated in `doc/submitted_results.json`."
        ),
        "",
        (
            f"[Live leaderboard]({LIVE_LEADERBOARD_URL}) | "
            f"[Evaluation method]({LIVE_LEADERBOARD_URL}#evaluation-method) | "
            f"[Leaderboard JSON]({LEADERBOARD_JSON_URL}) | "
            f"[Submit a result]({RESULT_SUBMISSION_URL})"
        ),
        "",
        "Error rates are shown in %, RTFx in ×, and latency in ms. JSON retains ratios and seconds.",
        "",
        "RTFx is the mean of per-sample audio duration / processing time after outlier exclusion, not total audio duration / total processing time.",
        "",
    ]
    if not rows:
        return "\n".join(lines + ["_No runs yet._", ""])

    standard, references = [], []
    for row in rows:
        policy = row.get("outlier_policy") or {}
        is_standard = (row.get("evaluation_protocol") == STANDARD_PROTOCOL
                       and row.get("normalization_preset") == "kspon"
                       and policy.get("metric") == "cer" and policy.get("threshold") == 1)
        (standard if is_standard else references).append(row)

    columns = [(key, label) for key, label in COLUMNS if key not in {"dataset", "subset"}]
    for title, description, section_rows in (
        ("Standard results", "Kspon normalization · CER > 100% excluded.", standard),
        ("Reference results", "Different or unverified evaluation protocols; excluded from standard rankings.", references),
    ):
        if not section_rows:
            continue
        lines.extend([f"## {title}", "", description, ""])
        groups = {}
        for row in section_rows:
            key = (row.get("dataset") or "Unknown dataset", row.get("subset") or "default",
                   row.get("evaluation_protocol") or "Unverified protocol")
            groups.setdefault(key, []).append(row)
        for (dataset, subset, protocol), group in sorted(groups.items()):
            heading = f"{dataset} · {subset}"
            if title == "Reference results":
                heading += f" · {protocol}"
            lines.extend([
                f"### {escape_markdown_table_cell(heading)}", "",
                "| " + " | ".join(label for _, label in columns) + " |",
                "| " + " | ".join(":--" if key in {"model", "gpu", "run_id"} else "--:" for key, _ in columns) + " |",
            ])
            for row in sorted(group, key=_markdown_sort_key):
                lines.append("| " + " | ".join(format_cell(row, key) for key, _ in columns) + " |")
            lines.append("")
    return "\n".join(lines)


def _markdown_sort_key(row):
    cer = _metric(row, "cer")
    score = cer if type(cer) in (int, float) and math.isfinite(cer) else math.inf
    return score, str(row.get("model", "")), str(row.get("run_id", ""))


def format_cell(row, key):
    if key == "model":
        return format_model_cell(row)
    if key in {"wer", "cer", "mer", "jer", "ser", "rtfx", "latency"}:
        value = _metric(row, key)
        if value is None:
            return ""
        if key == "rtfx":
            return f"{value:.2f}×"
        if key == "latency":
            return f"{value * 1000:,.1f}".rstrip("0").rstrip(".") + " ms"
        return f"{value:.2%}"
    if key == "outliers":
        count = row.get("outlier_count", 0)
        total = row.get("evaluated_samples", row.get("total_samples", 0))
        return f"{count / total:.2%} ({count} / {total})" if total else "N/A"
    if key == "all_samples_cer":
        value = row.get("metrics", {}).get("all_samples_micro", {}).get("cer")
        return "N/A" if value is None else f"{value:.2%}"
    value = row.get(key)
    return "" if value is None else escape_markdown_table_cell(value)


def format_model_cell(row):
    model = row.get("model")
    if model is None:
        return ""
    model_text = escape_markdown_table_cell(model)
    url = model_repo_url(row.get("model_repo"))
    if not url:
        return model_text
    return f"[{model_text}]({url})"


def model_repo_url(repo):
    value = str(repo or "").strip()
    if not value:
        return ""
    if re.match(r"^https?://", value, flags=re.IGNORECASE):
        return value
    if "/" not in value:
        return ""
    return "https://huggingface.co/" + "/".join(part for part in value.split("/") if part)


def escape_markdown_table_cell(value):
    return str(value).replace("|", "\\|")


def _metric(row, metric, default=None):
    macro = row.get("metrics", {}).get("macro", {})
    return macro.get(metric, default)


if __name__ == "__main__":
    main()
