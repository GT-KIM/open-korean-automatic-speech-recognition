#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


COLUMNS = [
    ("model", "Model"),
    ("dataset", "Dataset"),
    ("subset", "Subset"),
    ("wer", "WER"),
    ("cer", "CER"),
    ("mer", "MER"),
    ("jer", "JER"),
    ("ser", "SER"),
    ("rtfx", "RTFx"),
    ("latency", "Latency(s)"),
    ("gpu", "GPU"),
    ("outliers", "Outliers"),
    ("run_id", "Run"),
]

LIVE_LEADERBOARD_URL = "https://gt-kim.github.io/open-korean-automatic-speech-recognition/"
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
    rows = [sanitize_public_row(row) for row in rows]
    rows = dedupe_rows(rows)
    rows.sort(key=lambda row: _metric(row, "cer", default=float("inf")))

    markdown = render_markdown(rows)
    Path(args.markdown_path).write_text(markdown, encoding="utf-8")

    data_path = Path(args.data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.markdown_path} and {args.data_path} from {len(rows)} full run(s).")


def load_rows(results_dir, include_partial=False):
    rows = []
    for path in results_dir.glob("**/leaderboard_row.json"):
        with path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
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
    if command:
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
        if "rtfx" not in macro and isinstance(rtf, (int, float)) and rtf > 0:
            macro["rtfx"] = 1 / rtf
        metrics["macro"] = macro
    row["metrics"] = metrics
    return row


def dedupe_rows(rows):
    selected = {}
    for row in rows:
        key = (row.get("model"), row.get("dataset"), row.get("subset"))
        if None in key[:2]:
            key = (row.get("run_id"), row.get("model"), row.get("dataset"), row.get("subset"))
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
            "This table includes only full evaluation runs generated from "
            "`results/**/leaderboard_row.json` or curated in `doc/submitted_results.json`."
        ),
        "",
        (
            f"[Live leaderboard]({LIVE_LEADERBOARD_URL}) | "
            f"[Leaderboard JSON]({LEADERBOARD_JSON_URL}) | "
            f"[Submit a result]({RESULT_SUBMISSION_URL})"
        ),
        "",
        "| " + " | ".join(label for _, label in COLUMNS) + " |",
        "| " + " | ".join(":--" if key in {"model", "dataset", "subset", "gpu", "run_id"} else "--:" for key, _ in COLUMNS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row, key) for key, _ in COLUMNS) + " |")
    if not rows:
        lines.append("| _No runs yet_ |  |  |  |  |  |  |  |  |  |  |  |  |")
    lines.append("")
    return "\n".join(lines)


def format_cell(row, key):
    if key == "model":
        return format_model_cell(row)
    if key in {"wer", "cer", "mer", "jer", "ser", "rtfx", "latency"}:
        value = _metric(row, key)
        return "" if value is None else f"{value:.4f}"
    if key == "outliers":
        return f"{row.get('outlier_count', 0)} / {row.get('evaluated_samples', row.get('total_samples', 0))}"
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
