#!/usr/bin/env python3
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


STATIC_FILES = (
    "index.html",
    "styles.css",
    "app.js",
    "robots.txt",
    "sitemap.xml",
    "google340a95f996780abe.html",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site_dir", default="site")
    parser.add_argument("--data_path", default="doc/leaderboard_data.json")
    parser.add_argument("--markdown_path", default="leaderboard.md")
    parser.add_argument("--output_dir", default="_site")
    args = parser.parse_args()

    output_dir = build_pages(
        site_dir=Path(args.site_dir),
        data_path=Path(args.data_path),
        markdown_path=Path(args.markdown_path),
        output_dir=Path(args.output_dir),
    )
    print(f"Wrote GitHub Pages site to {output_dir}")


def build_pages(site_dir, data_path, markdown_path=None, output_dir=Path("_site")):
    site_dir = Path(site_dir)
    data_path = Path(data_path)
    output_dir = Path(output_dir)

    if not site_dir.is_dir():
        raise FileNotFoundError(f"Missing site directory: {site_dir}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing leaderboard data file: {data_path}")

    rows = _load_rows(data_path)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for filename in STATIC_FILES:
        source = site_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing static asset: {source}")
        shutil.copy2(source, output_dir / filename)

    shutil.copy2(data_path, output_dir / "leaderboard_data.json")
    if markdown_path:
        markdown_path = Path(markdown_path)
        if markdown_path.is_file():
            shutil.copy2(markdown_path, output_dir / "leaderboard.md")

    metadata = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows),
        "full_run_count": sum(1 for row in rows if row.get("is_full_evaluation") is not False),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")
    return output_dir


def _load_rows(data_path):
    with data_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Leaderboard data must be a JSON list: {data_path}")
    return rows


if __name__ == "__main__":
    main()
