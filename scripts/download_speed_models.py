#!/usr/bin/env python3
"""Prepare public pinned snapshots before an offline benchmark run."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    receipt = []
    for model in protocol["models"]:
        print(f"Preparing {model['repo']} at {model['model_revision']}", flush=True)
        path = Path(snapshot_download(model["repo"], revision=model["model_revision"], max_workers=4,
                                     allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja"]))
        config_hash = hashlib.sha256((path / "generation_config.json").read_bytes()).hexdigest()
        if config_hash != model["generation_config_sha256"]:
            raise ValueError("Downloaded generation config differs from frozen protocol")
        files = {}
        for file in sorted(path.rglob("*")):
            if file.is_file():
                digest = hashlib.sha256()
                with file.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                files[file.relative_to(path).as_posix()] = {"bytes": file.stat().st_size, "sha256": digest.hexdigest()}
        receipt.append({"repo": model["repo"], "revision": model["model_revision"], "files": files})
        output = Path(os.environ["HF_HOME"]) / "speed-models.json"
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"Prepared {model['repo']}: {len(files)} files verified", flush=True)


if __name__ == "__main__":
    main()
