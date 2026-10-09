#!/usr/bin/env python3
"""Build or preflight the same immutable benchmark image on either host."""

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "openkoasr-benchmark:20261006"
SEALED_BASE = "sha256:f1379555b571ec99d12029fc3492530e65de91f0a82e5a10e26c303fd5231aa7"


def run_command(image_id, environment_id, host_kind, gpu, output):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("An immutable Docker image ID is required")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", environment_id):
        raise ValueError("Environment ID must contain lowercase letters, digits or hyphens")
    if not re.fullmatch(r"[0-9]+|GPU-[a-fA-F0-9-]+", gpu):
        raise ValueError("Select one GPU index or UUID")
    if "," in str(output):
        raise ValueError("The output path must not contain commas")
    # WSL exposes GPUs via --gpus all; CUDA visibility selects exactly one for Torch.
    gpu_request = "all" if host_kind == "wsl2" else f"device={gpu}"
    visible = gpu if host_kind == "wsl2" else "0"
    return [
        "docker", "run", "--rm", "--pull=never", "--network=none", "--gpus", gpu_request,
        "--shm-size=1g", "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--env", f"CUDA_VISIBLE_DEVICES={visible}",
        "--env", f"ASR_IMAGE_ID={image_id}",
        "--env", f"ASR_ENVIRONMENT_ID={environment_id}",
        "--env", f"ASR_HOST_KIND={host_kind}",
        "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
        "--mount", f"type=bind,source={output},target=/results",
        image_id,
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    build = commands.add_parser("build")
    build.add_argument("--image", default=DEFAULT_IMAGE)
    speed = commands.add_parser("build-speed")
    speed.add_argument("--image", default="openkoasr-speed:20261007")
    check = commands.add_parser("check")
    check.add_argument("--image", default=DEFAULT_IMAGE)
    check.add_argument("--environment", required=True, help="Separate ID for each host")
    check.add_argument("--host-kind", required=True, choices=("wsl2", "native-linux"))
    check.add_argument("--gpu", default="0")
    args = parser.parse_args(argv)
    if args.action == "build-speed":
        base = json.loads(subprocess.check_output(["docker", "image", "inspect", SEALED_BASE], text=True))[0]
        if base["Id"] != SEALED_BASE or base["Os"] != "linux" or base["Architecture"] != "amd64":
            parser.error("Load the verified base image archive before building the speed image")
        subprocess.check_call(["docker", "tag", SEALED_BASE, "openkoasr-sealed-base:f1379555b571"])
        return subprocess.call(["docker", "build", "--platform=linux/amd64", "--file",
                                str(ROOT / "docker/Dockerfile.speed"), "--tag", args.image, str(ROOT)])
    if args.action == "build":
        return subprocess.call(["docker", "build", "--platform=linux/amd64", "--file",
                                str(ROOT / "docker/Dockerfile.benchmark"), "--tag", args.image,
                                str(ROOT)])
    # Resolve once, then run the exact content ID even if its tag changes.
    inspection = json.loads(subprocess.check_output(
        ["docker", "image", "inspect", args.image], text=True,
    ))[0]
    if inspection["Os"] != "linux" or inspection["Architecture"] != "amd64":
        parser.error("Expected a linux/amd64 image")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "results" / "environments" / args.environment / run_id
    try:
        command = run_command(inspection["Id"], args.environment, args.host_kind, args.gpu, output)
    except ValueError as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"environment_id": args.environment, "host_kind": args.host_kind,
               "image_id": inspection["Id"], "repo_digests": inspection.get("RepoDigests", []),
               "gpu_selector": args.gpu}
    (output / "image.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
