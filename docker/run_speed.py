#!/usr/bin/env python3
"""Launch separate offline containers for each model/track on one host."""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from benchmark import run_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "inputs", "cache", "preflight", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--host-kind", choices=("wsl2", "native-linux"), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--tracks", nargs="+", choices=("latency-b1", "throughput-b4"),
                        default=["latency-b1", "throughput-b4"])
    parser.add_argument("--purpose", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="Pilot diagnostics only: collect the other pairs after a failed gate")
    args = parser.parse_args()
    if args.continue_on_failure and args.purpose != "pilot":
        parser.error("Continuing after failure is only permitted for diagnostic pilots")
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    image_id = protocol["environment"]["image_id"]
    inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", image_id], text=True))[0]
    if inspected["Id"] != image_id or inspected["Os"] != "linux" or inspected["Architecture"] != "amd64":
        parser.error("The sealed Linux image is required")
    for path in (args.protocol, args.inputs, args.cache, args.preflight):
        if not path.exists() or "," in str(path):
            parser.error(f"Missing or unsupported mount path: {path}")
    allowed = [m["repo"] for m in protocol["models"]]
    models = args.models or allowed
    if len(set(models)) != len(models) or any(m not in allowed for m in models):
        parser.error("Select distinct models from the frozen protocol")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = args.output_root.resolve() / args.environment / stamp
    root.mkdir(parents=True, exist_ok=False)
    frozen_protocol = root / "protocol.json"
    frozen_protocol.write_bytes(protocol_bytes)
    receipt = {"image_id": image_id, "environment": args.environment, "purpose": args.purpose,
               "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
               "models": models, "tracks": args.tracks, "runs": []}
    print(f"Output: {root}", flush=True)
    for model in models:
        for track in args.tracks:
            if args.host_kind == "native-linux":
                active = subprocess.check_output(["nvidia-smi", "-i", args.gpu,
                                                  "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
                if active:
                    raise RuntimeError("Another GPU compute process is active; do not overlap experiments")
            output = root / model.replace("/", "--") / track
            output.mkdir(parents=True, exist_ok=False)
            command = run_command(image_id, args.environment, args.host_kind, args.gpu, output)
            mounts = []
            for source, target in ((frozen_protocol, "/protocol.json"), (args.inputs, "/inputs"),
                                   (args.cache, "/cache/huggingface"), (args.preflight, "/preflight.json")):
                mounts += ["--mount", f"type=bind,source={source.resolve()},target={target},readonly"]
            command[-1:-1] = mounts
            command += ["python", "/app/scripts/run_speed_benchmark.py", "--protocol", "/protocol.json",
                        "--inputs", "/inputs", "--preflight", "/preflight.json", "--model", model,
                        "--track", track, "--purpose", args.purpose, "--output", "/results/run"]
            print(f"Running {model} {track}", flush=True)
            with (output / "console.log").open("w", encoding="utf-8") as log:
                code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
            receipt["runs"].append({"model": model, "track": track, "exit_code": code,
                                    "output": output.relative_to(root).as_posix()})
            (root / "matrix.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            print(f"{'PASS' if code == 0 else 'FAIL'} {model} {track}", flush=True)
            if code and not args.continue_on_failure:
                return code
    return int(any(run["exit_code"] for run in receipt["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
