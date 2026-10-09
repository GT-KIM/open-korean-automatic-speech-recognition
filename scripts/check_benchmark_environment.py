#!/usr/bin/env python3
"""Check the shared GPU container without downloading or loading ASR weights."""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_runtime


def package_mismatches(constraints, installed):
    normalized = {re.sub(r"[-_.]+", "-", k).lower(): v for k, v in installed.items()}
    failures = []
    for line in constraints.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        actual = normalized.get(re.sub(r"[-_.]+", "-", name).lower())
        # PEP 440: ==0.14.0 permits 0.14.0+cpu; explicit +cu130 stays exact.
        comparable = actual.split("+", 1)[0] if actual and "+" not in expected else actual
        if comparable != expected:
            failures.append(f"{name}: expected {expected}, found {actual}")
    return failures


def package_inventory():
    versions = {}
    # A venv may shadow a distribution in the base image. Follow sys.path precedence.
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            key = re.sub(r"[-_.]+", "-", name).lower()
            versions.setdefault(key, distribution.version)
    return dict(sorted(versions.items()))


def gpu_check():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA device with --gpu")
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"Expected CUDA runtime 13.0, found {torch.version.cuda}")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native BF16 support is required")
    value = torch.ones((32, 32), dtype=torch.bfloat16, device="cuda:0")
    torch.testing.assert_close(value @ value, torch.full_like(value, 32))
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": properties.name,
        "uuid": str(getattr(properties, "uuid", "unknown")),
        "memory_bytes": properties.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "visible_device_count": torch.cuda.device_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "bf16_matmul": "passed",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; use a new preflight run directory")
    versions = package_inventory()
    constraint_path = ROOT / "docker/benchmark-constraints.txt"
    constraints = constraint_path.read_text(encoding="utf-8")
    failures = package_mismatches(constraints, versions)
    image_id = os.environ.get("ASR_IMAGE_ID", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        failures.append("Missing immutable image ID; launch with docker/benchmark.py")
    if platform.system() != "Linux":
        failures.append("The shared benchmark runtime must be a Linux container")
    report = {
        "schema_version": 1,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_id": os.environ.get("ASR_ENVIRONMENT_ID"),
        "host_kind": os.environ.get("ASR_HOST_KIND"),
        "image_id": image_id,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "os_release": platform.freedesktop_os_release() if platform.system() == "Linux" else {},
        "packages": dict(sorted(versions.items())),
        "constraints_sha256": hashlib.sha256(constraint_path.read_bytes()).hexdigest(),
        "checks": {},
        "failures": failures,
        "measurement_ready": False,
        "remaining_gates": ["seal original audio/reference contents", "verify ASR model pilot",
                            "validate model-reuse timing harness", "seal both host environments"],
    }
    for name, check in (("runtime_imports_and_audio", lambda: check_runtime.main([])),
                        ("gpu", gpu_check)):
        try:
            result = check()
            if name == "runtime_imports_and_audio" and result != 0:
                raise RuntimeError("Runtime import/audio check failed")
            report["checks"][name] = result
        except Exception as error:
            failures.append(f"{name}: {type(error).__name__}: {error}")
    try:
        report["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT, timeout=15,
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        failures.append(f"nvidia-smi: {error}")
    try:
        from openkoasr.evaluation.provenance import code_metadata
        report["code"] = code_metadata()
    except Exception as error:
        failures.append(f"source provenance: {type(error).__name__}: {error}")
    report["status"] = "passed" if not failures else "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Environment preflight {report['status']}: {args.output}")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
