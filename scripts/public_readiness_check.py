#!/usr/bin/env python3
import re
import subprocess
import sys
from pathlib import Path


SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*=\s*['\"][^'\"]{8,}['\"]"),
]

LOCAL_PATH_PATTERNS = [
    re.compile(r"C:\\Users\\", re.IGNORECASE),
    re.compile(r"/mnt/[a-z]/", re.IGNORECASE),
    re.compile("Pycharm" + "Projects", re.IGNORECASE),
]

BINARY_EXTENSIONS = {
    ".bin",
    ".dll",
    ".exe",
    ".flac",
    ".jpg",
    ".jpeg",
    ".mp3",
    ".npy",
    ".pcm",
    ".png",
    ".pt",
    ".safetensors",
    ".wav",
    ".zip",
}


def main():
    root = Path.cwd()
    problems = []
    for path in tracked_files(root):
        if path.suffix.lower() in BINARY_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        rel = path.relative_to(root).as_posix()
        scan_text(rel, text, problems)

    history_text = git_history_patch(root)
    if history_text:
        scan_text("git history", history_text, problems)

    if problems:
        print("Public readiness check failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("Public readiness check passed.")
    return 0


def scan_text(label, text, problems):
    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            problems.append(f"{label}: possible secret-like token near {match.start()}")
    for pattern in LOCAL_PATH_PATTERNS:
        for match in pattern.finditer(text):
            problems.append(f"{label}: local path leak near {match.group(0)!r}")


def tracked_files(root):
    try:
        output = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return [
            path
            for path in root.rglob("*")
            if path.is_file() and not any(part.startswith(".git") for part in path.parts)
        ]
    return [root / line for line in output.splitlines() if line.strip()]


def git_history_patch(root):
    try:
        return subprocess.check_output(
            ["git", "log", "HEAD", "-p", "--no-ext-diff"],
            cwd=root,
            text=True,
            encoding="utf-8",
            errors="ignore",
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""


if __name__ == "__main__":
    sys.exit(main())
