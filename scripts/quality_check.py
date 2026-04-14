#!/usr/bin/env python3
"""Run pre-commit when inside a Git repo; otherwise Ruff F821,E9 (matches .pre-commit-config)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Same package list as ``scripts/run_checks.py`` / smoke Ruff scope.
_PKGS = "agents core engine mind sensors utils soul main.py".split()


def main() -> int:
    git_dir = ROOT / ".git"
    if git_dir.exists():
        r = subprocess.run(
            [
                "pre-commit",
                "run",
                "--all-files",
                "--show-diff-on-failure",
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
        )
        return r.returncode
    print(
        "No .git here: skipping pre-commit (needs a Git repo). "
        "Running Ruff with --select F821,E9 on core packages.",
        flush=True,
    )
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            *_PKGS,
            "--select",
            "F821,E9",
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
    )
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
