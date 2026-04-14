#!/usr/bin/env python3
"""Cross-platform smoke checks (same steps as ``check_haroma`` / Pixi ``check_smoke``)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    pkgs = "agents core engine mind sensors utils soul main.py"
    steps = [
        [sys.executable, "-m", "compileall", "-q", *pkgs.split()],
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            *pkgs.split(),
            "--select",
            "F821",
        ],
        [
            sys.executable,
            "-m",
            "mypy",
            "mind/deliberative_choice.py",
            "mind/cognitive_contracts.py",
            "mind/chat_visibility.py",
            "mind/deliberative_llm_merge.py",
            "mind/response_text.py",
            "mind/user_identity.py",
            "mind/cognitive_observability.py",
            "mind/robot_body_state.py",
            "utils/coerce_bool.py",
            "engine/LLMContextReasoner.py",
            "--follow-imports=skip",
            "--explicit-package-bases",
        ],
    ]
    for cmd in steps:
        print(f"== {' '.join(cmd)} ==", flush=True)
        r = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            return r.returncode
    print("CHECK OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
