"""Run focused pytest: training hooks + query smoke (no GGUF). From repo root:

    python scripts/run_train_query_checks.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    tests = [
        "tests/test_llm_backend_train_query.py",
        "tests/test_vw_rl_bridge.py",
    ]
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", *tests]
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
