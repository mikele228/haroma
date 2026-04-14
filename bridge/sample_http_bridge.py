#!/usr/bin/env python3
"""
Demo: build a minimal command batch (like cognition would), "execute" with the stub,
POST feedback to Haroma ``/robot/bridge/feedback``.

Usage (from repo root)::

    python bridge/sample_http_bridge.py --haroma-url http://127.0.0.1:8193

Dry-run (no network)::

    python bridge/sample_http_bridge.py --dry-run

Requires Haroma running for non-dry-run. Uses stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow `python bridge/sample_http_bridge.py` without installing the package
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bridge.haroma_client import health_ping, post_robot_bridge_feedback  # noqa: E402
from bridge.stub_executor import feedback_block_from_results, simulate_command_results  # noqa: E402


def _demo_batch():
    from mind.robot_execution_contract import build_executor_command_batch

    return build_executor_command_batch(
        [
            {
                "label": "demo noop",
                "command": "noop",
                "priority": 0.5,
                "supports_goal_id": "",
            }
        ],
        source="bridge_sample",
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Stub bridge: demo batch → Haroma feedback")
    p.add_argument(
        "--haroma-url",
        default=os.environ.get("HAROMA_URL", "http://127.0.0.1:8193"),
        help="Base URL for Haroma (default HAROMA_URL or http://127.0.0.1:8193)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print JSON only; do not POST")
    args = p.parse_args()

    batch = _demo_batch()
    results = simulate_command_results(batch)
    fb = feedback_block_from_results(batch, results)

    if args.dry_run:
        print(json.dumps({"batch": batch, "feedback": fb}, indent=2, ensure_ascii=False))
        return 0

    st, code = health_ping(args.haroma_url)
    if code != 200:
        print(f"[bridge] GET /status failed: HTTP {code} body={st}", file=sys.stderr)
        return 1

    body, code = post_robot_bridge_feedback(args.haroma_url, fb)
    print(json.dumps({"http_status": code, "response": body}, indent=2, ensure_ascii=False))
    return 0 if code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
