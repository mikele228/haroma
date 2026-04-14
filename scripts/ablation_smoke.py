"""Smoke run with ELARION_ABLATION tags + optional NDJSON trace (set env before imports)."""

from __future__ import annotations

import argparse
import os
import sys

_REPO = __file__.rsplit("scripts", 1)[0].rstrip("\\/")
sys.path.insert(0, _REPO)


def main():
    p = argparse.ArgumentParser(
        description="Run 1-3 light cycles with ablated modules (e.g. imagination,metacognition).",
    )
    p.add_argument(
        "--ablate",
        default="imagination,metacognition",
        help="Comma-separated optional steps to force off (see ProcessGate.GATABLE_STEPS + reconciliation).",
    )
    p.add_argument("--trace", action="store_true", help="Enable logs/cognitive_trace.ndjson")
    p.add_argument("--cycles", type=int, default=2)
    args = p.parse_args()

    os.environ["ELARION_ABLATION"] = args.ablate.lower().strip()
    if args.trace:
        os.environ["ELARION_TRACE"] = "1"
    else:
        os.environ.pop("ELARION_TRACE", None)

    from mind.control import ElarionController

    ctrl = ElarionController()
    for i in range(max(1, args.cycles)):
        r = ctrl.run_cycle(
            {"content": f"ablation smoke {i}", "tags": ["test"]},
            role="observer",
        )
        co = r.get("canonical_outcome") or {}
        print(
            f"cycle {i + 1} canonical_score={co.get('score')} "
            f"planner={r.get('planner_arbitration', {}).get('chosen_source')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
