"""Run N controller cycles and report outcome score statistics (continual smoke)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

_REPO = __file__.rsplit("scripts", 1)[0].rstrip("\\/")
sys.path.insert(0, _REPO)


def main():
    p = argparse.ArgumentParser(description="Continual eval: mean/min/max outcome over N cycles.")
    p.add_argument("--cycles", type=int, default=30)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if args.trace:
        os.environ["ELARION_TRACE"] = "1"

    from mind.control import ElarionController

    ctrl = ElarionController()
    scores: list[float] = []
    for i in range(max(1, args.cycles)):
        r = ctrl.run_cycle(
            {"content": f"eval cycle {i}", "tags": ["eval"]},
            role="observer",
        )
        oc = r.get("canonical_outcome") or {}
        try:
            s = float(oc.get("score", r.get("outcome", {}).get("score", 0.5)) or 0.5)
        except (TypeError, ValueError):
            s = 0.5
        scores.append(s)

    report = {
        "cycles": len(scores),
        "mean": round(statistics.mean(scores), 4) if scores else 0.0,
        "stdev": round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0,
        "min": round(min(scores), 4) if scores else 0.0,
        "max": round(max(scores), 4) if scores else 0.0,
    }
    if args.json:
        print(json.dumps(report))
    else:
        print(report)


if __name__ == "__main__":
    main()
