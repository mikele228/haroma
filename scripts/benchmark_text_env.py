"""Frozen TextEnvironment tasks for regression metrics (no full cognitive stack)."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable, Dict, List, Tuple

# Repo root on path
sys.path.insert(0, __file__.rsplit("scripts", 1)[0].rstrip("\\/"))

from environment.TextEnvironment import TextEnvironment

Task = Tuple[str, List[str], Callable[[TextEnvironment], bool]]


def _loc(e: TextEnvironment) -> str:
    return e.stats()["player_location"]


TASKS: List[Task] = [
    (
        "reach_library",
        ["north"],
        lambda e: _loc(e) == "library",
    ),
    (
        "reach_tower",
        ["north", "up"],
        lambda e: _loc(e) == "tower",
    ),
    (
        "reach_deep_cave",
        ["south", "south"],
        lambda e: _loc(e) == "deep_cave",
    ),
    (
        "reach_meadow",
        ["east", "north", "east"],
        lambda e: _loc(e) == "meadow",
    ),
    (
        "reach_garden",
        ["west"],
        lambda e: _loc(e) == "garden",
    ),
]


def run_benchmark() -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    env = TextEnvironment()
    passed = 0
    for task_id, moves, check in TASKS:
        env.reset()
        ok_moves = True
        for direction in moves:
            out = env.execute_action("explore", direction)
            if not out.get("success", False):
                ok_moves = False
                break
        success = ok_moves and check(env)
        results[task_id] = {
            "success": bool(success),
            "final_location": _loc(env),
            "steps": len(moves),
        }
        if success:
            passed += 1
    results["_summary"] = {
        "tasks_total": len(TASKS),
        "tasks_passed": passed,
        "pass_rate": passed / max(len(TASKS), 1),
    }
    return results


def main():
    p = argparse.ArgumentParser(description="Run frozen TextEnvironment benchmark tasks.")
    p.add_argument("--json", action="store_true", help="Print JSON line")
    args = p.parse_args()
    out = run_benchmark()
    if args.json:
        import json

        print(json.dumps(out, default=str))
    else:
        s = out.pop("_summary")
        for k, v in out.items():
            print(f"  {k}: {'OK' if v['success'] else 'FAIL'} -> {v['final_location']}")
        print(
            f"Summary: {s['tasks_passed']}/{s['tasks_total']} ({s['pass_rate']:.0%})",
        )


if __name__ == "__main__":
    main()
