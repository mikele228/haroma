"""Load ``RLlibTransitionLogger`` JSONL for offline Ray RLlib / batch workflows.

Does not import Ray at module import time — safe to use for counting and
inspection without ``pip install 'ray[rllib]'``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional

from mind.config_env import env_truthy

_MISSING_FILE_WARNED: set[str] = set()


def default_transitions_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "data", "rllib", "transitions.jsonl")


def iter_bandit_steps(path: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    p = path or os.environ.get("HAROMA_RLLIB_TRANSITIONS_PATH") or default_transitions_path()
    if not os.path.isfile(p):
        if env_truthy("HAROMA_RLLIB_OFFLINE_VERBOSE", False) and p not in _MISSING_FILE_WARNED:
            _MISSING_FILE_WARNED.add(p)
            print(
                f"[rllib_jsonl_offline] no transitions file at {p!r} (empty iteration).",
                flush=True,
            )
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("type") == "bandit_step":
                yield rec


def load_bandit_steps(path: Optional[str] = None, limit: int = 100_000) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, rec in enumerate(iter_bandit_steps(path)):
        if i >= limit:
            break
        out.append(rec)
    return out


def summarize_file(path: Optional[str] = None) -> Dict[str, Any]:
    """Return counts and reward stats without Ray."""
    n = 0
    rw_sum = 0.0
    with_fp = 0
    for rec in iter_bandit_steps(path):
        n += 1
        try:
            rw_sum += float(rec.get("reward", 0.0))
        except (TypeError, ValueError):
            pass
        info = rec.get("info") or {}
        if isinstance(info, dict) and info.get("agent_environment_fp"):
            with_fp += 1
    return {
        "count": n,
        "mean_reward": (rw_sum / n) if n else None,
        "with_agent_environment_fp": with_fp,
        "path": path or os.environ.get("HAROMA_RLLIB_TRANSITIONS_PATH") or default_transitions_path(),
    }


def try_run_ray_offline_bandit(path: Optional[str] = None) -> bool:
    """If Ray RLlib is installed, print a hook point for batch training.

    Full RLlib bandit training is deployment-specific; this verifies imports and
    loads transitions for a custom ``AlgorithmConfig`` / trainer script.
    """
    try:
        import ray  # noqa: F401
        from ray import tune  # noqa: F401
    except ImportError:
        print(
            "[rllib_jsonl_offline] ray not installed — "
            "pip install 'ray[rllib]' for offline RLlib workflows.",
            flush=True,
        )
        return False
    steps = load_bandit_steps(path, limit=8)
    print(
        f"[rllib_jsonl_offline] ray OK; sample loaded {len(steps)} rows "
        f"(use tune / RLlib BanditLinUCB or your trainer on obs/action/reward).",
        flush=True,
    )
    return True
