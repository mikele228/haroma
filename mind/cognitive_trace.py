"""Unified cognitive trace (NDJSON) + canonical outcome spine + ablation env helpers."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Set

TRACE_FILE_ENV = "ELARION_TRACE_FILE"
TRACE_ENABLE_ENV = "ELARION_TRACE"
ABLATION_ENV = "ELARION_ABLATION"


def parse_ablation_tags() -> Set[str]:
    raw = os.environ.get(ABLATION_ENV, "").strip().lower()
    if not raw:
        return set()
    return {t.strip() for t in raw.split(",") if t.strip()}


def apply_ablation_overrides(gate_decisions: Dict[str, Any]) -> Dict[str, Any]:
    """Force optional steps off when listed in ELARION_ABLATION (comma-separated)."""
    tags = parse_ablation_tags()
    if not tags:
        return gate_decisions
    out = dict(gate_decisions)
    for step in tags:
        if step in out:
            out[step] = False
    return out


def reconciliation_ablated() -> bool:
    return "reconciliation" in parse_ablation_tags()


def _trace_file_path() -> Optional[str]:
    explicit = os.environ.get(TRACE_FILE_ENV, "").strip()
    if explicit:
        return explicit
    if os.environ.get(TRACE_ENABLE_ENV, "").lower() in ("1", "true", "yes"):
        return os.path.join("logs", "cognitive_trace.ndjson")
    return None


def build_canonical_outcome(
    *,
    episode_id: str,
    cycle_id: int,
    role: str,
    outcome: Dict[str, Any],
    gate_decisions: Dict[str, Any],
    steps_run: int,
    steps_total: int,
    action_strategy: str,
    action_type: str,
) -> Dict[str, Any]:
    bd = outcome.get("breakdown") if isinstance(outcome.get("breakdown"), dict) else {}
    keys = list(bd.keys()) if bd else []
    return {
        "schema": "haroma.canonical_outcome.v1",
        "episode_id": episode_id,
        "cycle_id": cycle_id,
        "role": role,
        "timestamp": time.time(),
        "score": float(outcome.get("score", 0.0) or 0.0),
        "breakdown_keys": sorted(keys),
        "gate_enabled_count": sum(1 for v in gate_decisions.values() if v is True),
        "gate_total": len(gate_decisions),
        "steps_run": steps_run,
        "steps_total": steps_total,
        "action_strategy": action_strategy,
        "action_type": action_type,
    }


def build_planner_arbitration(
    *,
    memory_hint: Optional[str],
    imagined_strategy: Optional[str],
    current_plan: Optional[List[str]],
    plan_step: int,
    resolved_hint: Optional[str],
) -> Dict[str, Any]:
    """Audit trail for resolve_strategy_hint (plan overrides memory when both active)."""
    plan_step_name: Optional[str] = None
    try:
        _ps = int(plan_step)
    except (TypeError, ValueError):
        _ps = 0
    if current_plan and 0 <= _ps < len(current_plan):
        plan_step_name = current_plan[_ps]

    if current_plan and 0 <= _ps < len(current_plan):
        chosen = "imagined_plan"
    elif memory_hint:
        chosen = "action_memory"
    elif imagined_strategy:
        chosen = "imagination"
    else:
        chosen = "none"

    return {
        "chosen_source": chosen,
        "resolved_hint": resolved_hint,
        "candidates": {
            "action_memory": memory_hint,
            "imagination": imagined_strategy,
            "plan_step": plan_step_name,
        },
        "memory_disagrees_imagination": bool(
            memory_hint and imagined_strategy and memory_hint != imagined_strategy
        ),
    }


class CognitiveTraceRecorder:
    """Append one JSON object per line (NDJSON) for offline metrics / ablations."""

    def __init__(self, path: Optional[str] = None):
        self._path = path

    @classmethod
    def from_env(cls) -> CognitiveTraceRecorder:
        return cls(_trace_file_path())

    def record(self, record: Dict[str, Any]) -> None:
        if not self._path:
            return
        line = json.dumps(record, default=str)
        try:
            parent = os.path.dirname(os.path.abspath(self._path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            print(f"[CognitiveTrace] write failed: {exc}", flush=True)
