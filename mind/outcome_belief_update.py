"""
Outcome-grounded belief updates — nudge inferred propositions using the realized
action outcome score, and persist audit traces on ``belief_tree``.

High-level outcomes reinforce hypotheses the agent acted under; poor outcomes
weaken them. This is not truth in the world — it is **credit assignment** to
structured inferences the cycle already produced.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional, Set

from utils.coerce_bool import env_flag


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _proposition_from_inference(row: Dict[str, Any]) -> str:
    subj = str(row.get("subject", "") or "").strip()
    pred = str(row.get("predicate", "") or "").strip()
    obj = str(row.get("object", "") or "").strip()
    if subj and pred and obj:
        return f"{subj} —{pred}→ {obj}"
    if subj and pred:
        return f"{subj} —{pred}"
    return (subj or pred or obj or "").strip()


def _merge_inference_rows(
    reasoning_result: Optional[Dict[str, Any]],
    llm_context: Optional[Dict[str, Any]],
    max_rows: int = 12,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for src in (reasoning_result, llm_context):
        if not isinstance(src, dict):
            continue
        for row in src.get("inferences") or []:
            if not isinstance(row, dict):
                continue
            prop = _proposition_from_inference(row)
            if not prop:
                continue
            h = hashlib.sha256(prop.lower().encode("utf-8", errors="ignore")).hexdigest()[:16]
            if h in seen:
                continue
            seen.add(h)
            out.append(row)
            if len(out) >= max_rows:
                return out
    return out


def apply_outcome_grounded_belief_updates(
    *,
    memory: Any,
    outcome: Dict[str, Any],
    reasoning_result: Optional[Dict[str, Any]],
    llm_context: Optional[Dict[str, Any]],
    cycle_id: int,
    branch_name: str,
    agent_id: str,
    plasticity_index: Optional[float] = None,
) -> int:
    """Write outcome-conditioned belief traces to ``belief_tree``.

    Returns the number of memory nodes appended.
    """
    if not env_flag("HAROMA_OUTCOME_BELIEF_UPDATE", True):
        return 0
    if memory is None or not hasattr(memory, "add_node"):
        return 0

    try:
        from core.Memory import MemoryNode
    except Exception:
        return 0

    score = float(outcome.get("score", 0.5) or 0.5)
    score = max(0.0, min(1.0, score))
    strength = _env_float("HAROMA_OUTCOME_BELIEF_STRENGTH", 0.12)
    if env_flag("HAROMA_BRAIN_PLASTICITY_COUPLING", True) and plasticity_index is not None:
        try:
            pi = max(0.0, min(1.0, float(plasticity_index)))
            strength *= 0.45 + 0.55 * pi
        except (TypeError, ValueError):
            pass
    residual = score - 0.5
    # Shift confidence: good outcomes push inferred beliefs up, bad outcomes down.
    delta = residual * 2.0 * strength

    rows = _merge_inference_rows(reasoning_result, llm_context)
    if not rows:
        return 0

    lesson = str(outcome.get("lesson") or "")[:160]
    n_added = 0
    for row in rows:
        prop = _proposition_from_inference(row)
        if not prop:
            continue
        try:
            base = float(row.get("confidence", 0.5))
        except (TypeError, ValueError):
            base = 0.5
        new_c = max(0.05, min(0.98, base + delta))
        tags = [
            "outcome_grounded",
            "belief_update",
            f"persona:{agent_id}",
            f"cycle:{cycle_id}",
            f"outcome_score:{score:.3f}",
        ]
        line = (
            f"[outcome {score:.2f}] Δconf≈{delta:+.3f} | "
            f"{prop[:220]}{'…' if len(prop) > 220 else ''} | "
            f"conf {base:.2f}→{new_c:.2f}"
        )
        if lesson:
            line += f" | lesson: {lesson}"
        node = MemoryNode(
            content=line[:900],
            emotion="neutral",
            confidence=new_c,
            tags=tags,
        )
        try:
            memory.add_node("belief_tree", branch_name, node)
            n_added += 1
        except Exception:
            continue

    if n_added and env_flag("HAROMA_OUTCOME_BELIEF_LOG", False):
        print(
            f"[OutcomeBelief] persona={agent_id} cycle={cycle_id} "
            f"score={score:.3f} nodes={n_added}",
            flush=True,
        )
    return n_added


def deliberative_belief_outcome_multiplier(outcome_score: float) -> float:
    """Scale deliberative ``confidence_delta`` by how well the turn went.

    Neutral (0.5) → 1.0; strong success → up to ~1.25; failure → ~0.75.
    """
    s = max(0.0, min(1.0, float(outcome_score)))
    beta = _env_float("HAROMA_DELIB_OUTCOME_BETA", 0.5)
    return 1.0 + beta * (s - 0.5)
