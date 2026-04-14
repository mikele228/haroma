"""
derivation_merge — Aggregate structural proposals from cycle producers.

Normalises reasoning inferences, LLM-context inferences / env_updates,
and goal-synthesis outputs into a single ``derivation`` dict on
``EpisodeContext`` so downstream consumers (action selection, logging,
optional apply pipeline) have one audit trail.
"""

from typing import Any, Dict, List


def _norm_inference(raw: Dict[str, Any], source: str) -> Dict[str, Any]:
    subj = str(raw.get("subject") or raw.get("subject_name") or "").strip()
    pred = str(raw.get("predicate", "") or "").strip()
    obj = str(raw.get("object") or raw.get("object_name") or "").strip()
    conf = 0.0
    try:
        conf = float(raw.get("confidence", 0))
    except (TypeError, ValueError):
        pass
    return {
        "kind": "kg_inference",
        "source": source,
        "confidence": round(min(1.0, max(0.0, conf)), 3),
        "payload": {
            "subject": subj,
            "predicate": pred,
            "object": obj,
            "verified": raw.get("verified"),
        },
    }


def _norm_env_update(key: str, value: Any) -> Dict[str, Any]:
    return {
        "kind": "env_update",
        "source": "llm_context",
        "confidence": 0.5,
        "payload": {"key": str(key), "value": value},
    }


def _norm_goal(goal: Dict[str, Any], source: str) -> Dict[str, Any]:
    conf = 0.0
    try:
        conf = float(goal.get("priority", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return {
        "kind": "goal",
        "source": source,
        "confidence": round(min(1.0, max(0.0, conf)), 3),
        "payload": {
            "goal_id": str(goal.get("goal_id", "")),
            "description": str(goal.get("description", "")),
        },
    }


def _norm_memory_note(text: str, source: str, confidence: float) -> Dict[str, Any]:
    return {
        "kind": "memory_note",
        "source": source,
        "confidence": round(min(1.0, max(0.0, confidence)), 3),
        "payload": {"text": str(text)},
    }


def merge_derivation_artifacts(
    episode: Any,
    *,
    synthesized_goals: List[Dict[str, Any]] | None = None,
) -> None:
    """Mutate *episode*.derivation from already-bound fields.

    Call **after** reasoning, LLM context, and goal synthesis have been
    bound to the episode, but **before** ``build_action_episode_payload``.
    """
    proposals: List[Dict[str, Any]] = []

    # --- reasoning inferences ---
    reasoning = getattr(episode, "reasoning", None) or {}
    for inf in reasoning.get("inferences", []):
        if not isinstance(inf, dict):
            continue
        ni = _norm_inference(inf, "reasoning")
        pl = ni["payload"]
        if pl["subject"] or pl["predicate"] or pl["object"]:
            proposals.append(ni)

    # --- LLM-context inferences + env_updates ---
    llm_ctx = getattr(episode, "llm_context", None) or {}
    for inf in llm_ctx.get("inferences", []):
        if not isinstance(inf, dict):
            continue
        ni = _norm_inference(inf, "llm_context")
        pl = ni["payload"]
        if pl["subject"] or pl["predicate"] or pl["object"]:
            proposals.append(ni)
    for k, v in (llm_ctx.get("env_updates") or {}).items():
        proposals.append(_norm_env_update(k, v))

    # --- goal synthesis ---
    for sg in synthesized_goals or []:
        if isinstance(sg, dict):
            proposals.append(_norm_goal(sg, sg.get("source", "goal_synthesis")))

    # --- LLM-context answer as a memory_note when grounded ---
    llm_answer = (llm_ctx.get("answer") or "").strip()
    llm_conf = 0.0
    try:
        llm_conf = float(llm_ctx.get("confidence", 0))
    except (TypeError, ValueError):
        pass
    if llm_answer and llm_conf >= 0.4 and not llm_ctx.get("requires_confirmation", True):
        proposals.append(_norm_memory_note(llm_answer, "llm_context", llm_conf))

    proposals.sort(key=lambda p: p["confidence"], reverse=True)

    parts: List[str] = []
    for p in proposals[:5]:
        kind = p["kind"]
        pl = p["payload"]
        if kind == "kg_inference":
            parts.append(f"{pl['subject']} -[{pl['predicate']}]-> {pl['object']}")
        elif kind == "goal":
            parts.append(f"goal: {pl['description']}")
        elif kind == "env_update":
            parts.append(f"env: {pl['key']}={pl['value']}")
        elif kind == "memory_note":
            text = pl.get("text", "")
            parts.append(text[:80] if len(text) > 80 else text)
    summary = "; ".join(parts) if parts else ""

    episode.bind_derivation({"proposals": proposals, "summary": summary})
