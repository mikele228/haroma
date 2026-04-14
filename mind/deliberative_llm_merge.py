"""Merge deliberative scoring into packed-LLM results — pure orchestration.

Callers gather cognitive fields; this module runs ``select_deliberative_candidate``
and writes ``deliberative_scores`` / ``chosen_action`` on the result dict.
The merge function is also re-exported from :mod:`mind.cognitive_contracts`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from mind.deliberative_choice import (
    canonical_emotion_for_deliberation,
    select_deliberative_candidate,
)


def _dominant_drive_and_levels(
    drive_state: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, float]]:
    dd = ""
    dlevels: Dict[str, float] = {}
    if not isinstance(drive_state, dict):
        return dd, dlevels
    dd = str(drive_state.get("dominant_drive") or "")
    inner = drive_state.get("drives")
    if isinstance(inner, dict):
        for k, v in inner.items():
            try:
                dlevels[str(k)] = float(v)
            except (TypeError, ValueError):
                pass
    return dd, dlevels


def _law_violations(symbolic_law: Optional[Dict[str, Any]]) -> List[Any]:
    if not isinstance(symbolic_law, dict):
        return []
    raw_v = symbolic_law.get("violations", [])
    if isinstance(raw_v, list):
        return raw_v
    if raw_v is not None:
        return [raw_v]
    return []


def merge_deliberative_into_llm_context(
    llm_context_result: Dict[str, Any],
    *,
    deliberative_flag: bool,
    current_values: Dict[str, Any],
    active_goals: List[Any],
    drive_state: Optional[Dict[str, Any]],
    episode_affect: Any,
    emotion_summary: Optional[Dict[str, Any]],
    symbolic_law: Optional[Dict[str, Any]],
    log_context: str = "",
) -> None:
    """Mutates *llm_context_result* when deliberation applies; no-op otherwise."""
    if not deliberative_flag:
        return
    candidates = llm_context_result.get("candidate_actions")
    if not candidates:
        return
    dominant_drive, drive_levels = _dominant_drive_and_levels(drive_state)
    law_viol = _law_violations(symbolic_law)
    emo = canonical_emotion_for_deliberation(episode_affect, emotion_summary)
    try:
        winner, board = select_deliberative_candidate(
            candidates,
            current_values=current_values,
            active_goals=active_goals,
            dominant_drive=dominant_drive,
            drive_levels=drive_levels,
            law_violations=law_viol,
            emotion_summary=emo,
        )
        llm_context_result["deliberative_scores"] = board
        if winner:
            llm_context_result["chosen_action"] = winner
    except Exception as exc:
        prefix = f"[DeliberativeLLM]{log_context} " if log_context else "[DeliberativeLLM] "
        print(f"{prefix}deliberative select error: {exc}", flush=True)
