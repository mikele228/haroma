"""Score LLM-proposed candidate actions and pick one for consequence application.

Impacts (value / goal / belief) are **advisory** — the model proposes deltas;
this module ranks candidates using current cognitive state, then the persona
applies the winner's impacts into ValueManager, GoalEngine, and WorkingMemory.

Four scoring pillars:
  1. **Goal completion** — positive goal_impact; small closure term for mild
     negative deltas (deprioritise / wind-down) on active goals.
  2. **Value / belief adherence** — value_impact weighted by current values;
     belief_impact count.
  3. **Law compliance** — model ``law_risk``; extra penalty when symbolic
     violations exist; tag overlap via optional ``action_tags`` (preferred) or
     legacy strategy name in ``matched_tags``.
  4. **Emotion fit** — LLM ``emotion_alignment``; dominant-label affinity;
     valence/arousal; **regulation** bonus under high intensity (calming strategies).

Env tuning: ``HAROMA_DELIBERATIVE_*`` — see ``_env_float`` defaults below.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

_STRATEGY_DRIVE_AFFINITY = {
    "understanding": {"inform": 0.15, "inquire": 0.12, "llm_context": 0.1, "derivation": 0.08},
    "coherence": {"reflect": 0.12, "inform": 0.06, "llm_context": 0.06},
    "expression": {"inform": 0.1, "empathize": 0.08, "llm_context": 0.06},
    "rest": {"reflect": 0.1, "empathize": 0.06},
    "connection": {"empathize": 0.12, "inquire": 0.08, "inform": 0.05},
}

_STRATEGY_EMOTION_AFFINITY: Dict[str, Dict[str, float]] = {
    "sadness": {"empathize": 0.12, "reflect": 0.08, "inquire": 0.04},
    "fear": {"empathize": 0.10, "reflect": 0.08, "inform": 0.04},
    "anger": {"reflect": 0.10, "empathize": 0.08, "observe": 0.04},
    "resolve": {"advance_goal": 0.12, "inform": 0.08, "explore": 0.06},
    "curiosity": {"inquire": 0.12, "explore": 0.10, "inform": 0.06},
    "joy": {"inform": 0.10, "empathize": 0.06, "explore": 0.04},
    "wonder": {"explore": 0.10, "inquire": 0.08, "inform": 0.04},
    "peace": {"reflect": 0.08, "observe": 0.06, "inform": 0.04},
    "surprise": {"inquire": 0.10, "explore": 0.08, "observe": 0.04},
    "neutral": {},
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)) or str(default))
    except (TypeError, ValueError):
        return default


def _clampf(x: Any, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def canonical_emotion_for_deliberation(
    episode_affect: Optional[Dict[str, Any]],
    emotion_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge ``episode.affect`` (authoritative labels) with engine summarize().

    Prefer non-empty values from *episode_affect*; fill gaps from *emotion_summary*
    so scoring matches JSON state and both ``dominant`` / ``dominant_emotion`` work.
    """
    merged: Dict[str, Any] = {}
    ea = episode_affect if isinstance(episode_affect, dict) else {}
    es = emotion_summary if isinstance(emotion_summary, dict) else {}
    for src in (es, ea):
        for k, v in src.items():
            if v is None:
                continue
            if k not in merged or merged.get(k) in ("", None):
                merged[k] = v
    if not _dominant_emotion_label(merged):
        d = es.get("dominant") or ea.get("dominant_emotion")
        if d:
            merged["dominant_emotion"] = d
    return merged


def _dominant_emotion_label(emo: Dict[str, Any]) -> str:
    """Resolve label from episode.affect or EmotionEngine.summarize() shapes."""
    for key in ("dominant_emotion", "dominant", "current_emotion"):
        raw = emo.get(key)
        if raw is None:
            continue
        s = str(raw).strip().lower()
        if s:
            return s
    return ""


def _emotion_model_weight() -> float:
    return max(0.0, min(0.5, _env_float("HAROMA_DELIBERATIVE_EMOTION_MODEL_WEIGHT", 0.15)))


def _emotion_affinity_scale() -> float:
    return max(0.0, min(2.0, _env_float("HAROMA_DELIBERATIVE_EMOTION_AFFINITY_SCALE", 1.0)))


def _emotion_valence_weight() -> float:
    return max(0.0, min(0.25, _env_float("HAROMA_DELIBERATIVE_EMOTION_VALENCE_WEIGHT", 0.08)))


def _emotion_arousal_weight() -> float:
    return max(0.0, min(0.25, _env_float("HAROMA_DELIBERATIVE_EMOTION_AROUSAL_WEIGHT", 0.06)))


def _emotion_regulation_weight() -> float:
    """Bonus for stabilizing strategies when intensity is high."""
    return max(0.0, min(0.2, _env_float("HAROMA_DELIBERATIVE_EMOTION_REGULATION_WEIGHT", 0.06)))


def _emotion_regulation_intensity_threshold() -> float:
    return max(
        0.35, min(0.95, _env_float("HAROMA_DELIBERATIVE_EMOTION_REGULATION_INTENSITY", 0.65))
    )


def _law_violation_risk_scale() -> float:
    """Extra penalty on law_risk when symbolic violations already fired."""
    return max(0.0, min(1.5, _env_float("HAROMA_DELIBERATIVE_LAW_VIOLATION_RISK_SCALE", 0.45)))


def _goal_closure_weight() -> float:
    """Weight for mild negative goal_impact (wind-down) on active goals."""
    return max(0.0, min(0.2, _env_float("HAROMA_DELIBERATIVE_GOAL_CLOSURE_WEIGHT", 0.08)))


def _emotion_fit_components(
    strat: str,
    emo: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Heuristic emotion fit split for telemetry."""
    parts: Dict[str, float] = {
        "emotion_label_affinity": 0.0,
        "emotion_valence": 0.0,
        "emotion_arousal": 0.0,
        "emotion_regulation": 0.0,
    }
    if not emo:
        return 0.0, parts

    bonus = 0.0
    intensity = _clampf(emo.get("intensity"), 0.0, 1.0, 0.5)
    intensity_gate = max(0.15, min(1.0, intensity))

    dominant = _dominant_emotion_label(emo)
    if dominant and strat:
        base = _STRATEGY_EMOTION_AFFINITY.get(dominant, {}).get(strat, 0.0)
        p = _emotion_affinity_scale() * base * intensity_gate
        parts["emotion_label_affinity"] = p
        bonus += p

    valence = _clampf(emo.get("valence"), -1.0, 1.0, 0.0)
    neg = max(0.0, -valence)
    pos = max(0.0, valence)
    vw = _emotion_valence_weight()
    vp = 0.0
    if strat == "empathize":
        vp = vw * neg * (1.15 * intensity_gate) + vw * pos * (0.35 * intensity_gate)
    elif strat == "reflect":
        vp = vw * neg * (0.75 * intensity_gate) + vw * pos * (0.25 * intensity_gate)
    elif strat == "inform":
        vp = vw * pos * (0.85 * intensity_gate) + vw * neg * (0.25 * intensity_gate)
    elif strat == "inquire":
        vp = vw * (neg + pos) * (0.45 * intensity_gate)
    elif strat in ("explore", "observe"):
        vp = vw * pos * (0.5 * intensity_gate)
    parts["emotion_valence"] = vp
    bonus += vp

    arousal = _clampf(emo.get("arousal"), -1.0, 1.0, 0.0)
    ar_norm = (arousal + 1.0) / 2.0
    aw = _emotion_arousal_weight()
    ap = 0.0
    if strat in ("inquire", "explore"):
        ap += aw * ar_norm * intensity_gate
    elif strat in ("reflect", "observe"):
        ap += aw * (1.0 - ar_norm) * (0.85 * intensity_gate)
    elif strat == "empathize":
        ap += aw * max(ar_norm, 1.0 - ar_norm) * (0.4 * intensity_gate)
    parts["emotion_arousal"] = ap
    bonus += ap

    # Regulation: high intensity → prefer grounding / calming strategies
    thr = _emotion_regulation_intensity_threshold()
    if intensity >= thr:
        rw = _emotion_regulation_weight()
        ramp = min(1.0, (intensity - thr) / max(0.05, 1.0 - thr))
        if strat in ("reflect", "observe", "inform"):
            reg = rw * ramp * intensity_gate
            parts["emotion_regulation"] = reg
            bonus += reg

    return bonus, parts


def compute_deliberative_score(
    ca: Dict[str, Any],
    *,
    current_values: Dict[str, float],
    active_goal_ids: Optional[set] = None,
    dominant_drive: str = "",
    drive_level: float = 0.5,
    law_violations: Optional[List[Dict[str, Any]]] = None,
    emotion_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Return (total_score, breakdown) for telemetry and tuning."""
    active_goal_ids = active_goal_ids or set()
    br: Dict[str, float] = {}

    # -- Values
    v_score = 0.0
    vi = ca.get("value_impact") or {}
    if isinstance(vi, dict):
        for k, delta in vi.items():
            try:
                d = float(delta)
            except (TypeError, ValueError):
                continue
            try:
                cur = float((current_values or {}).get(str(k), 0.5))
            except (TypeError, ValueError):
                cur = 0.5
            v_score += d * (0.4 + 0.6 * cur)
    br["values"] = v_score

    # -- Goals: progress + mild closure (negative delta)
    g_prog = 0.0
    g_close = 0.0
    gc_w = _goal_closure_weight()
    gi = ca.get("goal_impact") or {}
    if isinstance(gi, dict):
        for gid, delta in gi.items():
            try:
                d = float(delta)
            except (TypeError, ValueError):
                continue
            if str(gid) not in active_goal_ids:
                continue
            if d > 0:
                g_prog += 0.25 * min(1.0, d * 2.0)
            elif -0.35 < d < 0:
                g_close += gc_w * min(1.0, abs(d) * 2.5)
    br["goals_progress"] = g_prog
    br["goals_closure"] = g_close

    # -- Beliefs
    b_score = 0.0
    bi = ca.get("belief_impact") or []
    if isinstance(bi, list) and bi:
        b_score = 0.05 * min(len(bi), 3)
    br["beliefs"] = b_score

    # -- Confidence
    try:
        c_score = 0.12 * float(ca.get("confidence", 0.5))
    except (TypeError, ValueError):
        c_score = 0.06
    br["confidence"] = c_score

    strat = str(ca.get("strategy") or "").lower()
    dd = str(dominant_drive or "").lower()
    d_bonus = 0.0
    if dd and strat:
        d_bonus = _STRATEGY_DRIVE_AFFINITY.get(dd, {}).get(strat, 0.0) * max(
            0.2, min(1.0, float(drive_level))
        )
    br["drive_affinity"] = d_bonus

    try:
        law_risk = max(0.0, min(1.0, float(ca.get("law_risk", 0.0))))
    except (TypeError, ValueError):
        law_risk = 0.0
    law_model = -0.3 * law_risk
    br["law_model_penalty"] = law_model

    violations = law_violations or []
    law_ctx = 0.0
    law_tag = 0.0
    if violations:
        n_v = min(5, len(violations))
        sev_sum = 0.0
        violated_tags: Set[str] = set()
        max_severity = 0.5
        for v in violations:
            if not isinstance(v, dict):
                continue
            for t in v.get("matched_tags") or []:
                violated_tags.add(str(t).lower())
            try:
                sev = float(v.get("severity", 1.0))
            except (TypeError, ValueError):
                sev = 1.0
            max_severity = max(max_severity, sev)
            sev_sum += sev
        avg_sev = sev_sum / max(1, len([v for v in violations if isinstance(v, dict)]))
        norm_sev = max(0.5, min(2.0, avg_sev))
        # When violations exist, scale law_risk penalty harder
        law_ctx = -_law_violation_risk_scale() * law_risk * min(1.0, n_v / 3.0) * (norm_sev / 1.2)
        br["law_violation_context"] = law_ctx

        action_tags = ca.get("action_tags")
        tag_set: Set[str] = set()
        if isinstance(action_tags, (list, tuple)):
            tag_set = {str(x).lower() for x in action_tags if x is not None}
        overlap = tag_set and (tag_set & violated_tags)
        if overlap:
            law_tag = -1.0 * max(0.5, min(2.0, max_severity))
        elif strat and strat in violated_tags:
            law_tag = -1.0 * max(0.5, min(2.0, max_severity))
        br["law_tag_overlap"] = law_tag
    else:
        br["law_violation_context"] = 0.0
        br["law_tag_overlap"] = 0.0

    try:
        emo_align = max(-1.0, min(1.0, float(ca.get("emotion_alignment", 0.0))))
    except (TypeError, ValueError):
        emo_align = 0.0
    emo_model = _emotion_model_weight() * emo_align
    br["emotion_model"] = emo_model

    emo_h, emo_parts = _emotion_fit_components(strat, emotion_summary or {})
    br.update(emo_parts)
    br["emotion_heuristic_total"] = emo_h

    total = (
        v_score
        + g_prog
        + g_close
        + b_score
        + c_score
        + d_bonus
        + law_model
        + law_ctx
        + law_tag
        + emo_model
        + emo_h
    )
    return total, br


def score_deliberative_candidate(
    ca: Dict[str, Any],
    *,
    current_values: Dict[str, float],
    active_goal_ids: Optional[set] = None,
    dominant_drive: str = "",
    drive_level: float = 0.5,
    law_violations: Optional[List[Dict[str, Any]]] = None,
    emotion_summary: Optional[Dict[str, Any]] = None,
) -> float:
    """Higher = better fit to current state (heuristic, not ground truth)."""
    total, _ = compute_deliberative_score(
        ca,
        current_values=current_values,
        active_goal_ids=active_goal_ids,
        dominant_drive=dominant_drive,
        drive_level=drive_level,
        law_violations=law_violations,
        emotion_summary=emotion_summary,
    )
    return total


def _tie_sort_tuple(score: float, law_risk: float, confidence: float, cid: str) -> Tuple:
    """Sort key: best first — high score, low law_risk, high confidence, stable id."""
    return (-round(score, 6), round(law_risk, 6), -round(confidence, 6), cid)


def select_deliberative_candidate(
    candidates: List[Dict[str, Any]],
    *,
    current_values: Dict[str, float],
    active_goals: Optional[List[Dict[str, Any]]] = None,
    dominant_drive: str = "",
    drive_levels: Optional[Dict[str, float]] = None,
    law_violations: Optional[List[Dict[str, Any]]] = None,
    emotion_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (winner, scoreboard).

    Each scoreboard row includes ``score``, ``law_risk``, ``confidence``, and
    ``score_breakdown`` for tuning. Tie-break: score, then lower law_risk,
    then higher confidence, then lexicographic id.
    """
    if not candidates:
        return None, []

    gids = set()
    if active_goals:
        for g in active_goals:
            if isinstance(g, dict) and g.get("goal_id"):
                gids.add(str(g["goal_id"]))

    dl = drive_levels or {}
    d_level = 0.5
    if dominant_drive and isinstance(dl, dict):
        try:
            d_level = float(dl.get(dominant_drive, 0.5))
        except (TypeError, ValueError):
            d_level = 0.5

    rows: List[Dict[str, Any]] = []
    for ca in candidates:
        if not isinstance(ca, dict):
            continue
        total, breakdown = compute_deliberative_score(
            ca,
            current_values=current_values,
            active_goal_ids=gids,
            dominant_drive=dominant_drive,
            drive_level=d_level,
            law_violations=law_violations,
            emotion_summary=emotion_summary,
        )
        cid = str(ca.get("id") or ca.get("label") or "?")
        try:
            lr = max(0.0, min(1.0, float(ca.get("law_risk", 0.0))))
        except (TypeError, ValueError):
            lr = 0.0
        try:
            cf = max(0.0, min(1.0, float(ca.get("confidence", 0.5))))
        except (TypeError, ValueError):
            cf = 0.5
        rows.append(
            {
                "id": cid,
                "label": ca.get("label", cid),
                "score": round(total, 4),
                "law_risk": round(lr, 4),
                "confidence": round(cf, 4),
                "score_breakdown": {
                    k: round(v, 6) if isinstance(v, float) else v for k, v in breakdown.items()
                },
                "_ca": ca,
                "_tie": _tie_sort_tuple(total, lr, cf, cid),
            }
        )

    if not rows:
        return None, []

    rows.sort(key=lambda r: r["_tie"])
    best_row = rows[0]
    best = best_row["_ca"]
    board: List[Dict[str, Any]] = []
    for r in rows:
        board.append(
            {
                "id": r["id"],
                "label": r["label"],
                "score": r["score"],
                "law_risk": r["law_risk"],
                "confidence": r["confidence"],
                "score_breakdown": r["score_breakdown"],
            }
        )
    return best, board


def value_step_scale() -> float:
    """Multiplies raw value_impact deltas before applying to Doctrine weights."""
    return max(0.02, min(0.5, _env_float("HAROMA_DELIBERATIVE_VALUE_STEP", 0.12)))


def goal_delta_scale() -> float:
    """Scales goal priority deltas from candidate actions."""
    return max(0.02, min(0.4, _env_float("HAROMA_DELIBERATIVE_GOAL_STEP", 0.08)))
