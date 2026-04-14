"""
Humanoid brain-like integrative state (engineering analog, not neuroscience).

Fuses signals that are already computed each cycle into a **single snapshot**
similar in role to combined limbic + neuromodulatory gating: how alert the
system is, how much it should explore, how stable executive control is, how
strongly to allow plasticity this moment, and how much consolidation pressure
has built up (sleep/consolidation analog).

Downstream code can read ``episode.brain_like_state`` instead of re-deriving
ad-hoc weights from scattered fields.

**Not** a claim of consciousness or human equivalence — a **coordination layer**
for multi-module cognition.
"""

from __future__ import annotations

import os
from typing import Any, Dict


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _f(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


def compute_brain_like_state(
    *,
    affect: Dict[str, Any],
    curiosity: Dict[str, Any],
    drives: Dict[str, Any],
    dominant_drive: str,
    embodied_modulation: Dict[str, Any],
    self_surprise: Dict[str, Any],
    appraisal: Dict[str, Any],
    drift_score: float = 0.0,
    prediction_error: float = 0.0,
) -> Dict[str, Any]:
    """Return a stable dict bound on ``EpisodeContext`` each cycle.

    Keys are floats in ``[0, 1]`` unless noted.
    """
    intensity = _f(affect, "intensity")
    arousal = abs(_f(affect, "arousal"))
    valence = _f(affect, "valence")

    novelty = _f(curiosity, "novelty_score")
    uncertainty = _f(curiosity, "uncertainty_score")
    pe = max(0.0, float(prediction_error or 0.0))

    surprise = _f(self_surprise, "overall_surprise")
    acc = _f(self_surprise, "accuracy")

    # Drive "pressure": how far from rest the dominant need is (rough analog).
    dom_level = 0.35
    if isinstance(drives, dict) and dominant_drive and dominant_drive in drives:
        try:
            dom_level = float(drives[dominant_drive])
        except (TypeError, ValueError):
            dom_level = 0.35
    elif isinstance(drives, dict) and drives:
        try:
            dom_level = max(float(v) for v in drives.values() if isinstance(v, (int, float)))
        except ValueError:
            dom_level = 0.35

    appr_rel = _f(appraisal, "relevance")
    appr_override = 1.0 if appraisal.get("overrides") else 0.0

    nov_bias = _f(embodied_modulation, "novelty_bias")
    cur_damp = _f(embodied_modulation, "curiosity_damping", 1.0)
    if cur_damp <= 0:
        cur_damp = 1.0

    # --- Indices (interpretable composites) ---
    arousal_index = _clamp(0.45 * intensity + 0.35 * arousal + 0.2 * appr_rel)

    exploration_index = _clamp(
        0.35 * _clamp(pe * 3.0)
        + 0.25 * novelty
        + 0.2 * uncertainty
        + 0.1 * max(0.0, nov_bias) * 5.0
        + 0.1 * (1.0 / cur_damp) * 0.5
    )

    stress = _clamp(0.4 * surprise + 0.25 * drift_score + 0.2 * dom_level + 0.15 * appr_override)
    stability_index = _clamp(1.0 - stress)

    # Plasticity: learn more when prediction fails or world is novel; less when very stable and accurate.
    plasticity_index = _clamp(
        0.45 * exploration_index + 0.35 * surprise + 0.2 * (1.0 - _clamp(acc))
    )

    consolidation_pressure = _clamp(
        0.35 * surprise + 0.25 * drift_score + 0.2 * intensity + 0.2 * novelty
    )

    # Valence-tinged "mood" for logging (-1..1)
    mood_tilt = _clamp(valence, -1.0, 1.0)

    narrative = (
        f"alert={arousal_index:.2f} explore={exploration_index:.2f} "
        f"stable={stability_index:.2f} plasticity={plasticity_index:.2f} "
        f"consolidate≈{consolidation_pressure:.2f} mood={mood_tilt:+.2f}"
    )

    return {
        "schema_version": 1,
        "arousal_index": round(arousal_index, 4),
        "exploration_index": round(exploration_index, 4),
        "stability_index": round(stability_index, 4),
        "plasticity_index": round(plasticity_index, 4),
        "consolidation_pressure": round(consolidation_pressure, 4),
        "mood_tilt": round(mood_tilt, 4),
        "dominant_drive_level": round(dom_level, 4),
        "narrative": narrative,
    }


def maybe_log_brain_state(agent_id: str, state: Dict[str, Any]) -> None:
    if str(os.environ.get("HAROMA_BRAIN_STATE_LOG", "") or "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    nar = state.get("narrative", "")
    print(f"[BrainState] {agent_id} | {nar}", flush=True)
