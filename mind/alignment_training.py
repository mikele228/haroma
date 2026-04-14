"""Alignment-aware reward blending and logging for continuous training.

Blends outcome evaluator scores with deliberative pillar signals (values, goals,
beliefs, law penalties, emotion fit) so ``LLMBackend.record_outcome`` and JSONL
exports reflect value / belief / goal / law / emotion alignment—not only
surface outcome.

Env
---
* ``HAROMA_ALIGNMENT_REWARD_BLEND`` — weight in ``[0,1]`` for deliberative
  signal vs raw outcome (default ``0.35``). Set ``0`` to disable blending.
* ``HAROMA_ALIGNMENT_LOG`` — ``1``/``true`` to append NDJSON lines to
  ``data/alignment/events.ndjson`` (default ``1``).
* ``HAROMA_ALIGNMENT_LOG_PATH`` — override log file path.
"""

from __future__ import annotations

import json
import math
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from mind.config_env import env_float, env_truthy

_LOG_LOCK = threading.Lock()
_DEFAULT_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "alignment",
    "events.ndjson",
)


def _score_to_unit(x: float) -> float:
    """Map deliberative total score (roughly unbounded) to ``[0, 1]``."""
    try:
        t = math.tanh(float(x) * 0.35)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, 0.5 + 0.5 * t))


def _winner_row(
    board: List[Dict[str, Any]],
    chosen: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not board or not chosen:
        return None
    cid = str(chosen.get("id") or chosen.get("label") or "")
    for row in board:
        if str(row.get("id") or row.get("label") or "") == cid:
            return row
    # Do not fall back to board[0]: id mismatch would mis-attribute deliberative scores.
    return None


def compute_blended_alignment_reward(
    outcome_score: float,
    llm_context_result: Optional[Dict[str, Any]],
    *,
    episode: Any = None,
) -> Tuple[float, Dict[str, Any]]:
    """Return (reward_for_training, diagnostics) with reward in ``[0, 1]``."""
    try:
        oc = max(0.0, min(1.0, float(outcome_score)))
    except (TypeError, ValueError):
        oc = 0.5

    blend_w = max(0.0, min(1.0, env_float("HAROMA_ALIGNMENT_REWARD_BLEND", 0.35)))
    diag: Dict[str, Any] = {
        "outcome_score": oc,
        "blend_weight": blend_w,
        "deliberative_used": False,
        "deliberative_unit": None,
        "chosen_id": None,
    }

    lc = llm_context_result if isinstance(llm_context_result, dict) else {}
    board = lc.get("deliberative_scores")
    chosen = lc.get("chosen_action")
    if not isinstance(board, list) or not board:
        return oc, diag
    if not isinstance(chosen, dict):
        return oc, diag

    row = _winner_row(board, chosen)
    if not row:
        return oc, diag

    try:
        total = float(row.get("score", 0.0))
    except (TypeError, ValueError):
        total = 0.0
    du = _score_to_unit(total)
    diag["deliberative_used"] = True
    diag["deliberative_unit"] = du
    diag["chosen_id"] = row.get("id")
    sb = row.get("score_breakdown")
    if isinstance(sb, dict):
        diag["pillars"] = {k: round(float(v), 4) if isinstance(v, (int, float)) else v for k, v in sb.items()}

    if blend_w <= 0.0:
        return oc, diag

    blended = (1.0 - blend_w) * oc + blend_w * du
    blended = max(0.0, min(1.0, blended))
    diag["blended_reward"] = blended
    return blended, diag


def log_alignment_event(record: Dict[str, Any]) -> None:
    """Append one NDJSON object (thread-safe)."""
    if not env_truthy("HAROMA_ALIGNMENT_LOG", True):
        return
    path = os.environ.get("HAROMA_ALIGNMENT_LOG_PATH") or _DEFAULT_LOG
    try:
        _d = os.path.dirname(path)
        if _d:
            os.makedirs(_d, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with _LOG_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass
