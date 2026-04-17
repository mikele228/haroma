"""Single source of truth for background-training defer policy (env → cadence + /status).

``HAROMA_BG_DEFER_TRAINING_ON_INPUT_PIPELINE`` (preferred) controls defer while the
**input pipeline** is busy (HTTP ``/chat`` lifecycle plus :class:`~agents.input_agent.InputAgent`
queues). When unset, ``HAROMA_BG_DEFER_TRAINING_ON_HTTP_CHAT`` is used (legacy name).
"""

from __future__ import annotations

import os
from typing import Optional


def defer_training_on_input_pipeline() -> bool:
    """When True, background training may defer while the input pipeline is busy."""
    raw_new = os.environ.get("HAROMA_BG_DEFER_TRAINING_ON_INPUT_PIPELINE")
    if raw_new is not None and str(raw_new).strip() != "":
        return str(raw_new).strip().lower() in ("1", "true", "yes", "on")
    raw_legacy = str(os.environ.get("HAROMA_BG_DEFER_TRAINING_ON_HTTP_CHAT", "1") or "").strip().lower()
    return raw_legacy in ("1", "true", "yes", "on")


def defer_training_on_http_chat() -> bool:
    """Same as :func:`defer_training_on_input_pipeline` (legacy name)."""
    return defer_training_on_input_pipeline()


def bg_training_defer_cap_sec() -> Optional[float]:
    """Unset/empty → ``None``; valid number → ``>= 0`` (including explicit ``0``); invalid → ``None``."""
    raw = str(os.environ.get("HAROMA_BG_DEFER_TRAINING_CAP_SEC", "") or "").strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def bg_training_defer_cap_effective_seconds() -> float:
    """Value used by :class:`agents.background_cadence.BackgroundCadence` (0 = no periodic bypass)."""
    v = bg_training_defer_cap_sec()
    return float(v) if v is not None else 0.0
