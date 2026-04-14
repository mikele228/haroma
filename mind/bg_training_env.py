"""Single source of truth for background-training defer policy (env → cadence + /status)."""

from __future__ import annotations

import os
from typing import Optional


def defer_training_on_http_chat() -> bool:
    """When True, background training may defer while HTTP chat slots are in flight."""
    raw = str(os.environ.get("HAROMA_BG_DEFER_TRAINING_ON_HTTP_CHAT", "1") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


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
