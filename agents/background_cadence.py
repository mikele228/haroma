"""Centralized eligibility for background training and web-learn ticks.

Reads :class:`agents.runtime_signals.RuntimeSignals` and environment only —
no Flask imports. :class:`BackgroundAgent` delegates defer/cap and web tick
math here.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from mind.bg_training_env import (
    bg_training_defer_cap_effective_seconds,
    defer_training_on_http_chat,
)

if TYPE_CHECKING:
    from agents.shared_resources import SharedResources
    from engine.WebLearnCrawler import WebLearnCrawler


class BackgroundCadence:
    def __init__(self, shared: SharedResources) -> None:
        self._shared = shared

    def should_run_training_now(self) -> bool:
        """Defer-on-chat + optional time cap (``HAROMA_BG_DEFER_TRAINING_CAP_SEC``)."""
        s = self._shared
        sig = s.signals
        if not defer_training_on_http_chat():
            return True
        try:
            if s.http_chat_inflight <= 0:
                return True
        except Exception:
            return True
        _cap_sec = bg_training_defer_cap_effective_seconds()
        if _cap_sec <= 0:
            return False
        try:
            last = float(sig.last_background_training_at)
        except Exception:
            last = 0.0
        return (time.time() - last) >= _cap_sec

    def should_run_web_learn(self, bg_tick_count: int, crawler: WebLearnCrawler) -> bool:
        return (
            bool(getattr(crawler, "enabled", False))
            and bg_tick_count > 0
            and bg_tick_count % max(1, int(getattr(crawler, "every_n_ticks", 1))) == 0
        )
