"""Centralized eligibility for background training and web-learn ticks.

Reads :class:`agents.runtime_signals.RuntimeSignals` and environment only —
no Flask imports. :class:`BackgroundAgent` delegates defer/cap and web tick
math here.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

from mind.bg_training_env import (
    bg_training_defer_cap_effective_seconds,
    defer_training_on_input_pipeline,
)
from mind.chat_priority import input_pipeline_busy

if TYPE_CHECKING:
    from agents.shared_resources import SharedResources
    from engine.WebLearnCrawler import WebLearnCrawler


class BackgroundCadence:
    def __init__(self, shared: SharedResources, boot_agent: Optional[Any] = None) -> None:
        self._shared = shared
        self._boot_agent = boot_agent

    def should_run_training_now(self) -> bool:
        """Defer while input pipeline busy + optional time cap (``HAROMA_BG_DEFER_TRAINING_CAP_SEC``)."""
        s = self._shared
        sig = s.signals
        if not defer_training_on_input_pipeline():
            return True
        try:
            if not input_pipeline_busy(s, self._boot_agent):
                return True
        except Exception:
            # Do not assume idle when pipeline state is unreadable (see input_pipeline_busy).
            pass
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
