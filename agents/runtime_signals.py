"""Cross-cutting runtime signals for HTTP and background cadence (Option B).

Depth stack and inflight count are updated together under
:class:`SharedResources` ``_http_chat_lock`` via ``http_chat_begin`` /
``http_chat_end``. Cadence reads :attr:`SharedResources.signals` only.
"""

from __future__ import annotations

import threading
import time
import weakref
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from mind.chat_priority import input_pipeline_busy

if TYPE_CHECKING:
    from agents.shared_resources import SharedResources


class RuntimeSignals:
    """Chat depth stack (parallel to inflight) + last background training time."""

    def __init__(self, shared: SharedResources) -> None:
        self._shared_ref = weakref.ref(shared)
        self._depth_stack: List[Optional[str]] = []
        self.last_background_training_at: float = 0.0
        self.last_background_training_had_effect: bool = False
        self._training_ts_lock = threading.Lock()

    def _deref(self) -> SharedResources:
        s = self._shared_ref()
        if s is None:
            raise RuntimeError("SharedResources deallocated while RuntimeSignals still live")
        return s

    def record_background_training_completed(self, had_effect: bool = False) -> None:
        """Record timestamp after a training pass; ``had_effect`` if any module returned a loss."""
        with self._training_ts_lock:
            self.last_background_training_at = time.time()
            self.last_background_training_had_effect = bool(had_effect)

    def snapshot(self) -> Dict[str, Any]:
        sh = self._deref()
        with sh._http_chat_lock:
            depths = list(self._depth_stack)
            inflight = sh._http_chat_inflight
        with self._training_ts_lock:
            lbt = self.last_background_training_at
            lbt_effect = self.last_background_training_had_effect
        return {
            "http_chat_inflight": inflight,
            "input_pipeline_busy": input_pipeline_busy(sh, None),
            "http_chat_depth_stack": depths,
            "last_background_training_at": lbt,
            "last_background_training_had_effect": lbt_effect,
        }

    def append_depth_under_http_lock(self, depth: Optional[str]) -> None:
        """Call only while holding ``shared._http_chat_lock`` (pairs with ``http_chat_begin``)."""
        self._depth_stack.append(depth)

    def pop_depth_under_http_lock(self) -> None:
        """Call only while holding ``shared._http_chat_lock`` (pairs with ``http_chat_end``)."""
        if self._depth_stack:
            self._depth_stack.pop()
