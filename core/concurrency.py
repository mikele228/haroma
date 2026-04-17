"""
Cross-cutting locks for SharedResources and the cognitive stack.

Lock ordering (when holding more than one lock, acquire in this order — lowest
index first)::

    http_chat → cycle → neural → autonomy_metrics → autonomy_stimulus
    → reward_replay → process_gate_pending → imagination_buffer
    → language_composer_data

Higher layers (HTTP counters, cycle id) come before long-running neural work;
autonomy bookkeeping before per-module training-buffer locks. Training-buffer
locks are independent of each other — do not nest order_gate ↔ imagination
unless you first sort keys with :meth:`ConcurrencyCoordinator.acquire_ordered`.

Subsystems embed their own locks (e.g. MemoryForest, MessageBus); this module
only centralizes locks shared at boot and optional ordered multi-lock use.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# When nesting multiple coordinator locks, acquire in this order (first listed first).
LOCK_ORDER: Tuple[str, ...] = (
    "http_chat",
    "cycle",
    "neural",
    "autonomy_metrics",
    "autonomy_stimulus",
    "reward_replay",
    "process_gate_pending",
    "imagination_buffer",
    "language_composer_data",
)

_LockTrace = Optional[Callable[[str, str], None]]


@dataclass
class NeuralTrainingLocks:
    """Locks for replay / pending buffers mutated during record_* vs train_step."""

    reward_replay: threading.Lock
    process_gate_pending: threading.Lock
    imagination_buffer: threading.Lock
    language_composer_data: threading.Lock


class NeuralRWLock:
    """Readers–writer lock for shared neural weights.

    *Inference* (input encoder, persona forwards) takes the **read** side so
    multiple threads can overlap. *Background training* takes the **write** side
    exclusively so optimizer updates never race forward passes.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False

    def acquire_read(self) -> None:
        with self._cond:
            while self._writer:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            while self._readers > 0 or self._writer:
                self._cond.wait()
            self._writer = True

    def release_write(self) -> None:
        with self._cond:
            self._writer = False
            self._cond.notify_all()


class ConcurrencyCoordinator:
    """Owns shared Lock/RLock instances and provides ordered multi-lock helpers."""

    def __init__(self, trace: _LockTrace = None):
        self.http_chat = threading.Lock()
        self.cycle = threading.Lock()
        self.neural = threading.RLock()
        self.autonomy_metrics = threading.Lock()
        self.autonomy_stimulus = threading.Lock()
        self.training = NeuralTrainingLocks(
            reward_replay=threading.Lock(),
            process_gate_pending=threading.Lock(),
            imagination_buffer=threading.Lock(),
            language_composer_data=threading.Lock(),
        )
        self._trace = trace
        self._order_index: Dict[str, int] = {k: i for i, k in enumerate(LOCK_ORDER)}
        self._all_locks: Dict[str, Any] = {
            "http_chat": self.http_chat,
            "cycle": self.cycle,
            "neural": self.neural,
            "autonomy_metrics": self.autonomy_metrics,
            "autonomy_stimulus": self.autonomy_stimulus,
            "reward_replay": self.training.reward_replay,
            "process_gate_pending": self.training.process_gate_pending,
            "imagination_buffer": self.training.imagination_buffer,
            "language_composer_data": self.training.language_composer_data,
        }

    def _emit(self, event: str, key: str) -> None:
        if self._trace is not None:
            try:
                self._trace(event, key)
            except Exception:
                pass

    @contextlib.contextmanager
    def acquire_ordered(self, *keys: str) -> Iterator[None]:
        """Acquire named locks in global order; release in reverse order."""
        for k in keys:
            if k not in self._all_locks:
                raise KeyError(f"unknown lock key: {k!r}; known: {sorted(self._all_locks)}")
        sorted_keys: List[str] = sorted(keys, key=lambda x: self._order_index[x])
        acquired: List[threading.Lock] = []
        try:
            for k in sorted_keys:
                lk = self._all_locks[k]
                self._emit("acquire", k)
                lk.acquire()
                acquired.append(lk)
            yield
        finally:
            for lk in reversed(acquired):
                lk.release()
