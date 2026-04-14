"""
TrainingScheduler — Prioritized replay and adaptive training for HaromaX6 (Phase 16).

PrioritizedBuffer: replaces uniform random.sample with priority-weighted
sampling where higher-surprise experiences are replayed more often.

TrainingScheduler: coordinates training frequency across modules based on
buffer fill rates and recent loss trends. Modules with plateaued loss train
less frequently; modules with high loss train more often.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional, Callable
import random
import math


class PrioritizedBuffer:
    """Priority-weighted replay buffer with importance sampling."""

    def __init__(self, capacity: int = 1024, alpha: float = 0.6):
        self._capacity = capacity
        self._alpha = alpha
        self._buffer: List[Tuple[Any, float]] = []
        self._max_priority = 1.0

    @property
    def size(self) -> int:
        return len(self._buffer)

    def add(self, sample: Any, priority: Optional[float] = None):
        p = abs(priority) if priority is not None else self._max_priority
        p = max(p, 1e-6)
        self._max_priority = max(self._max_priority, p)
        self._buffer.append((sample, p))
        if len(self._buffer) > self._capacity:
            self._buffer = self._buffer[-self._capacity :]

    def sample(self, batch_size: int) -> Tuple[List[Any], List[float]]:
        """Return (samples, importance_weights)."""
        if not self._buffer:
            return [], []

        n = min(batch_size, len(self._buffer))
        priorities = [entry[1] ** self._alpha for entry in self._buffer]
        total = sum(priorities)
        probs = [p / total for p in priorities]

        indices = []
        seen = set()
        for _ in range(n):
            r = random.random()
            cumulative = 0.0
            for i, p in enumerate(probs):
                cumulative += p
                if r <= cumulative and i not in seen:
                    indices.append(i)
                    seen.add(i)
                    break
            else:
                for i in range(len(self._buffer)):
                    if i not in seen:
                        indices.append(i)
                        seen.add(i)
                        break

        if not indices:
            indices = random.sample(range(len(self._buffer)), min(n, len(self._buffer)))

        N = len(self._buffer)
        min_prob = min(probs) if probs else 1.0
        samples = []
        weights = []
        for idx in indices:
            samples.append(self._buffer[idx][0])
            w = (1.0 / (N * max(probs[idx], 1e-10))) / (1.0 / (N * max(min_prob, 1e-10)))
            weights.append(w)

        return samples, weights

    def update_priority(self, indices: List[int], new_priorities: List[float]):
        for idx, p in zip(indices, new_priorities):
            if 0 <= idx < len(self._buffer):
                sample = self._buffer[idx][0]
                p_abs = max(abs(p), 1e-6)
                self._buffer[idx] = (sample, p_abs)
                self._max_priority = max(self._max_priority, p_abs)

    def clear(self):
        self._buffer.clear()
        self._max_priority = 1.0

    def as_list(self) -> List[Any]:
        return [entry[0] for entry in self._buffer]


class _ModuleRecord:
    """Tracks training state for a single module."""

    def __init__(self, name: str, base_interval: int = 5):
        self.name = name
        self.base_interval = base_interval
        self.loss_history: List[float] = []
        self._loss_window = 20
        self.cycles_since_train = 0
        self.total_trains = 0

    def record_loss(self, loss: float):
        self.loss_history.append(loss)
        if len(self.loss_history) > self._loss_window * 2:
            self.loss_history = self.loss_history[-self._loss_window :]
        self.total_trains += 1

    @property
    def recent_avg_loss(self) -> float:
        if not self.loss_history:
            return float("inf")
        recent = [v for v in self.loss_history[-self._loss_window :] if v is not None]
        if not recent:
            return float("inf")
        return sum(recent) / len(recent)

    @property
    def is_plateaued(self) -> bool:
        if len(self.loss_history) < self._loss_window:
            return False
        first_half = [
            v
            for v in self.loss_history[-self._loss_window : -self._loss_window // 2]
            if v is not None
        ]
        second_half = [v for v in self.loss_history[-self._loss_window // 2 :] if v is not None]
        if not first_half or not second_half:
            return False
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        return abs(avg_first - avg_second) < 0.01 * max(avg_first, 1e-6)

    def adaptive_interval(self) -> int:
        if not self.loss_history:
            return self.base_interval
        if self.is_plateaued:
            return min(self.base_interval * 4, 40)
        if self.recent_avg_loss > 1.0:
            return max(1, self.base_interval // 2)
        return self.base_interval


class TrainingScheduler:
    """Coordinates training frequency across modules."""

    def __init__(self):
        self._modules: Dict[str, _ModuleRecord] = {}

    def register_module(self, name: str, base_interval: int = 5):
        self._modules[name] = _ModuleRecord(name, base_interval)

    def should_train(self, name: str) -> bool:
        rec = self._modules.get(name)
        if rec is None:
            return True
        rec.cycles_since_train += 1
        interval = rec.adaptive_interval()
        if rec.cycles_since_train >= interval:
            rec.cycles_since_train = 0
            return True
        return False

    def record_loss(self, name: str, loss: float):
        rec = self._modules.get(name)
        if rec is not None:
            rec.record_loss(loss)
            rec.cycles_since_train = 0

    def stats(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "registered_modules": len(self._modules),
        }
        for name, rec in self._modules.items():
            info[name] = {
                "total_trains": rec.total_trains,
                "recent_avg_loss": round(rec.recent_avg_loss, 5),
                "plateaued": rec.is_plateaued,
                "adaptive_interval": rec.adaptive_interval(),
            }
        return info

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for name, rec in self._modules.items():
            data[name] = {
                "base_interval": rec.base_interval,
                "loss_history": rec.loss_history[-40:],
                "total_trains": rec.total_trains,
                "cycles_since_train": rec.cycles_since_train,
            }
        return data

    def from_dict(self, data: Dict[str, Any]):
        for name, rec_data in data.items():
            if name not in self._modules:
                self._modules[name] = _ModuleRecord(name, rec_data.get("base_interval", 5))
            rec = self._modules[name]
            rec.base_interval = rec_data.get("base_interval", rec.base_interval)
            rec.loss_history = rec_data.get("loss_history", [])
            rec.total_trains = rec_data.get("total_trains", 0)
            rec.cycles_since_train = rec_data.get("cycles_since_train", 0)
