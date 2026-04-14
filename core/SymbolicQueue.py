"""SymbolicQueue — two-slot queue with namespace ownership and hash-based drift control.

Slot 0 holds generator outputs; slot 1 holds processor outputs.
Each entry is keyed by ``role.module.tree.key`` and carries a value + MD5 hash.
If a write produces the same hash as the prior cycle, the entry is marked stale
so consumers can skip feedback loops.

Also contains ``SymbolicFingerprintEngine`` for cross-cycle stale detection.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from core.CellRoles import GENERATOR, PROCESSOR, CONSUMER


def _md5(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


class _QueueEntry:
    __slots__ = ("namespace", "key", "value", "hash", "stale_cycles", "timestamp")

    def __init__(self, namespace: str, key: str, value: Any, hash_val: str):
        self.namespace = namespace
        self.key = key
        self.value = value
        self.hash = hash_val
        self.stale_cycles = 0
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "namespace": self.namespace,
            "key": self.key,
            "value_preview": str(self.value)[:120],
            "hash": self.hash,
            "stale_cycles": self.stale_cycles,
        }


class SymbolicQueue:
    """Two-slot queue with namespace ownership and hash-based drift control."""

    SLOT_GENERATOR = 0
    SLOT_PROCESSOR = 1

    def __init__(self, stale_threshold: int = 3):
        self._lock = threading.Lock()
        self._slots: Tuple[Dict[str, _QueueEntry], Dict[str, _QueueEntry]] = ({}, {})
        self._stale_threshold = stale_threshold
        self._write_count = 0
        self._stale_suppressed = 0

    def write(self, slot: int, namespace: str, key: str, value: Any) -> bool:
        """Write to a queue slot. Returns False if value is stale (unchanged hash)."""
        fq_key = f"{namespace}.{key}"
        new_hash = _md5(value)

        with self._lock:
            existing = self._slots[slot].get(fq_key)
            if existing and existing.hash == new_hash:
                existing.stale_cycles += 1
                self._stale_suppressed += 1
                return False

            entry = _QueueEntry(namespace, key, value, new_hash)
            self._slots[slot][fq_key] = entry
            self._write_count += 1
            return True

    def read(self, slot: int, *, namespace_filter: Optional[str] = None) -> Dict[str, Any]:
        """Return non-stale entries from a slot."""
        with self._lock:
            result: Dict[str, Any] = {}
            for fq_key, entry in self._slots[slot].items():
                if entry.stale_cycles >= self._stale_threshold:
                    continue
                if namespace_filter and not entry.namespace.startswith(namespace_filter):
                    continue
                result[fq_key] = entry.value
            return result

    def drain(self, slot: int) -> Dict[str, Any]:
        """Return all pending entries from a slot and clear it."""
        with self._lock:
            result: Dict[str, Any] = {}
            for fq_key, entry in self._slots[slot].items():
                if entry.stale_cycles < self._stale_threshold:
                    result[fq_key] = entry.value
            self._slots[slot].clear()
            return result

    def flush_stale(self) -> int:
        """Remove entries unchanged for >= stale_threshold cycles. Returns count removed."""
        with self._lock:
            removed = 0
            for slot_dict in self._slots:
                stale_keys = [
                    k for k, e in slot_dict.items() if e.stale_cycles >= self._stale_threshold
                ]
                for k in stale_keys:
                    del slot_dict[k]
                    removed += 1
            return removed

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "slot_0_entries": len(self._slots[0]),
                "slot_1_entries": len(self._slots[1]),
                "slot_0_role": "generator",
                "slot_1_role": "processor",
                "consumer_role_mask": CONSUMER,
                "x7_role_masks": {
                    "generator": GENERATOR,
                    "processor": PROCESSOR,
                    "consumer": CONSUMER,
                },
                "total_writes": self._write_count,
                "stale_suppressed": self._stale_suppressed,
            }


# ═══════════════════════════════════════════════════════════════════════
# SymbolicFingerprintEngine
# ═══════════════════════════════════════════════════════════════════════


class SymbolicFingerprintEngine:
    """Tracks per-module output hashes across cycles for drift/stagnation detection."""

    def __init__(self, stagnation_threshold: int = 5):
        self._lock = threading.Lock()
        self._fingerprints: Dict[str, str] = {}
        self._unchanged_counts: Dict[str, int] = {}
        self._stagnation_threshold = stagnation_threshold
        self._total_checks = 0

    def is_novel(self, module: str, output: Any) -> bool:
        """Return True if the module's output has changed since last cycle."""
        new_hash = _md5(output)
        with self._lock:
            self._total_checks += 1
            prev = self._fingerprints.get(module)
            self._fingerprints[module] = new_hash
            if prev == new_hash:
                self._unchanged_counts[module] = self._unchanged_counts.get(module, 0) + 1
                return False
            self._unchanged_counts[module] = 0
            return True

    def is_stagnant(self, module: str) -> bool:
        """Return True if module output hasn't changed for >= threshold cycles."""
        with self._lock:
            return self._unchanged_counts.get(module, 0) >= self._stagnation_threshold

    def stagnation_report(self) -> Dict[str, int]:
        """Return dict of module -> unchanged_cycle_count for stagnant modules."""
        with self._lock:
            return {
                m: c for m, c in self._unchanged_counts.items() if c >= self._stagnation_threshold
            }

    def snapshot(self) -> Dict[str, str]:
        """Return current fingerprints (for persistence)."""
        with self._lock:
            return dict(self._fingerprints)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            stagnant = sum(
                1 for c in self._unchanged_counts.values() if c >= self._stagnation_threshold
            )
            return {
                "tracked_modules": len(self._fingerprints),
                "total_checks": self._total_checks,
                "stagnant_modules": stagnant,
            }
