"""Cooperative shared-lock hold time budget (observability + optional assert).

Python cannot forcibly release locks from another thread. Call sites must keep
critical sections short. **Hold time** (seconds the lock is actually held after
a successful acquire) is compared to the budget. **Wait time** (blocking before
acquire) is reported separately when ``HAROMA_SHARED_LOCK_LOG_WAIT=1``.

Env:
  HAROMA_SHARED_LOCK_BUDGET_SEC — default ``1.0``; ``0`` or negative disables checks.
  HAROMA_SHARED_LOCK_BUDGET_MODE — ``warn`` (default), ``off``, or ``assert``.
  HAROMA_SHARED_LOCK_LOG_WAIT — ``1`` to print waits when wait or hold logging is useful.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional


def shared_lock_budget_sec() -> Optional[float]:
    raw = (os.environ.get("HAROMA_SHARED_LOCK_BUDGET_SEC") or "").strip()
    if raw == "":
        return 1.0
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if v <= 0:
        return None
    return v


def shared_lock_budget_mode() -> str:
    m = (os.environ.get("HAROMA_SHARED_LOCK_BUDGET_MODE") or "warn").strip().lower()
    if m in ("off", "warn", "assert"):
        return m
    return "warn"


def shared_lock_log_wait_enabled() -> bool:
    return str(os.environ.get("HAROMA_SHARED_LOCK_LOG_WAIT", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def report_shared_lock_section(
    name: str,
    *,
    wait_sec: float,
    hold_sec: float,
    cognitive_metrics: Optional[Any] = None,
) -> None:
    """Warn/assert on **hold** time over budget; optional wait log; optional metrics bump.

    *wait_sec* is time blocked before the lock was acquired (not counted against budget).
    *hold_sec* is time while the lock was held after acquire (this is what the budget limits).
    """
    budget = shared_lock_budget_sec()
    mode = shared_lock_budget_mode()
    if shared_lock_log_wait_enabled() and wait_sec >= 0.005:
        print(
            f"[SharedLockBudget] {name} wait={wait_sec:.3f}s before acquire",
            flush=True,
        )
    if mode == "off" or budget is None:
        return
    if hold_sec <= budget:
        return
    msg = (
        f"[SharedLockBudget] {name} hold={hold_sec:.3f}s "
        f"(budget {budget:.3f}s; wait was {wait_sec:.3f}s) — shorten critical section "
        f"or split work across ticks"
    )
    if cognitive_metrics is not None:
        rec = getattr(cognitive_metrics, "record_shared_lock_over_budget", None)
        if callable(rec):
            try:
                rec(name)
            except Exception:
                pass
    if mode == "assert":
        raise AssertionError(msg)
    print(msg, flush=True)


# Back-compat for tests that monkeypatch ``report_shared_lock_hold``
def report_shared_lock_hold(name: str, seconds: float) -> None:
    """Legacy: treat *seconds* as hold time only, zero wait."""
    report_shared_lock_section(name, wait_sec=0.0, hold_sec=seconds, cognitive_metrics=None)
