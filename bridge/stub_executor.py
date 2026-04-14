"""
Stub on-robot executor: maps a Haroma command batch to feedback rows (no ROS, no sockets).

Replace ``simulate_command_results`` with calls into your motion stack / ROS 2 actions.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List


def simulate_command_results(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return one feedback row per command (always ``completed`` in this stub)."""
    t = time.time()
    rows: List[Dict[str, Any]] = []
    for c in batch.get("commands") or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("command_id") or "").strip()
        if not cid:
            continue
        rows.append(
            {
                "command_id": cid,
                "status": "completed",
                "detail": "stub_executor: accepted (replace with real controller)",
                "t": t,
            }
        )
    return rows


def feedback_block_from_results(
    batch: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Shape expected by :func:`mind.robot_execution_contract.normalize_feedback_payload`."""
    try:
        bsv = int(batch.get("bridge_schema_version", 1))
    except (TypeError, ValueError):
        bsv = 1
    corr = str(batch.get("correlation_id") or "").strip()
    return {
        "bridge_schema_version": bsv,
        "correlation_id": corr,
        "results": results,
    }
