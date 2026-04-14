"""
Wire contract between Haroma cognition and an **on-robot executor** (HTTP-first).

This module does **not** open sockets or talk to ROS; it defines normalized
shapes so a small bridge process can:

* POST **command batches** derived from packed-LLM ``body_actions`` to the robot.
* POST **feedback** into ``agent_environment.extensions.robot_bridge`` (merged
  by :mod:`integrations.robot_http_bridge`).

For ROS 2 deployments, map ``commands[]`` to ``action`` / ``topic`` pairs in
your bridge; keep the same JSON fields for traceability.

Env ``HAROMA_ROBOT_BRIDGE_SCHEMA`` (default ``1``): expected
``bridge_schema_version`` on batches and feedback blocks.

Architecture note: Haroma is **not** a real-time motor stack; keep torque/safety on
the robot. See ``docs/robot-cognitive-control-split.md`` for the cognitive vs RT
split, data-flow diagram, and integration checklist.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

BRIDGE_SCHEMA_VERSION = 1

# Rolling history cap in ``extensions.robot_bridge.results`` (deduped by ``command_id``).
MAX_ROBOT_BRIDGE_RESULTS = 48

# Canonical command kinds for executors (extend in lockstep with body_actions.command hints).
COMMAND_TYPES = frozenset(
    {
        "noop",
        "move_base",
        "set_pose",
        "look_at",
        "joint_targets",
        "gripper",
        "gesture",
        "speak",
        "torque",
        "estop_ack",
        "generic",
    }
)

COMMAND_STATUSES = frozenset(
    {
        "pending",
        "accepted",
        "running",
        "completed",
        "rejected",
        "failed",
        "cancelled",
    }
)


def bridge_schema_expected() -> int:
    raw = str(os.environ.get("HAROMA_ROBOT_BRIDGE_SCHEMA", "") or "").strip()
    if not raw:
        return BRIDGE_SCHEMA_VERSION
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return BRIDGE_SCHEMA_VERSION


def _canon_command_type(raw: str) -> str:
    s = (raw or "").strip().lower().replace("-", "_")
    if not s:
        return "generic"
    if s in COMMAND_TYPES:
        return s
    # Common aliases
    aliases = {
        "nav": "move_base",
        "navigate": "move_base",
        "base_move": "move_base",
        "arm": "joint_targets",
        "say": "speak",
    }
    return aliases.get(s, "generic")


def robot_command_from_body_action(
    ba: Dict[str, Any],
    seq: int,
    *,
    correlation_id: str,
) -> Dict[str, Any]:
    """Map one normalized :class:`~engine.LLMContextReasoner` body_action row to a wire command."""
    label = str(ba.get("label") or f"action_{seq}")[:200]
    cmd_hint = str(ba.get("command") or "generic")
    ctype = _canon_command_type(cmd_hint)
    cid = f"cmd_{seq}_{correlation_id[:8]}"

    pri = 0.5
    try:
        pri = max(0.0, min(1.0, float(ba.get("priority", 0.5))))
    except (TypeError, ValueError):
        pass

    out: Dict[str, Any] = {
        "command_id": cid,
        "correlation_id": correlation_id,
        "label": label,
        "type": ctype,
        "resource": str(ba.get("resource") or "none").strip().lower()[:32],
        "priority": pri,
        "cancel_previous_on_resource": bool(ba.get("cancel_current")),
        "safety_class": str(ba.get("safety_class") or "")[:48],
        "layer": str(ba.get("layer") or "")[:40],
        "supports_goal_id": str(ba.get("supports_goal_id") or "")[:80],
        "parameters": ba.get("parameters") if isinstance(ba.get("parameters"), dict) else {},
        "coordinate_frame": str(ba.get("coordinate_frame") or "")[:64],
        "rationale_excerpt": str(ba.get("rationale") or "")[:240],
    }
    if ba.get("duration_hint_sec") is not None:
        try:
            out["duration_hint_sec"] = max(0.0, min(86400.0, float(ba["duration_hint_sec"])))
        except (TypeError, ValueError):
            pass
    try:
        out["confidence"] = max(0.0, min(1.0, float(ba.get("confidence", 0.5))))
    except (TypeError, ValueError):
        out["confidence"] = 0.5
    if ba.get("preconditions"):
        pre = ba.get("preconditions")
        if isinstance(pre, list):
            out["preconditions"] = [str(p)[:120] for p in pre[:8] if str(p).strip()]
    return out


def build_executor_command_batch(
    body_actions: List[Dict[str, Any]],
    *,
    source: str = "haromax6",
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Batch payload for POST to an on-robot HTTP executor (or envelope for ROS bridge)."""
    cid = correlation_id or str(uuid.uuid4())
    cmds: List[Dict[str, Any]] = []
    for i, ba in enumerate(body_actions or []):
        if not isinstance(ba, dict):
            continue
        cmds.append(robot_command_from_body_action(ba, i, correlation_id=cid))
    return {
        "bridge_schema_version": bridge_schema_expected(),
        "source": str(source)[:64],
        "correlation_id": cid,
        "issued_at_epoch": time.time(),
        "commands": cmds,
    }


def validate_executor_command_batch(batch: Any) -> Optional[str]:
    """Return an error code string or ``None`` if *batch* is structurally valid."""
    if not isinstance(batch, dict):
        return "expected_object"
    try:
        if int(batch.get("bridge_schema_version", -1)) != bridge_schema_expected():
            return "bridge_schema_mismatch"
    except (TypeError, ValueError):
        return "bridge_schema_invalid"
    if not isinstance(batch.get("commands"), list):
        return "missing_commands"
    return None


def _normalize_bridge_result_row(r: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "command_id": str(r.get("command_id") or "")[:120],
        "status": str(r.get("status") or "").strip().lower()[:24],
        "detail": str(r.get("detail") or r.get("message") or "")[:400],
    }
    if r.get("t") is not None:
        try:
            row["t"] = float(r["t"])
        except (TypeError, ValueError):
            row["t"] = time.time()
    return row


def merge_robot_bridge_history(
    previous_robot_bridge: Optional[Dict[str, Any]],
    incoming_block: Dict[str, Any],
    *,
    max_results: int = MAX_ROBOT_BRIDGE_RESULTS,
) -> Dict[str, Any]:
    """Merge prior ``robot_bridge`` with a new feedback *incoming_block* (rolling, deduped).

    Later rows with the same ``command_id`` replace earlier ones. Results are sorted
    by ``t`` and truncated to *max_results* (keep most recent by time).
    """
    prev = dict(previous_robot_bridge) if isinstance(previous_robot_bridge, dict) else {}
    inc = dict(incoming_block) if isinstance(incoming_block, dict) else {}

    try:
        bsv = max(
            int(prev.get("bridge_schema_version") or BRIDGE_SCHEMA_VERSION),
            int(inc.get("bridge_schema_version") or BRIDGE_SCHEMA_VERSION),
        )
    except (TypeError, ValueError):
        bsv = BRIDGE_SCHEMA_VERSION

    try:
        r_prev = float(prev.get("received_at_epoch") or 0.0)
    except (TypeError, ValueError):
        r_prev = 0.0
    try:
        r_inc = float(inc.get("received_at_epoch") or time.time())
    except (TypeError, ValueError):
        r_inc = time.time()

    out: Dict[str, Any] = {
        "bridge_schema_version": bsv,
        "correlation_id": str(inc.get("correlation_id") or prev.get("correlation_id") or "")[:80],
        "received_at_epoch": max(r_prev, r_inc),
    }

    combined: List[Dict[str, Any]] = []
    for src in (prev.get("results"), inc.get("results")):
        if not isinstance(src, list):
            continue
        for r in src:
            if isinstance(r, dict) and str(r.get("command_id") or "").strip():
                combined.append(_normalize_bridge_result_row(r))

    by_id: Dict[str, Dict[str, Any]] = {}
    for row in combined:
        cid = str(row["command_id"])
        by_id[cid] = row

    ordered = sorted(
        by_id.values(),
        key=lambda x: float(x.get("t", 0.0)),
    )
    if len(ordered) > max_results:
        ordered = ordered[-max_results:]
    out["results"] = ordered
    return out


def summarize_robot_bridge(robot_bridge: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compact metrics for :meth:`agents.shared_resources.SharedResources.agent_environment_status`."""
    if not isinstance(robot_bridge, dict) or not robot_bridge:
        return {"has_bridge": False}
    res = robot_bridge.get("results")
    if not isinstance(res, list) or not res:
        return {
            "has_bridge": True,
            "results_count": 0,
            "correlation_id": str(robot_bridge.get("correlation_id") or "")[:80] or None,
        }
    counts: Dict[str, int] = {}
    last_fail: Optional[str] = None
    last_ok: Optional[str] = None
    for r in res:
        if not isinstance(r, dict):
            continue
        st = str(r.get("status") or "").strip().lower()
        counts[st] = counts.get(st, 0) + 1
        cid = str(r.get("command_id") or "").strip()
        if st in ("failed", "rejected", "cancelled"):
            last_fail = cid
        if st == "completed":
            last_ok = cid
    return {
        "has_bridge": True,
        "results_count": len(res),
        "correlation_id": str(robot_bridge.get("correlation_id") or "")[:80] or None,
        "status_counts": counts,
        "last_completed_command_id": last_ok,
        "last_failed_command_id": last_fail,
    }


def validate_feedback_entry(entry: Dict[str, Any]) -> Optional[str]:
    """Return an error string or ``None`` if *entry* is usable."""
    if not isinstance(entry, dict):
        return "entry_not_object"
    st = str(entry.get("status") or "").strip().lower()
    if st not in COMMAND_STATUSES:
        return "bad_status"
    if not str(entry.get("command_id") or "").strip():
        return "missing_command_id"
    return None


def robot_bridge_prompt_block(
    robot_bridge: Optional[Dict[str, Any]],
    *,
    max_chars: int = 2000,
) -> str:
    """Compact JSON for packed LLM ``[ROBOT BRIDGE FEEDBACK]`` (``extensions.robot_bridge``)."""
    if not isinstance(robot_bridge, dict) or not robot_bridge:
        return ""
    import json

    slim: Dict[str, Any] = {}
    sv = robot_bridge.get("bridge_schema_version", robot_bridge.get("schema_version"))
    if sv is not None:
        try:
            slim["bridge_schema_version"] = int(sv)
        except (TypeError, ValueError):
            slim["bridge_schema_version"] = BRIDGE_SCHEMA_VERSION
    cid = str(robot_bridge.get("correlation_id") or "").strip()
    if cid:
        slim["correlation_id"] = cid[:80]
    ra = robot_bridge.get("received_at_epoch")
    if ra is not None:
        try:
            slim["received_at_epoch"] = float(ra)
        except (TypeError, ValueError):
            pass
    results = robot_bridge.get("results")
    if isinstance(results, list) and results:
        tail: List[Dict[str, Any]] = []
        for r in results[-24:]:
            if not isinstance(r, dict):
                continue
            tail.append(
                {
                    "command_id": str(r.get("command_id") or "")[:100],
                    "status": str(r.get("status") or "")[:20],
                    "detail": str(r.get("detail") or "")[:160],
                }
            )
        if tail:
            slim["results"] = tail
    try:
        txt = json.dumps(slim, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        txt = "{}"
    if len(txt) > max_chars:
        txt = txt[: max_chars - 24] + "\n…[truncated]"
    return txt


def normalize_feedback_payload(raw: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """Return ``(extensions.robot_bridge block, error)``."""
    if not isinstance(raw, dict):
        return {}, "expected_object"
    sv = raw.get("bridge_schema_version", raw.get("schema_version"))
    try:
        v = int(sv) if sv is not None else BRIDGE_SCHEMA_VERSION
    except (TypeError, ValueError):
        v = BRIDGE_SCHEMA_VERSION
    if v != bridge_schema_expected():
        return {}, f"bridge_schema_mismatch_expected_{bridge_schema_expected()}"
    corr = str(raw.get("correlation_id") or "").strip()
    results = raw.get("results")
    if not isinstance(results, list):
        return {}, "missing_results_array"
    clean: List[Dict[str, Any]] = []
    for r in results[:64]:
        if not isinstance(r, dict):
            continue
        err = validate_feedback_entry(r)
        if err:
            continue
        clean.append(_normalize_bridge_result_row(r))
    block: Dict[str, Any] = {
        "bridge_schema_version": bridge_schema_expected(),
        "correlation_id": corr or str(uuid.uuid4())[:36],
        "received_at_epoch": time.time(),
        "results": clean,
    }
    return block, None
