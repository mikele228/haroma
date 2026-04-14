"""
General-agent environment — structured world state from any host (home, game,
office, custom). Validates and normalizes JSON; HaromaX6 does not execute tools here.

Integrators POST snapshots via ``POST /agent/environment`` or embed
``agent_environment`` on ``POST /chat``. The cognitive loop binds the latest
snapshot onto ``EpisodeContext`` so reasoning / packed LLM see facts, not only text.
For how summaries enter the packed LLM user message, see
:mod:`mind.packed_llm_context` (environment + robot body sections).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

_MAX_JSON_BYTES = int(os.environ.get("HAROMA_AGENT_ENV_MAX_BYTES", "262144"))


def _clamp_size(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        blob = json.dumps(raw, ensure_ascii=False, default=str).encode("utf-8")
    except (TypeError, ValueError):
        return {}, "not_json_serializable"
    if len(blob) > _MAX_JSON_BYTES:
        return {}, f"payload_too_large_max_{_MAX_JSON_BYTES}_bytes"
    return raw, None


def validate_agent_environment(raw: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """Return ``(normalized_dict, error_or_none)``."""
    if raw is None:
        return {}, None
    if not isinstance(raw, dict):
        return {}, "expected_object"
    raw, err = _clamp_size(raw)
    if err:
        return {}, err
    sv = raw.get("schema_version", 1)
    try:
        schema_version = int(sv)
    except (TypeError, ValueError):
        schema_version = 1
    domain = str(raw.get("domain", "") or "").strip()[:64]
    entities = raw.get("entities")
    if entities is not None and not isinstance(entities, dict):
        entities = {}
    metrics = raw.get("metrics")
    if metrics is not None and not isinstance(metrics, dict):
        metrics = {}
    alerts = raw.get("alerts")
    if alerts is None or not isinstance(alerts, list):
        alerts = []
    extensions = raw.get("extensions")
    if extensions is not None and not isinstance(extensions, dict):
        extensions = {}
    ts = raw.get("timestamp")
    try:
        client_ts = float(ts) if ts is not None else time.time()
    except (TypeError, ValueError):
        client_ts = time.time()

    normalized = {
        "schema_version": max(1, min(999, schema_version)),
        "domain": domain or "custom",
        "entities": entities or {},
        "metrics": metrics or {},
        "alerts": alerts[:200],
        "extensions": extensions or {},
        "timestamp": client_ts,
        "received_at": time.time(),
    }
    norm_blob = json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str)
    normalized["fingerprint"] = hashlib.sha256(norm_blob.encode("utf-8")).hexdigest()[:16]
    return normalized, None


def environment_summary_for_prompt(env: Dict[str, Any], max_chars: int = 4000) -> str:
    """Compact text for optional inclusion in packed prompts."""
    if not env:
        return ""
    parts = [
        f"domain={env.get('domain', '?')}",
        f"entities={len(env.get('entities') or {})}",
        f"metrics={len(env.get('metrics') or {})}",
        f"alerts={len(env.get('alerts') or [])}",
    ]
    ext = env.get("extensions")
    if isinstance(ext, dict) and ext:
        parts.append(f"extensions_keys={list(ext.keys())[:12]}")
        rb = ext.get("robot_body")
        if isinstance(rb, dict):
            if rb.get("body_defined") is False:
                parts.append("robot_body=undefined(no_sensor_history)")
            elif rb.get("body_defined") is True and isinstance(rb.get("readings"), dict):
                parts.append(f"robot_body_keys={list(rb['readings'].keys())[:16]}")
        bridge = ext.get("robot_bridge")
        if isinstance(bridge, dict) and bridge:
            from mind.robot_execution_contract import summarize_robot_bridge

            sm = summarize_robot_bridge(bridge)
            if sm.get("has_bridge"):
                n = int(sm.get("results_count") or 0)
                parts.append(f"robot_bridge_results≈{n}")
                cid = sm.get("correlation_id")
                if cid:
                    parts.append(f"robot_bridge_corr={str(cid)[:24]}")
                sc = sm.get("status_counts") or {}
                if sc:
                    tail = ",".join(f"{k}:{v}" for k, v in list(sc.items())[:6])
                    parts.append(f"robot_bridge_status={tail}")
    s = " | ".join(parts)
    if len(s) > max_chars:
        return s[: max_chars - 3] + "..."
    return s


def propose_structured_actions(
    *,
    episode: Any,
    action: Dict[str, Any],
    outcome: Dict[str, Any],
    agent_environment: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Heuristic tool proposals for integrators (host executes). Conservative default."""
    if str(os.environ.get("HAROMA_AGENT_STRUCTURED_ACTIONS", "1") or "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return []
    out: List[Dict[str, Any]] = []
    domain = str((agent_environment or {}).get("domain", "") or "").lower()
    score = 0.5
    try:
        score = float(outcome.get("score", 0.5) or 0.5)
    except (TypeError, ValueError):
        pass

    strat = str(action.get("strategy", "") or "")
    text = str(action.get("text", "") or "")[:240]

    if score >= 0.45 and strat in ("inform", "inquire", "empathize", "reflect"):
        out.append(
            {
                "tool": "notify.user",
                "args": {
                    "channel": "default",
                    "summary": text[:200],
                    "priority": "low" if score < 0.65 else "normal",
                },
                "risk_tier": "low",
                "idempotency_key": f"cycle:{getattr(episode, 'cycle_id', 0)}:notify",
            }
        )

    if domain == "home" and (agent_environment or {}).get("entities"):
        out.append(
            {
                "tool": "home.refresh_state",
                "args": {},
                "risk_tier": "low",
                "idempotency_key": f"cycle:{getattr(episode, 'cycle_id', 0)}:refresh",
            }
        )

    return out[:12]
