"""Helpers to attach simulation observations to ``agent_environment`` / extensions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


def merge_simulation_into_extensions(
    extensions: Optional[Dict[str, Any]],
    *,
    backend_id: str,
    bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Return *extensions* with ``simulation`` block merged (rolling ``last_bundle``)."""
    ext = deepcopy(extensions) if extensions else {}
    sim = dict(ext.get("simulation") if isinstance(ext.get("simulation"), dict) else {})
    sim["backend_id"] = backend_id
    sim["last_bundle"] = dict(bundle) if isinstance(bundle, dict) else {"raw": bundle}
    ext["simulation"] = sim
    return ext


def simulation_summary_for_prompt(bundle: Dict[str, Any], max_chars: int = 1200) -> str:
    """Compact string for LLM context from a host observation dict."""
    if not isinstance(bundle, dict):
        return ""
    parts = []
    for key in ("description", "summary", "message", "observation"):
        v = bundle.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(f"{key}: {v.strip()[: max_chars // 2]}")
            break
    obs = bundle.get("observation")
    if isinstance(obs, dict) and obs:
        parts.append(f"obs_keys: {','.join(list(obs.keys())[:24])}")
    t = " | ".join(parts)
    return t[:max_chars]


__all__ = [
    "merge_simulation_into_extensions",
    "simulation_summary_for_prompt",
]
