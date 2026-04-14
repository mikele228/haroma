"""Standard envelope for environment observations (world loop + tracing)."""

from __future__ import annotations

from typing import Any, Dict, Optional


def normalize_environment_observation(
    raw: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a stable schema: reward, done, info, world.

    ``world`` holds the remaining keys so existing TextEnvironment dicts stay usable.
    """
    if not raw:
        return {
            "reward": 0.0,
            "done": False,
            "info": {},
            "world": {},
        }
    r = dict(raw)
    try:
        reward = float(r.pop("reward", 0.0))
    except (TypeError, ValueError):
        reward = 0.0
    done = bool(r.pop("done", False))
    info = r.pop("info", {})
    if not isinstance(info, dict):
        info = {}
    return {
        "reward": reward,
        "done": done,
        "info": dict(info),
        "world": r,
    }


def merge_world_into_observation(norm: Dict[str, Any], legacy: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer explicit keys on legacy observation; fill gaps from norm['world']."""
    world = norm.get("world") if isinstance(norm.get("world"), dict) else {}
    merged = dict(world)
    merged.update(legacy)
    return merged
