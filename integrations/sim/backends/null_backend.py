"""No-op simulation backend — default when no external sim is configured."""

from __future__ import annotations

from typing import Any, Dict, Optional


class NullSimulationBackend:
    """Returns structured stubs; useful for tests and headless runs."""

    def __init__(self) -> None:
        self._step_count = 0

    def backend_id(self) -> str:
        return "null"

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": [],
            "action_schema": "dialogue_and_text_env_only",
            "notes": "NullSimulationBackend performs no external IO.",
        }

    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
        self._step_count = 0
        return {"ok": True, "phase": "reset", "seed": seed, "observation": {}}

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._step_count += 1
        return {
            "ok": True,
            "phase": "step",
            "step_index": self._step_count,
            "action_echo": dict(action),
            "observation": {},
            "reward": None,
            "done": False,
        }

    def observe(self, **kwargs: Any) -> Dict[str, Any]:
        return {"ok": True, "phase": "observe", "observation": {}}

    def close(self) -> None:
        return None


__all__ = ["NullSimulationBackend"]
