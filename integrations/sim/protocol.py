"""Pluggable simulation contract — any host (Unity, Isaac, Gymnasium, custom) can implement this."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class SimulationBackend(Protocol):
    """Minimal interface so Haroma can interact with **any** simulator behind one API.

    Implementations may wrap a visual engine, physics sim, text world, or HTTP proxy.
    Use :func:`integrations.sim.registry.create` or :func:`load_backend_from_env`.
    """

    def backend_id(self) -> str:
        """Stable tag for logging (e.g. ``http_json``, ``null``, ``my_plugin``)."""

    def capabilities(self) -> Dict[str, Any]:
        """Optional modalities, action schema hints, version (for prompts / UI)."""

    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
        """Start or clear an episode; return host-specific observation bundle."""

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply *action*; return observation / reward / done in host format."""

    def observe(self, **kwargs: Any) -> Dict[str, Any]:
        """Passive sense without a new action (optional; may mirror last frame)."""

    def close(self) -> None:
        """Release handles / disconnect."""


__all__ = ["SimulationBackend"]
