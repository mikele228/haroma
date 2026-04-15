"""
Central HTTP server state for the Flask app.

Avoids scattered module-level globals for :class:`agents.boot_agent.BootAgent`,
sensor polling, and the async chat registry. :func:`mind.elarion_server_v2._init`
initializes this **once per process** (thread-safe; repeat calls are ignored).
Attached as::

    app.extensions[HAROMA_FLASK_EXTENSION_KEY]  # HaromaServerState

Routes and helpers read state via :func:`mind.elarion_server_v2._haroma` (defined
next to the Flask ``app`` instance) or :func:`get_haroma_server_state`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Single source of truth for Flask ``app.extensions`` key (tests use :func:`get_haroma_server_state`).
HAROMA_FLASK_EXTENSION_KEY = "haroma"

__all__ = [
    "HAROMA_FLASK_EXTENSION_KEY",
    "HaromaServerState",
    "get_haroma_server_state",
]


@dataclass
class HaromaServerState:
    """Single holder for runtime objects tied to one Werkzeug worker process."""

    boot_agent: Any = None
    sensor_poller: Any = None
    chat_async_registry: Any = None


def get_haroma_server_state(app: Any) -> HaromaServerState:
    """Return Haroma runtime state on ``app``, creating an empty state if missing."""
    ext = getattr(app, "extensions", None)
    if ext is None:
        raise RuntimeError("Flask app has no extensions dict")
    st = ext.get(HAROMA_FLASK_EXTENSION_KEY)
    if st is None:
        st = HaromaServerState()
        ext[HAROMA_FLASK_EXTENSION_KEY] = st
    return st
