"""Register and construct :class:`integrations.sim.protocol.SimulationBackend` implementations."""

from __future__ import annotations

import importlib
import json
import os
from typing import Any, Callable, Dict, Optional, Type

from integrations.sim.backends.http_json_backend import HttpJsonSimulationBackend
from integrations.sim.backends.null_backend import NullSimulationBackend
from integrations.sim.protocol import SimulationBackend

BackendFactory = Callable[..., SimulationBackend]

_REGISTRY: Dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory) -> None:
    """Register a factory under *name* (lowercased)."""
    _REGISTRY[name.strip().lower()] = factory


def _builtin_null(**_: Any) -> SimulationBackend:
    return NullSimulationBackend()


def _builtin_http_json(**_: Any) -> SimulationBackend:
    return HttpJsonSimulationBackend.from_env()


register_backend("null", _builtin_null)
register_backend("none", _builtin_null)
register_backend("http_json", _builtin_http_json)
register_backend("http", _builtin_http_json)


def create_backend(name: str, **kwargs: Any) -> SimulationBackend:
    """Instantiate a registered backend by name."""
    key = (name or "null").strip().lower()
    factory = _REGISTRY.get(key)
    if factory is None:
        raise KeyError(f"unknown simulation backend {name!r}; registered: {sorted(_REGISTRY.keys())}")
    return factory(**kwargs)


def _load_importable_backend(spec: str) -> SimulationBackend:
    """``some.module:Class`` — optional ``HAROMA_SIM_BACKEND_KWARGS`` JSON for ``__init__``."""
    if ":" not in spec:
        raise ValueError("importable backend must look like package.module:ClassName")
    mod_name, _, cls_name = spec.partition(":")
    mod_name = mod_name.strip()
    cls_name = cls_name.strip()
    if not mod_name or not cls_name:
        raise ValueError(f"invalid backend spec {spec!r}")
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    if not isinstance(cls, type):
        raise TypeError(f"{spec!r} must refer to a class")
    raw = str(os.environ.get("HAROMA_SIM_BACKEND_KWARGS", "") or "").strip()
    kwargs: Dict[str, Any] = {}
    if raw:
        kwargs = json.loads(raw)
    if not isinstance(kwargs, dict):
        raise ValueError("HAROMA_SIM_BACKEND_KWARGS must be a JSON object")
    return cls(**kwargs)


def load_backend_from_env() -> SimulationBackend:
    """Read ``HAROMA_SIM_BACKEND`` — built-in name, or ``module:Class`` for any sim SDK."""
    spec = str(os.environ.get("HAROMA_SIM_BACKEND", "null") or "null").strip()
    if not spec or spec.lower() in ("null", "none"):
        return NullSimulationBackend()
    if ":" in spec:
        return _load_importable_backend(spec)
    return create_backend(spec)


__all__ = [
    "create_backend",
    "load_backend_from_env",
    "register_backend",
]
