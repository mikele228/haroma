"""Generic HTTP JSON simulation — works with any REST-shaped sim that follows the contract."""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple


def _post_json(url: str, body: Dict[str, Any], *, timeout: float = 30.0) -> Tuple[Any, int]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            code = getattr(resp, "status", 200) or 200
            try:
                return (json.loads(raw) if raw.strip() else {}), int(code)
            except json.JSONDecodeError:
                return {"_parse_error": True, "raw": raw[:2000]}, int(code)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            return (json.loads(raw) if raw.strip() else {"error": e.reason}), int(e.code)
        except json.JSONDecodeError:
            return {"error": e.reason, "raw": raw[:2000]}, int(e.code)


def _get_json(url: str, *, timeout: float = 30.0) -> Tuple[Any, int]:
    req = urllib.request.Request(url, method="GET")
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            code = getattr(resp, "status", 200) or 200
            try:
                return (json.loads(raw) if raw.strip() else {}), int(code)
            except json.JSONDecodeError:
                return {"_parse_error": True, "raw": raw[:2000]}, int(code)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            return (json.loads(raw) if raw.strip() else {"error": e.reason}), int(e.code)
        except json.JSONDecodeError:
            return {"error": e.reason, "raw": raw[:2000]}, int(e.code)


class HttpJsonSimulationBackend:
    """POST/GET JSON to configurable paths — adapt any HTTP sim without code in Haroma.

    Expected server behavior (you can map your sim to this):

    - ``POST {base}{step_path}`` body ``{"action": {...}, "seed": null, ...}`` → observation dict.
    - ``POST {base}{reset_path}`` body ``{"seed": 42}`` (optional) → observation dict.
    - ``GET {base}{observe_path}`` (optional) → passive observation.

    Empty *reset_path* or *observe_path* skips that call and returns a minimal dict.
    """

    def __init__(
        self,
        base_url: str,
        *,
        step_path: str = "/sim/step",
        reset_path: str = "/sim/reset",
        observe_path: str = "/sim/observe",
        timeout_sec: float = 30.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._step_path = step_path if step_path.startswith("/") else f"/{step_path}"
        self._reset_path = reset_path.strip() if reset_path else ""
        self._observe_path = observe_path.strip() if observe_path else ""
        self._timeout = float(timeout_sec)

    @classmethod
    def from_env(cls) -> "HttpJsonSimulationBackend":
        base = str(os.environ.get("HAROMA_SIM_HTTP_BASE_URL", "") or "").strip()
        if not base:
            raise ValueError("HAROMA_SIM_HTTP_BASE_URL is required for http_json backend")
        return cls(
            base,
            step_path=os.environ.get("HAROMA_SIM_HTTP_STEP_PATH", "/sim/step"),
            reset_path=os.environ.get("HAROMA_SIM_HTTP_RESET_PATH", "/sim/reset"),
            observe_path=os.environ.get("HAROMA_SIM_HTTP_OBSERVE_PATH", "/sim/observe"),
            timeout_sec=float(os.environ.get("HAROMA_SIM_HTTP_TIMEOUT_SEC", "30") or 30),
        )

    def _url(self, path: str) -> str:
        p = path if path.startswith("/") else f"/{path}"
        return f"{self._base}{p}"

    def backend_id(self) -> str:
        return "http_json"

    def capabilities(self) -> Dict[str, Any]:
        return {
            "transport": "http+json",
            "base_url": self._base,
            "step_path": self._step_path,
            "reset_path": self._reset_path or None,
            "observe_path": self._observe_path or None,
        }

    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
        if not self._reset_path:
            return {"ok": True, "phase": "reset", "skipped": True, "reason": "no_reset_path"}
        body: Dict[str, Any] = dict(kwargs)
        if seed is not None:
            body["seed"] = seed
        data, code = _post_json(self._url(self._reset_path), body, timeout=self._timeout)
        if isinstance(data, dict):
            data.setdefault("http_status", code)
        return data if isinstance(data, dict) else {"raw": data, "http_status": code}

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        body = {"action": dict(action)}
        data, code = _post_json(self._url(self._step_path), body, timeout=self._timeout)
        if isinstance(data, dict):
            data.setdefault("http_status", code)
        return data if isinstance(data, dict) else {"raw": data, "http_status": code}

    def observe(self, **kwargs: Any) -> Dict[str, Any]:
        if not self._observe_path:
            return {"ok": True, "phase": "observe", "skipped": True}
        pairs = {str(k): str(v) for k, v in kwargs.items() if v is not None}
        q = urllib.parse.urlencode(pairs) if pairs else ""
        url = self._url(self._observe_path)
        if q:
            url = f"{url}?{q}"
        data, code = _get_json(url, timeout=self._timeout)
        if isinstance(data, dict):
            data.setdefault("http_status", code)
        return data if isinstance(data, dict) else {"raw": data, "http_status": code}

    def close(self) -> None:
        return None
