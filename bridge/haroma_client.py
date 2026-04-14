"""
Minimal HTTP client for Haroma robot-bridge endpoints (stdlib only).

Use from a separate process on the robot or dev machine; set ``PYTHONPATH`` to the
repo root if you import this as a package.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple


def _optional_bearer_headers(extra: Dict[str, str]) -> Dict[str, str]:
    tok = str(os.environ.get("HAROMA_HTTP_BEARER_TOKEN", "") or "").strip()
    if tok:
        extra = dict(extra)
        extra.setdefault("Authorization", f"Bearer {tok}")
    return extra


def _post_json(url: str, body: Dict[str, Any], *, timeout: float = 15.0) -> Tuple[Any, int]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = _optional_bearer_headers({"Content-Type": "application/json; charset=utf-8"})
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers=headers,
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            code = getattr(resp, "status", 200) or 200
            try:
                return json.loads(raw) if raw.strip() else {}, int(code)
            except json.JSONDecodeError:
                return {"_parse_error": True, "raw": raw[:2000]}, int(code)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            return json.loads(raw) if raw.strip() else {"error": e.reason}, int(e.code)
        except json.JSONDecodeError:
            return {"error": e.reason, "raw": raw[:2000]}, int(e.code)


def _get_json(url: str, *, timeout: float = 15.0) -> Tuple[Any, int]:
    headers = _optional_bearer_headers({})
    req = urllib.request.Request(url, method="GET", headers=headers)
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            code = getattr(resp, "status", 200) or 200
            try:
                return json.loads(raw) if raw.strip() else {}, int(code)
            except json.JSONDecodeError:
                return {"_parse_error": True, "raw": raw[:2000]}, int(code)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            return json.loads(raw) if raw.strip() else {"error": e.reason}, int(e.code)
        except json.JSONDecodeError:
            return {"error": e.reason, "raw": raw[:2000]}, int(e.code)


def post_chat(
    base_url: str,
    message: str,
    *,
    depth: str = "normal",
    async_: bool = False,
    timeout: float = 600.0,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], int]:
    """POST ``{"message": ...}`` to ``POST /chat``.

    When *async_* is True and the server returns 202, use :func:`get_chat_result`
    or :func:`post_chat_wait_result` to retrieve the reply.
    """
    base = base_url.rstrip("/")
    body: Dict[str, Any] = {
        "message": message,
        "depth": depth,
        "async": async_,
    }
    if extra_fields:
        body.update(extra_fields)
    return _post_json(f"{base}/chat", body, timeout=timeout)


def get_chat_result(
    base_url: str,
    request_id: str,
    *,
    timeout: float = 30.0,
) -> Tuple[Dict[str, Any], int]:
    """GET ``/chat/result?id=...`` — poll async chat outcome."""
    base = base_url.rstrip("/")
    q = urllib.parse.quote(request_id, safe="")
    return _get_json(f"{base}/chat/result?id={q}", timeout=timeout)


def post_chat_wait_result(
    base_url: str,
    message: str,
    *,
    depth: str = "normal",
    poll_interval_sec: float = 0.25,
    max_wait_sec: float = 600.0,
    post_timeout: float = 60.0,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], int]:
    """POST /chat with ``async: true``, then poll until ``response`` is available or timeout."""
    data, code = post_chat(
        base_url,
        message,
        depth=depth,
        async_=True,
        timeout=post_timeout,
        extra_fields=extra_fields,
    )
    if code != 202:
        return data, code
    rid = str(data.get("request_id") or "").strip()
    if not rid:
        return {"error": "missing request_id in 202 response"}, code
    deadline = time.time() + max_wait_sec
    while time.time() < deadline:
        r, c = get_chat_result(base_url, rid, timeout=min(30.0, max_wait_sec))
        if c == 404:
            return r, c
        if c == 200 and r.get("status") != "pending" and "response" in r:
            return r, 200
        if c == 200 and r.get("status") == "pending":
            time.sleep(poll_interval_sec)
            continue
        if c == 200 and "response" in r:
            return r, 200
        time.sleep(poll_interval_sec)
    return {"error": "post_chat_wait_result timeout", "request_id": rid}, 504


def post_robot_bridge_feedback(
    base_url: str,
    feedback: Dict[str, Any],
    *,
    timeout: float = 15.0,
) -> Tuple[Dict[str, Any], int]:
    """POST *feedback* to ``{base_url}/robot/bridge/feedback`` (normalized bridge block)."""
    base = base_url.rstrip("/")
    return _post_json(f"{base}/robot/bridge/feedback", feedback, timeout=timeout)


def health_ping(base_url: str, *, timeout: float = 5.0) -> Tuple[Dict[str, Any], int]:
    """GET ``/status`` — quick check that Haroma is up."""
    base = base_url.rstrip("/")
    hdrs = _optional_bearer_headers({})
    req = urllib.request.Request(f"{base}/status", method="GET", headers=hdrs)
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        code = getattr(resp, "status", 200) or 200
        try:
            return json.loads(raw) if raw.strip() else {}, int(code)
        except json.JSONDecodeError:
            return {"_parse_error": True, "raw": raw[:2000]}, int(code)
