#!/usr/bin/env python3
"""Measure async /chat latency until first complete result (local dev)."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = (os.environ.get("HAROMA_URL") or "http://127.0.0.1:8193").rstrip("/")


def _get(path: str, timeout: float = 5.0) -> tuple[int, dict]:
    req = urllib.request.Request(f"{BASE}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", errors="replace")
        return getattr(r, "status", 200), json.loads(raw) if raw.strip() else {}


def _post_chat(msg: str) -> tuple[int, dict]:
    body = json.dumps({"message": msg, "async": True}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120.0) as r:
            raw = r.read().decode("utf-8", errors="replace")
            return getattr(r, "status", 200), json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            return e.code, json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return e.code, {"_raw": raw}


def main() -> int:
    # Wait for server (cold boot can exceed 90s on large persistence)
    for _ in range(400):
        try:
            code, _ = _get("/status", timeout=2.0)
            if code == 200:
                break
        except OSError:
            time.sleep(0.5)
    else:
        print("ERROR: server not reachable", file=sys.stderr)
        return 1

    # Discard first request (cold caches)
    c0, j0 = _post_chat("ping")
    if c0 == 202 and j0.get("request_id"):
        rid0 = j0["request_id"]
        for _ in range(600):
            c, j = _get(f"/chat/result?id={rid0}", timeout=5.0)
            if j.get("status") == "pending" or (isinstance(j.get("response"), str) and j.get("response") is None):
                time.sleep(0.05)
                continue
            if "response" in j or j.get("answer"):
                break
            time.sleep(0.05)

    t0 = time.perf_counter()
    code, j = _post_chat("latency check hello")
    if code != 202 or not j.get("request_id"):
        print(f"ERROR: unexpected POST /chat {code} {j}", file=sys.stderr)
        return 1
    rid = j["request_id"]
    while True:
        _, j = _get(f"/chat/result?id={rid}", timeout=30.0)
        if j.get("status") == "pending":
            time.sleep(0.02)
            continue
        if "response" in j or j.get("answer") is not None:
            break
        time.sleep(0.02)
    elapsed = time.perf_counter() - t0
    print(f"async_chat_e2e_sec={elapsed:.3f}")
    if j.get("llm_context") and isinstance(j["llm_context"], dict):
        lm = j["llm_context"].get("latency_ms")
        if lm is not None:
            print(f"llm_context_latency_ms={lm}")
    return 0 if elapsed < 2.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
