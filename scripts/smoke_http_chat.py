"""Time GET /status and POST /chat (depth=normal). Run while HaromaX6 is listening.

python scripts/smoke_http_chat.py
python scripts/smoke_http_chat.py --base http://127.0.0.1:8193
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8193")
    ap.add_argument("--timeout", type=float, default=120.0, help="Read timeout for /chat")
    ap.add_argument("--status-timeout", type=float, default=15.0, help="Read timeout for /status")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    t_status = (8.0, float(args.status_timeout))
    t_chat = (8.0, float(args.timeout))

    t0 = time.perf_counter()
    r = requests.get(f"{base}/status", timeout=t_status)
    dt = time.perf_counter() - t0
    print(f"GET /status -> {r.status_code} in {dt:.2f}s")
    if r.status_code != 200:
        return 1

    t0 = time.perf_counter()
    r = requests.post(
        f"{base}/chat",
        json={"message": "ping", "depth": "normal"},
        timeout=t_chat,
    )
    dt = time.perf_counter() - t0
    body = r.json() if r.text else {}
    reply = body.get("response") or ""
    print(f"POST /chat (fast) -> {r.status_code} in {dt:.2f}s")
    print(f"  reply[:200]: {reply[:200]!r}")
    if dt > 60:
        print("  WARNING: over 60s — check neural_sync / locks if this repeats.", file=sys.stderr)
    return 0 if r.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
