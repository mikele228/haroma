"""One synchronous /chat timing probe (for pipeline tuning).

Run the server in another terminal with dummy + full pipeline logging, then:

  set HAROMA_LLM_DUMMY_REPLY=1
  set HAROMA_CHAT_PIPELINE_LOG=full
  set HAROMA_LIVE_ROBOT_HEAD=0
  set HAROMA_BENCH_DISABLE_BG_TRAINING=1
  python -m mind.elarion_server_v2

Wait for boot (~40s), then:

  python scripts/bench_chat_pipeline_once.py

Uses POST /chat with ``\"async\": false`` so the HTTP call blocks until the slot
fills. Check server stderr for ``[ChatPipeline]`` lines with ``seg=`` / ``cum=`` to
see where time went (largest ``seg`` first).
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
    ap.add_argument("--message", default="bench hello")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()
    base = args.base.rstrip("/")

    t0 = time.perf_counter()
    r = requests.post(
        f"{base}/chat",
        json={"message": args.message, "async": False, "depth": "normal"},
        timeout=args.timeout,
    )
    dt = time.perf_counter() - t0
    print(f"POST /chat sync -> HTTP {r.status_code} in {dt:.2f}s")
    try:
        body = r.json()
    except Exception:
        print(r.text[:500])
        return 1 if r.status_code != 200 else 0
    err = body.get("error")
    if err:
        print(f"  error: {err}")
    rep = body.get("response") or ""
    print(f"  response[:200]: {rep[:200]!r}")
    return 0 if r.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
