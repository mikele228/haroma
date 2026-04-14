"""Benchmark POST /chat latency: depth=fast vs depth=normal (same messages).

  python -u scripts/benchmark_chat_depth.py
  python -u scripts/benchmark_chat_depth.py --base http://127.0.0.1:8193 --rounds 10

Run while elarion_server_v2 is listening.

For apples-to-apples latency (avoid multi-minute stalls when background
``neural_sync`` training holds the lock), restart the server with::

  set HAROMA_BENCH_DISABLE_BG_TRAINING=1

before ``python main.py`` (Windows) or export the same on Unix.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import List, Tuple

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    raise SystemExit(2)

DEFAULT_MESSAGES = [
    "What is 7 plus 5?",
    "Name any planet in one word.",
    "Say hi in one short sentence.",
    "What color is the sky on a clear day?",
    "Count from 1 to 3.",
    "Is water wet? One sentence.",
    "What does HTML stand for?",
    "Give me a synonym for happy.",
    "What is the capital of France?",
    "Thanks — reply with one word.",
]


def _percentile(sorted_times: List[float], p: float) -> float:
    if not sorted_times:
        return 0.0
    k = (len(sorted_times) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_times) - 1)
    if f == c:
        return sorted_times[f]
    return sorted_times[f] + (sorted_times[c] - sorted_times[f]) * (k - f)


def run_block(
    base: str,
    depth: str,
    messages: List[str],
    timeout: Tuple[float, float],
) -> List[float]:
    times: List[float] = []
    for msg in messages:
        t0 = time.perf_counter()
        try:
            r = requests.post(
                f"{base}/chat",
                json={"message": msg, "depth": depth},
                timeout=timeout,
            )
        except requests.RequestException as e:
            dt = time.perf_counter() - t0
            print(
                f"  ERROR depth={depth} after {dt:.2f}s: {e}",
                file=sys.stderr,
                flush=True,
            )
            continue
        dt = time.perf_counter() - t0
        times.append(dt)
        if r.status_code != 200:
            print(
                f"  ERROR depth={depth} status={r.status_code} after {dt:.2f}s",
                file=sys.stderr,
                flush=True,
            )
        else:
            preview = ""
            try:
                body = r.json()
                preview = (body.get("response") or "")[:60]
            except Exception:
                pass
            print(
                f"  depth={depth} ok {dt:.2f}s  resp[:60]={preview!r}",
                flush=True,
            )
    return times


def summarize(label: str, times: List[float]) -> None:
    if not times:
        print(f"{label}: no samples", flush=True)
        return
    s = sorted(times)
    parts = [
        f"{label}: n={len(times)}",
        f"mean={statistics.mean(times):.2f}s",
    ]
    if len(times) > 1:
        parts.append(f"std={statistics.stdev(times):.2f}s")
        parts.append(f"median={statistics.median(times):.2f}s")
        parts.append(f"min={min(times):.2f}s")
        parts.append(f"max={max(times):.2f}s")
        parts.append(f"p90={_percentile(s, 90):.2f}s")
    print("  ".join(parts), flush=True)
    for i, dt in enumerate(times, 1):
        print(f"    #{i:02d}  {dt:.2f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8193")
    ap.add_argument("--rounds", type=int, default=10, help="messages per depth")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--status-timeout", type=float, default=15.0)
    ap.add_argument(
        "--boot-wait",
        type=float,
        default=20.0,
        help="Seconds to wait after /status=200 before first /chat "
        "(cold boot may need 60–120; use 0 if the server is already idle).",
    )
    args = ap.parse_args()
    base = args.base.rstrip("/")
    n = max(1, min(args.rounds, len(DEFAULT_MESSAGES)))
    messages = DEFAULT_MESSAGES[:n]
    t_out = (10.0, float(args.timeout))
    t_status = (8.0, float(args.status_timeout))

    try:
        r = requests.get(f"{base}/status", timeout=t_status)
    except requests.RequestException as e:
        print(f"GET /status failed: {e}", file=sys.stderr)
        return 2
    if r.status_code != 200:
        print(f"GET /status -> {r.status_code}", file=sys.stderr)
        return 2

    bw = max(0.0, float(args.boot_wait))
    if bw > 0:
        print(
            f"Boot settle: sleeping {bw:.0f}s before chat (override --boot-wait)",
            flush=True,
        )
        time.sleep(bw)

    print(f"Base URL: {base}", flush=True)
    print(
        f"Messages per mode: {n} (same list for fast and normal)\n",
        flush=True,
    )

    print("--- FAST ---", flush=True)
    t_fast = run_block(base, "fast", messages, t_out)
    print(flush=True)
    print("--- NORMAL ---", flush=True)
    t_norm = run_block(base, "normal", messages, t_out)
    print(flush=True)

    summarize("FAST", t_fast)
    print()
    summarize("NORMAL", t_norm)
    print()
    if t_fast and t_norm:
        ratio = statistics.mean(t_norm) / max(statistics.mean(t_fast), 1e-6)
        print(f"Ratio mean(normal/fast): {ratio:.2f}x", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
