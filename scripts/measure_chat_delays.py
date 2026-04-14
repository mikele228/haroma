#!/usr/bin/env python3
"""In-process chat delay measurement (no HTTP). Boots agents, times fast vs normal.

  set HAROMA_BENCH_DISABLE_BG_TRAINING=1
  python -u scripts/measure_chat_delays.py

Prints wall-clock per message and latency_trace summaries when present.
"""

from __future__ import annotations

import os
import sys
import threading
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

os.environ.setdefault("HAROMA_BENCH_DISABLE_BG_TRAINING", "1")

from agents.boot_agent import BootAgent  # noqa: E402

MESSAGES = [
    "Say hi in one word.",
    "What is 7 plus 5?",
    "Thanks.",
]


def _top_spans(lt: dict, n: int = 6) -> str:
    spans = lt.get("spans") or []
    by_ms = sorted(spans, key=lambda x: -float(x.get("ms", 0.0)))[:n]
    return ", ".join(f"{s.get('phase')}:{s.get('ms')}ms" for s in by_ms)


def main() -> int:
    print("[measure_chat_delays] booting (BG training disabled)...", flush=True)
    boot = BootAgent()
    t_boot = time.perf_counter()
    shared = boot.boot()
    print(f"  boot_wall_s={time.perf_counter() - t_boot:.1f}", flush=True)
    if shared is None:
        print("boot failed", flush=True)
        return 1

    boot.input_agent.set_boot_agent(boot)
    boot.trueself_agent.set_boot_agent(boot)
    for p in boot.persona_agents:
        p.set_boot_agent(boot)

    boot.start_all()
    time.sleep(1.5)

    results = []
    for depth in ("fast", "normal"):
        print(f"\n--- depth={depth} ---", flush=True)
        for msg in MESSAGES:
            t0 = time.perf_counter()
            slot = boot.input_agent.push_text(
                msg,
                source="bench",
                depth=depth,
                trace_latency=True,
            )
            # fast completes inside push_text; wait() then returns immediately.
            ok = slot["event"].wait(timeout=240.0)
            dt = time.perf_counter() - t0
            res = slot.get("result") or {}
            lt = res.get("latency_trace") or {}
            line = f"  {dt:.2f}s wall | trace_total={lt.get('total_ms')}ms | {_top_spans(lt)}"
            print(line, flush=True)
            if not ok:
                print("  TIMEOUT", flush=True)
            results.append((depth, msg, dt, lt))

    print("\n=== summary ===", flush=True)
    for depth in ("fast", "normal"):
        xs = [r[2] for r in results if r[0] == depth]
        if xs:
            mean = sum(xs) / len(xs)
            print(f"  {depth}: n={len(xs)} mean={mean:.2f}s max={max(xs):.2f}s", flush=True)

    print("\n[measure_chat_delays] shutdown...", flush=True)
    boot.save_and_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
