"""
Run an N-turn HTTP /chat conversation against the booted multi-agent stack.

Uses Flask test_client (no separate server process). Writes JSONL results and
prints a summary: per-turn latency, whether the assistant reply looks like a question.

Usage:
  cd HaromaX6
  python scripts/run_conversation_benchmark.py --count 100 --output logs/conv_100.jsonl
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import sys
import time
from typing import Any, Dict, List

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Lazy import after path set


def _response_looks_like_question(text: str) -> bool:
    if not text or not text.strip():
        return False
    t = text.strip()
    if "?" in t:
        return True
    # Common question openers (English)
    return bool(
        re.search(
            r"(?i)\b("
            r"what|why|how|who|when|where|which|whose|whom|"
            r"can you|could you|would you|will you|do you|did you|"
            r"is it|are you|have you|should we|shall we|may i"
            r")\b",
            t[:280],
        )
    )


def _build_prompts(n: int) -> List[str]:
    """N varied user lines (mix statements and questions)."""
    seeds = [
        "Hello.",
        "What are you thinking about right now?",
        "Tell me something interesting about memory.",
        "How do you handle conflicting beliefs?",
        "I feel uncertain about today.",
        "What is curiosity to you?",
        "Describe your sense of identity in one paragraph.",
        "Do you ever dream?",
        "What would you like to know about me?",
        "Explain reconciliation between different viewpoints.",
        "I'm tired but want to keep talking.",
        "What is the role of emotion in decisions?",
        "Narrate a short inner monologue.",
        "Should AI systems have persistent goals?",
        "What does 'understanding' mean to you?",
        "I'm learning Python; any metaphor for cognition?",
        "How do you know if you are wrong?",
        "What is attention?",
        "Tell me a compact summary of your architecture.",
        "If you could ask one question of humanity, what would it be?",
    ]
    out: List[str] = []
    for i in range(n):
        base = seeds[i % len(seeds)]
        out.append(f"[turn {i + 1}/{n}] {base}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=100)
    p.add_argument(
        "--output",
        default=os.path.join(_REPO, "logs", "conversation_benchmark.jsonl"),
    )
    p.add_argument("--quiet", action="store_true", help="Less stdout; still writes JSONL")
    p.add_argument(
        "--depth",
        choices=("normal",),
        default="normal",
        help="POST /chat depth",
    )
    args = p.parse_args()

    n = max(1, min(args.count, 500))
    prompts = _build_prompts(n)

    import mind.elarion_server_v2 as srv

    print(f"[conv-bench] Booting stack (this can take a minute)...", flush=True)
    t_boot = time.perf_counter()
    srv._init()
    print(f"[conv-bench] Boot done in {time.perf_counter() - t_boot:.1f}s", flush=True)

    try:
        atexit.unregister(srv._shutdown_save)
    except Exception as _e:
        print(f"[ConvBenchmark] atexit unregister error: {_e}", flush=True)

    client = srv.app.test_client()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    latencies: List[float] = []
    assistant_asked_count = 0
    rows: List[Dict[str, Any]] = []

    with open(args.output, "w", encoding="utf-8") as fout:
        for i, message in enumerate(prompts):
            t0 = time.perf_counter()
            resp = client.post(
                "/chat",
                json={"message": message, "depth": args.depth},
            )
            dt = time.perf_counter() - t0
            latencies.append(dt)
            try:
                data = resp.get_json(silent=True) or {}
            except Exception:
                data = {}
            text = str(data.get("response", "") or "")
            asked = _response_looks_like_question(text)
            if asked:
                assistant_asked_count += 1

            row = {
                "turn": i + 1,
                "user_message": message,
                "request_depth": args.depth,
                "response_cycle_depth": data.get("cycle_depth", ""),
                "http_status": resp.status_code,
                "response_time_sec": round(dt, 3),
                "assistant_response": text,
                "assistant_response_preview": text[:400],
                "assistant_full_length": len(text),
                "assistant_reply_contains_question": asked,
                "strategy": data.get("strategy", ""),
                "cycle": data.get("cycle"),
            }
            rows.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            if not args.quiet:
                qmark = " [ASSISTANT ASKED? yes]" if asked else ""
                print(
                    f"  turn {i + 1}/{n}  {dt:6.2f}s  "
                    f"preview: {text[:72].replace(chr(10), ' ')!r}{qmark}",
                    flush=True,
                )

    s = sorted(latencies)

    def pct(p: float) -> float:
        return s[int(p * (len(s) - 1))] if s else 0.0

    summary = {
        "turns": n,
        "latency_mean_sec": round(sum(latencies) / len(latencies), 3),
        "latency_min_sec": round(min(latencies), 3),
        "latency_max_sec": round(max(latencies), 3),
        "latency_p50_sec": round(pct(0.50), 3),
        "latency_p95_sec": round(pct(0.95), 3),
        "assistant_turns_with_question": assistant_asked_count,
        "assistant_question_rate": round(assistant_asked_count / n, 3),
        "output_file": os.path.abspath(args.output),
    }
    with open(args.output + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("", flush=True)
    print("=== Conversation benchmark summary ===", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)
    print("", flush=True)
    print(
        "Note: 'assistant_reply_contains_question' uses heuristics (? or question words). "
        "See JSONL for each turn.",
        flush=True,
    )


if __name__ == "__main__":
    main()
