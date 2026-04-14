"""
Run R independent conversation benchmarks of T turns each against /chat.

Measures per-run latency (mean / max / % under threshold) and a heuristic
*quality* score (user–reply relevance, anti-boilerplate, anti-echo, length).

**Response quality over iterations** (two senses):
  1. *Within-session:* first-third vs last-third mean score, and a least-squares
     slope of quality vs turn index (does the chat get better as turns advance?).
  2. *Across runs:* mean quality per repeated run on the same prompt list
     (memory warm-up, drift, or regression detection).

Typical full session (default T=100, R=5) takes tens of minutes — use
--turns 12 --runs 3 for a quick check.

Usage:
  cd HaromaX6
  python scripts/run_quality_iterations.py --turns 20 --runs 2
  python scripts/run_quality_iterations.py --turns 20 --runs 2 --depth fast

Output:
  logs/quality_runs/run_XX.jsonl(.summary.json) and aggregate.summary.json
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

HARNESS_VERSION = "2.1"


def _build_prompts(n: int) -> List[str]:
    """N varied user lines (same seed list as run_conversation_benchmark)."""
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


def _strip_turn_prefix(user_msg: str) -> str:
    return re.sub(r"^\[turn\s+\d+/\d+\]\s*", "", user_msg, flags=re.I).strip()


def score_reply(user_msg: str, assistant: str) -> float:
    """Backward-compatible single number."""
    return float(score_reply_detailed(user_msg, assistant)["overall"])


def score_reply_detailed(user_msg: str, assistant: str) -> Dict[str, Any]:
    """Heuristic quality 0–100 plus parts (for iteration / regression analysis)."""
    u = _strip_turn_prefix(user_msg).lower()
    a = (assistant or "").strip().lower()
    z: Dict[str, Any] = {
        "overall": 0.0,
        "components": {
            "overlap_ratio": 0.0,
            "assistant_word_count": 0,
            "user_question": False,
            "question_bonus": 0.0,
            "boilerplate_hits": 0,
            "boiler_penalty": 0.0,
            "echo_penalty": 0.0,
            "length_penalty": 0.0,
            "base_before_penalties": 0.0,
        },
    }
    if not a:
        return z
    if "[processing error]" in a or "still thinking" in a:
        z["overall"] = 5.0
        return z
    if len(a) < 20:
        z["overall"] = 18.0
        return z

    ut = set(re.findall(r"[a-z]{4,}", u))
    atoks = set(re.findall(r"[a-z]{4,}", a))
    overlap = len(ut & atoks) / max(len(ut), 1)
    u_words = set(re.findall(r"[a-z]{3,}", u))
    a_words = set(re.findall(r"[a-z]{3,}", a))
    echo_pen = 0.0
    if len(a_words) > 8 and u_words:
        j = len(u_words & a_words) / max(len(a_words), 1)
        if j > 0.52:
            echo_pen = min(22.0, (j - 0.52) * 80.0)

    boiler = (
        a.count("regarding '")
        + a.count("on conversation")
        + a.count("i stand present")
        + a.count("i move forward with purpose")
    )
    bpen = min(45.0, boiler * 10.0)
    base = 35.0 + 55.0 * min(1.0, overlap * 2.5)
    qbonus = 0.0
    user_q = "?" in u
    if user_q and len(atoks) > 8:
        qbonus = 8.0
        base += qbonus

    # Very long, low-content replies: mild penalty
    lpen = 0.0
    if len(a) > 3500:
        lpen = min(12.0, (len(a) - 3500) / 800.0)

    raw = base - bpen - echo_pen - lpen
    overall = max(0.0, min(100.0, raw))
    z["overall"] = round(overall, 2)
    z["components"] = {
        "overlap_ratio": round(overlap, 4),
        "assistant_word_count": len(a_words),
        "user_question": user_q,
        "question_bonus": round(qbonus, 2),
        "boilerplate_hits": boiler,
        "boiler_penalty": round(bpen, 2),
        "echo_penalty": round(echo_pen, 2),
        "length_penalty": round(lpen, 2),
        "base_before_penalties": round(base, 2),
    }
    return z


def _least_squares_slope(y: List[float]) -> float:
    """Slope of y vs index 0..n-1 (quality change per turn)."""
    n = len(y)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(y) / n
    num = sum((i - x_mean) * (y[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den > 1e-12 else 0.0


def _third_slice_means(scores: List[float]) -> Dict[str, float]:
    n = len(scores)
    if n < 3:
        return {
            "first_third_mean": scores[0] if scores else 0.0,
            "mid_third_mean": scores[n // 2] if scores else 0.0,
            "last_third_mean": scores[-1] if scores else 0.0,
        }
    k = max(1, n // 3)
    first = scores[:k]
    last = scores[-k:]
    mid_start = (n - k) // 2
    mid = scores[mid_start : mid_start + k]
    return {
        "first_third_mean": round(sum(first) / len(first), 3),
        "mid_third_mean": round(sum(mid) / len(mid), 3) if mid else 0.0,
        "last_third_mean": round(sum(last) / len(last), 3),
    }


def _run_one_session(
    client: Any,
    prompts: List[str],
    depth: str,
    latency_threshold: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    latencies: List[float] = []
    scores: List[float] = []
    rows: List[Dict[str, Any]] = []
    for i, message in enumerate(prompts):
        t0 = time.perf_counter()
        resp = client.post("/chat", json={"message": message, "depth": depth})
        dt = time.perf_counter() - t0
        latencies.append(dt)
        data = resp.get_json(silent=True) or {}
        text = str(data.get("response", "") or "")
        det = score_reply_detailed(message, text)
        q = float(det["overall"])
        scores.append(q)
        cum_mean = sum(scores) / len(scores)
        row = {
            "turn": i + 1,
            "user_message": message,
            "response_time_sec": round(dt, 4),
            "under_threshold": dt < latency_threshold,
            "quality_score": round(q, 2),
            "quality_components": det.get("components", {}),
            "rolling_quality_mean": round(cum_mean, 3),
            "assistant_response": text[:2000],
            "http_status": resp.status_code,
            "cycle_depth": data.get("cycle_depth", ""),
        }
        rows.append(row)
    s = sorted(latencies)

    def pct(p: float) -> float:
        return s[int(p * (len(s) - 1))] if s else 0.0

    under = sum(1 for x in latencies if x < latency_threshold)
    thirds = _third_slice_means(scores)
    slope = _least_squares_slope(scores)
    delta_lt = round(thirds["last_third_mean"] - thirds["first_third_mean"], 3)
    qbt = [round(x, 2) for x in scores]
    qbt_note = None
    if len(qbt) > 120:
        qbt_note = f"truncated in summary; full series in jsonl ({len(scores)} turns)"
        qbt = qbt[:40] + ["..."] + qbt[-40:]

    summary = {
        "turns": len(prompts),
        "quality_mean": round(sum(scores) / len(scores), 3),
        "quality_min": round(min(scores), 3),
        "quality_max": round(max(scores), 3),
        "quality_slope_per_turn": round(slope, 5),
        "quality_first_third_mean": thirds["first_third_mean"],
        "quality_mid_third_mean": thirds["mid_third_mean"],
        "quality_last_third_mean": thirds["last_third_mean"],
        "within_run_quality_delta_last_minus_first": delta_lt,
        "latency_mean_sec": round(sum(latencies) / len(latencies), 4),
        "latency_max_sec": round(max(latencies), 4),
        "latency_p50_sec": round(pct(0.50), 4),
        "latency_p95_sec": round(pct(0.95), 4),
        "fraction_under_threshold": round(under / len(latencies), 4),
        "latency_threshold_sec": latency_threshold,
        "all_under_threshold": under == len(latencies),
        "quality_by_turn": qbt,
        "quality_by_turn_note": qbt_note,
    }
    return summary, rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--turns", type=int, default=100)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument(
        "--depth",
        choices=("fast", "normal"),
        default="normal",
        help="normal = full cognitive path; fast = latency-oriented shortcuts",
    )
    p.add_argument(
        "--latency-target-sec",
        type=float,
        default=1.0,
        help="Count fraction of turns strictly under this latency",
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join(_REPO, "logs", "quality_runs"),
    )
    p.add_argument(
        "--quality-trend-eps",
        type=float,
        default=3.0,
        help="Allowed drop from previous run's mean quality (heuristic scorer noise)",
    )
    p.add_argument(
        "--warmup-posts",
        type=int,
        default=0,
        help="Optional /chat warmup messages before measured runs (can skew memory)",
    )
    args = p.parse_args()

    t_turns = max(1, min(args.turns, 500))
    n_runs = max(1, min(args.runs, 20))
    prompts = _build_prompts(t_turns)

    import mind.elarion_server_v2 as srv

    print("[quality-bench] Booting stack...", flush=True)
    t_boot = time.perf_counter()
    srv._init()
    print(f"[quality-bench] Boot done in {time.perf_counter() - t_boot:.1f}s", flush=True)
    try:
        atexit.unregister(srv._shutdown_save)
    except Exception as _e:
        print(f"[QualityRunner] atexit unregister error: {_e}", flush=True)

    client = srv.app.test_client()
    os.makedirs(args.output_dir, exist_ok=True)

    for w in range(max(0, min(args.warmup_posts, 10))):
        client.post(
            "/chat",
            json={"message": f"[warmup {w + 1}] Hi.", "depth": args.depth},
        )

    run_summaries: List[Dict[str, Any]] = []
    for r in range(1, n_runs + 1):
        print(f"\n[quality-bench] === Run {r}/{n_runs} ({t_turns} turns) ===", flush=True)
        t0 = time.perf_counter()
        summary, rows = _run_one_session(client, prompts, args.depth, args.latency_target_sec)
        summary["run"] = r
        summary["wall_clock_sec"] = round(time.perf_counter() - t0, 2)
        run_summaries.append(summary)

        path = os.path.join(args.output_dir, f"run_{r:02d}.jsonl")
        with open(path, "w", encoding="utf-8") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

        spath = path + ".summary.json"
        with open(spath, "w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2)

        print(
            f"  quality_mean={summary['quality_mean']}  "
            f"slope/turn={summary['quality_slope_per_turn']}  "
            f"dlt(last-first third)={summary['within_run_quality_delta_last_minus_first']}  "
            f"latency_mean={summary['latency_mean_sec']}s  "
            f"max={summary['latency_max_sec']}s  "
            f"p95={summary['latency_p95_sec']}s  "
            f"<{args.latency_target_sec}s: {summary['fraction_under_threshold'] * 100:.1f}%  "
            f"all_ok={summary['all_under_threshold']}",
            flush=True,
        )

    slopes = [x["quality_slope_per_turn"] for x in run_summaries]
    deltas = [x["within_run_quality_delta_last_minus_first"] for x in run_summaries]

    # Aggregate across runs
    agg = {
        "harness": "run_quality_iterations",
        "harness_version": HARNESS_VERSION,
        "turns_per_run": t_turns,
        "num_runs": n_runs,
        "depth": args.depth,
        "latency_target_sec": args.latency_target_sec,
        "quality_means_per_run": [x["quality_mean"] for x in run_summaries],
        "quality_slope_per_turn_per_run": slopes,
        "within_run_delta_last_minus_first_per_run": deltas,
        "mean_within_run_delta_last_minus_first": round(sum(deltas) / len(deltas), 4)
        if deltas
        else 0.0,
        "mean_quality_slope_per_turn": round(sum(slopes) / len(slopes), 6) if slopes else 0.0,
        "latency_mean_per_run": [x["latency_mean_sec"] for x in run_summaries],
        "latency_max_per_run": [x["latency_max_sec"] for x in run_summaries],
        "all_runs_all_under_threshold": all(x["all_under_threshold"] for x in run_summaries),
        "quality_trend_eps": args.quality_trend_eps,
        "quality_trend_non_decreasing": all(
            run_summaries[i]["quality_mean"] + args.quality_trend_eps
            >= run_summaries[i - 1]["quality_mean"]
            for i in range(1, len(run_summaries))
        ),
        "quality_trend_strict_non_decreasing": all(
            run_summaries[i]["quality_mean"] + 1e-9 >= run_summaries[i - 1]["quality_mean"]
            for i in range(1, len(run_summaries))
        ),
        "runs": run_summaries,
        "note_within_run": (
            "within_run_delta_last_minus_first: mean quality last third of "
            "turns minus first third (positive ⇒ improving through session). "
            "quality_slope_per_turn: LS slope vs turn index."
        ),
        "note_across_runs": (
            "quality_means_per_run: same prompts repeated; watch for drift "
            "or warm-up as memory fills."
        ),
    }
    agg_path = os.path.join(args.output_dir, "aggregate.summary.json")
    with open(agg_path, "w", encoding="utf-8") as af:
        json.dump(agg, af, indent=2)

    print("\n=== Aggregate ===", flush=True)
    print(f"  wrote: {agg_path}", flush=True)
    print(f"  quality_means: {agg['quality_means_per_run']}", flush=True)
    print(f"  latency_max each run: {agg['latency_max_per_run']}", flush=True)
    print(
        f"  all_runs_all_under_{args.latency_target_sec:g}s: {agg['all_runs_all_under_threshold']}",
        flush=True,
    )
    print(
        f"  quality_monotone_non_decreasing (eps={args.quality_trend_eps}): "
        f"{agg['quality_trend_non_decreasing']}",
        flush=True,
    )
    print(
        f"  quality_strict_monotone: {agg['quality_trend_strict_non_decreasing']}",
        flush=True,
    )
    print(
        f"  mean_within_run_delta (last-first third): "
        f"{agg['mean_within_run_delta_last_minus_first']}",
        flush=True,
    )
    print(
        f"  mean_quality_slope_per_turn: {agg['mean_quality_slope_per_turn']}",
        flush=True,
    )
    print(
        "\nNote: Heuristic quality is not human judgment. Sub-second latency on every\n"
        "turn may be infeasible on CPU with full models; use depth=fast and tune further.",
        flush=True,
    )


if __name__ == "__main__":
    main()
