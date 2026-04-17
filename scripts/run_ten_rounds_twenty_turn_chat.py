"""
Run the same 20-turn chat script 10 times in one process (single boot) and
report whether simple repetition / template metrics improve from round 1 → 10.

Usage (from repo root):
  python scripts/run_ten_rounds_twenty_turn_chat.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from agents.boot_agent import BootAgent  # noqa: E402

_RTC_PATH = os.path.join(_ROOT, "scripts", "run_twenty_turn_chat.py")
_spec = importlib.util.spec_from_file_location("run_twenty_turn_chat", _RTC_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load {_RTC_PATH}")
_rtc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rtc)
USER_TURNS = _rtc.USER_TURNS
WAIT_PER_TURN = _rtc.WAIT_PER_TURN

NUM_ROUNDS = 10
LOG_PATH = os.path.join(_ROOT, "logs", "ten_rounds_twenty_turn_eval.json")

# Heuristic “low quality” templates seen in organic composer runs.
_RE_EXPRESSION = re.compile(
    r"expression is at the heart|ties to expression|need for expression",
    re.I,
)
_RE_ANGER = re.compile(
    r"sense of anger|stirs anger|feeling quite anger|anger rising",
    re.I,
)
_RE_CLEARING = re.compile(r"the clearing", re.I)


def _tokens(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9']+", s.lower()) if len(w) > 2}


def _mean_jaccard_distance(replies: List[str]) -> float:
    """Average 1 - Jaccard over unordered pairs (higher = more diverse)."""
    vecs = [_tokens(r) for r in replies]
    n = len(vecs)
    if n < 2:
        return 0.0
    dists: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = vecs[i], vecs[j]
            if not a and not b:
                continue
            inter = len(a & b)
            union = len(a | b) or 1
            dists.append(1.0 - inter / union)
    return sum(dists) / len(dists) if dists else 0.0


def _round_metrics(replies: List[str]) -> Dict[str, Any]:
    uniq = len({r.strip().lower() for r in replies})
    lens = [len(r) for r in replies]
    return {
        "unique_replies": uniq,
        "unique_ratio": round(uniq / max(len(replies), 1), 3),
        "mean_chars": round(sum(lens) / max(len(lens), 1), 1),
        "template_expression_hits": sum(1 for r in replies if _RE_EXPRESSION.search(r)),
        "template_anger_hits": sum(1 for r in replies if _RE_ANGER.search(r)),
        "clearing_hits": sum(1 for r in replies if _RE_CLEARING.search(r)),
        "pairwise_jacc_dist": round(_mean_jaccard_distance(replies), 3),
    }


def main() -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    boot: BootAgent | None = None
    all_rounds: List[Dict[str, Any]] = []
    t0 = time.time()

    try:
        print("Booting (once)...", flush=True)
        boot = BootAgent()
        shared = boot.boot()
        if shared is None:
            raise RuntimeError("boot() returned None")
        boot.input_agent.set_boot_agent(boot)
        boot.trueself_agent.set_boot_agent(boot)
        for p in boot.persona_agents:
            p.set_boot_agent(boot)
        boot.start_all()
        time.sleep(1.0)

        for round_i in range(1, NUM_ROUNDS + 1):
            print(f"\n{'#' * 60}", flush=True)
            print(f" ROUND {round_i}/{NUM_ROUNDS} (20 turns)", flush=True)
            print(f"{'#' * 60}\n", flush=True)
            replies: List[str] = []
            turn_logs: List[Dict[str, str]] = []

            for ti, user_text in enumerate(USER_TURNS, start=1):
                slot = boot.input_agent.push_text(user_text, source="user", depth="normal")
                if not slot["event"].wait(timeout=WAIT_PER_TURN):
                    reply = f"[timeout {WAIT_PER_TURN:.0f}s]"
                else:
                    res = slot.get("result") or {}
                    reply = str(res.get("response", "")).strip()
                replies.append(reply)
                turn_logs.append({"turn": ti, "user": user_text, "reply": reply})
                if ti % 5 == 0 or ti == 1:
                    print(f"  r{round_i} t{ti}/20 ok | {reply[:72]}...", flush=True)

            m = _round_metrics(replies)
            m["round"] = round_i
            m["turns"] = turn_logs
            all_rounds.append(m)
            print(
                f"\n  Round {round_i} metrics: {json.dumps({k: v for k, v in m.items() if k != 'turns'}, indent=2)}",
                flush=True,
            )

        # Trend: compare first vs last + simple slopes
        def pick(key: str) -> List[float]:
            return [float(r[key]) for r in all_rounds]

        summary = {
            "rounds": NUM_ROUNDS,
            "turns_per_round": len(USER_TURNS),
            "elapsed_sec": round(time.time() - t0, 1),
            "run_metrics": [{k: v for k, v in r.items() if k != "turns"} for r in all_rounds],
        }
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "full": all_rounds}, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {LOG_PATH}", flush=True)

        r1, r10 = all_rounds[0], all_rounds[-1]
        print("\n" + "=" * 60)
        print(" COMPARISON: Round 1  vs  Round 10 (same 20 prompts)")
        print("=" * 60)
        for key in (
            "unique_ratio",
            "pairwise_jacc_dist",
            "mean_chars",
            "template_expression_hits",
            "template_anger_hits",
            "clearing_hits",
        ):
            print(f"  {key:26}  r1={r1[key]!s}  r10={r10[key]!s}")

        # Linear trend (least squares slope) for a few scalars
        xs = list(range(1, NUM_ROUNDS + 1))

        def slope(ys: List[float]) -> float:
            n = len(xs)
            mx = sum(xs) / n
            my = sum(ys) / n
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = sum((x - mx) ** 2 for x in xs) or 1e-9
            return num / den

        ur_slope = slope(pick("unique_ratio"))
        jd_slope = slope(pick("pairwise_jacc_dist"))
        ex_slope = slope([float(r["template_expression_hits"]) for r in all_rounds])
        cl_slope = slope([float(r["clearing_hits"]) for r in all_rounds])

        print("\n  Per-round linear trend (slope vs round index 1–10):")
        print(
            f"    unique_ratio           slope={ur_slope:+.4f}  (positive => more distinct replies)"
        )
        print(f"    pairwise_jacc_dist     slope={jd_slope:+.4f}  (positive => less sameness)")
        print(f"    template_expression    slope={ex_slope:+.4f}  (negative => less boilerplate)")
        print(
            f"    clearing_hits          slope={cl_slope:+.4f}  (negative => fewer odd fragments)"
        )

        print("\n  INTERPRETATION (heuristic, not human judgment):", flush=True)
        improved = (
            r10["unique_ratio"] >= r1["unique_ratio"]
            or r10["pairwise_jacc_dist"] >= r1["pairwise_jacc_dist"]
        ) and (
            r10["template_expression_hits"] <= r1["template_expression_hits"]
            or r10["clearing_hits"] <= r1["clearing_hits"]
        )
        worse = (
            r10["template_expression_hits"] > r1["template_expression_hits"] + 2
            and r10["unique_ratio"] < r1["unique_ratio"] - 0.05
        )
        if worse:
            print(
                "  • Net: metrics moved in a *worse* direction by round 10 (more repetition/templates)."
            )
        elif improved and (ur_slope > -0.01 and ex_slope <= 0.5):
            print(
                "  • Net: modest *improvement* toward less template-heavy / more distinct replies by round 10."
            )
        else:
            print(
                "  • Net: *mixed / flat* — no clear quality climb on these automatic scores alone."
            )
        print("  • For real quality, add human ratings or an external judge model.", flush=True)

    finally:
        if boot is not None:
            print("\nShutting down...", flush=True)
            boot.save_and_shutdown()
            print("Done.", flush=True)


if __name__ == "__main__":
    main()
