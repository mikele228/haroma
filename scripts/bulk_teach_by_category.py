"""
Send N categorized teaching prompts to Elarion's POST /chat and write a
quality-over-time report (JSON + console summary).

Requires a running server (python main.py). Default:
  http://127.0.0.1:8193/chat

Usage:
  python scripts/bulk_teach_by_category.py
  python scripts/bulk_teach_by_category.py --url http://127.0.0.1:8193 --count 1000
  python scripts/bulk_teach_by_category.py --schedule mixed --mixed-every 8 --count 500
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any, Dict, List, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_OUT = os.path.join(_ROOT, "logs", "bulk_teach_report.json")

_RQI_PATH = os.path.join(_ROOT, "scripts", "run_quality_iterations.py")
_spec = importlib.util.spec_from_file_location("run_quality_iterations", _RQI_PATH)
assert _spec is not None, f"Cannot find module at {_RQI_PATH}"
_run_quality = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_run_quality)
score_reply_detailed = _run_quality.score_reply_detailed

# -----------------------------------------------------------------------------
# Category pools: short teaching-style prompts (varied subjects)
# -----------------------------------------------------------------------------
_CATEGORIES: Dict[str, List[str]] = {
    "physics": [
        "Teach me one idea: why does entropy tend to increase?",
        "In simple terms, what is wave–particle duality?",
        "What is conserved in an isolated system in classical mechanics?",
        "Explain gravitational time dilation without math.",
        "Why do we use Lagrangians in physics?",
    ],
    "biology": [
        "What is the difference between DNA replication and transcription?",
        "How does natural selection differ from genetic drift?",
        "Explain what a ribosome does in one paragraph.",
        "Why is ATP called the energy currency of the cell?",
        "What is homeostasis and a concrete example?",
    ],
    "chemistry": [
        "What makes a covalent bond different from ionic?",
        "Explain pH in practical terms for daily life.",
        "What is activation energy in a chemical reaction?",
        "Why does catalyst speed up a reaction without being consumed?",
    ],
    "math_foundations": [
        "Explain proof by induction with a tiny example.",
        "What is the difference between a vector and a scalar?",
        "Intuitively, what does a derivative measure?",
        "Why can you not divide by zero?",
    ],
    "cs_algorithms": [
        "Explain big-O notation with one example.",
        "When would you prefer a hash table over a binary search tree?",
        "What problem does dynamic programming solve better than plain recursion?",
        "What is the difference between concurrency and parallelism?",
    ],
    "software_engineering": [
        "What is idempotency in APIs?",
        "Why do we use version control for teams?",
        "Explain technical debt in one analogy.",
        "What is the point of unit tests vs integration tests?",
    ],
    "history": [
        "What were key effects of the printing press on Europe?",
        "Summarize why the Industrial Revolution mattered.",
        "What triggered the fall of the Western Roman Empire (high level)?",
    ],
    "economics": [
        "Explain supply and demand with a concrete market.",
        "What is opportunity cost?",
        "Why might inflation happen in simple terms?",
    ],
    "philosophy": [
        "What is the is–ought problem?",
        "Explain utilitarianism vs deontology in one sentence each.",
        "What does 'consciousness is like something' mean?",
    ],
    "psychology": [
        "What is cognitive bias? Give one named example.",
        "How does classical conditioning differ from operant conditioning?",
        "What is working memory vs long-term memory?",
    ],
    "ethics_law": [
        "What is procedural fairness vs outcome fairness?",
        "Why might rights-based ethics conflict with utilitarian tradeoffs?",
    ],
    "medicine_health": [
        "What is the difference between efficacy and effectiveness in trials?",
        "Why is randomized control trial evidence stronger than anecdotes?",
    ],
    "arts_music": [
        "How does harmony differ from melody?",
        "What role does negative space play in visual composition?",
    ],
    "geography_earth": [
        "Why do we have seasons on Earth?",
        "What drives plate tectonics at a high level?",
    ],
    "language_linguistics": [
        "What is the difference between syntax and semantics?",
        "Why are some languages more morphologically complex than others?",
    ],
}

_RE_EXPR = re.compile(
    r"expression is at the heart|ties to expression|need for expression",
    re.I,
)
_RE_CLEAR = re.compile(r"the clearing", re.I)
_STOP = frozenset(
    "a an the to of in for on with and or is are was were be been being "
    "it this that these those you i we they he she as at by from into "
    "what how why when where which who me my your our their about can could "
    "would should will just than then so if not no yes but all any some more "
    "most very much such do does did doing done teach explain one".split()
)


def _norm_tokens(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9']+", s.lower()) if len(w) > 2 and w not in _STOP}


def _overlap(user: str, reply: str) -> float:
    u, r = _norm_tokens(user), _norm_tokens(reply)
    if not u:
        return 0.0
    return len(u & r) / max(len(u), 1)


def _build_schedule(count: int, seed: int) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    cats = list(_CATEGORIES.keys())
    sched: List[Tuple[str, str]] = []
    i = 0
    while len(sched) < count:
        cat = cats[i % len(cats)]
        pool = _CATEGORIES[cat]
        msg = rng.choice(pool)
        # light paraphrase variety
        if rng.random() < 0.35:
            msg = f"[Learn {cat.replace('_', ' ')}] {msg}"
        sched.append((cat, msg))
        i += 1
    return sched


def _base_from_chat_url(chat_url: str) -> str:
    if chat_url.rstrip("/").endswith("/chat"):
        return chat_url.rstrip("/")[: -len("/chat")].rstrip("/") or chat_url
    return chat_url.rstrip("/")


def _check_server(chat_url: str, timeout: float = 5.0) -> None:
    base = _base_from_chat_url(chat_url)
    status_url = f"{base}/status"
    req = urllib.request.Request(status_url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        _ = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"GET {status_url} -> {resp.status}")


def _post_json(url: str, payload: Dict, timeout: float) -> Tuple[int, Dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return resp.status, json.loads(raw) if raw else {}


def _chunk_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    replies = [r.get("reply") or "" for r in rows]
    un = len({x.strip().lower() for x in replies})
    lens = [len(x) for x in replies]
    overlap = [r.get("keyword_overlap") or 0 for r in rows]
    qvals = [
        float((r.get("quality") or {}).get("overall", 0.0)) for r in rows if not r.get("error")
    ]
    return {
        "n": len(rows),
        "mean_chars": round(sum(lens) / max(len(lens), 1), 1),
        "unique_replies": un,
        "unique_ratio": round(un / max(len(replies), 1), 3),
        "mean_keyword_overlap": round(sum(overlap) / max(len(overlap), 1), 3),
        "mean_quality": round(sum(qvals) / max(len(qvals), 1), 2) if qvals else 0.0,
        "template_expr_hits": sum(1 for x in replies if _RE_EXPR.search(x)),
        "clearing_hits": sum(1 for x in replies if _RE_CLEAR.search(x)),
        "timeouts": sum(1 for r in rows if r.get("error")),
    }


def _chat_depth() -> str:
    """POST /chat depth (only ``normal`` is supported)."""
    return "normal"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8193/chat", help="Full /chat URL")
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument(
        "--depth",
        default="normal",
        choices=("normal",),
        help="POST /chat depth (only ``normal`` is supported)",
    )
    ap.add_argument(
        "--schedule",
        default="uniform",
        choices=("uniform", "mixed"),
        help="uniform: all messages use normal depth; mixed is legacy (same as uniform)",
    )
    ap.add_argument(
        "--mixed-every",
        type=int,
        default=10,
        help="With --schedule mixed: use normal depth every Nth message (1-based multiples).",
    )
    ap.add_argument(
        "--debug-recall",
        action="store_true",
        help="Send debug_recall:true to /chat (response may include recall_debug).",
    )
    ap.add_argument("--timeout", type=float, default=125.0, help="HTTP timeout per message")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds between requests")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=_DEFAULT_OUT)
    ap.add_argument("--chunks", type=int, default=10, help="Report buckets (e.g. 10 x 100)")
    args = ap.parse_args()

    try:
        _check_server(args.url)
    except Exception as exc:
        print(
            f"Cannot reach server ({exc}). Start Elarion first, e.g.:\n"
            f"  python main.py\n"
            f"Then retry. Expected GET {_base_from_chat_url(args.url)}/status",
            flush=True,
        )
        sys.exit(2)

    if args.schedule == "mixed" and args.mixed_every < 1:
        print("--mixed-every must be >= 1", flush=True)
        sys.exit(2)

    schedule = _build_schedule(args.count, args.seed)
    rows: List[Dict[str, Any]] = []
    t_start = time.time()

    depth_desc = f"schedule={args.schedule}" if args.schedule == "mixed" else f"depth={args.depth}"
    print(
        f"POST {args.url} | n={args.count} {depth_desc}"
        + (f" mixed_every={args.mixed_every}" if args.schedule == "mixed" else ""),
        flush=True,
    )

    for idx, (cat, message) in enumerate(schedule, start=1):
        depth_used = _chat_depth()
        t0 = time.time()
        err = None
        resp_text = ""
        strat = ""
        recall_dbg = None
        try:
            payload: Dict[str, Any] = {"message": message, "depth": depth_used}
            if args.debug_recall:
                payload["debug_recall"] = True
            status, body = _post_json(
                args.url,
                payload,
                timeout=args.timeout,
            )
            if status != 200:
                err = f"http_{status}"
            elif isinstance(body, dict) and body.get("error"):
                err = str(body.get("error"))
            else:
                resp_text = str((body or {}).get("response", "") or "")
                strat = str((body or {}).get("strategy", "") or "")
                if args.debug_recall and isinstance(body, dict):
                    recall_dbg = body.get("recall_debug")
        except urllib.error.URLError as e:
            err = f"urlerror:{e.reason}"
        except TimeoutError:
            err = "timeout"
        except Exception as e:
            err = f"{type(e).__name__}:{e}"

        dt = time.time() - t0
        qdet = (
            score_reply_detailed(message, resp_text)
            if not err
            else {"overall": 0.0, "components": {}}
        )
        row = {
            "i": idx,
            "category": cat,
            "depth": depth_used,
            "user": message,
            "reply": resp_text if not err else "",
            "strategy": strat,
            "latency_s": round(dt, 3),
            "keyword_overlap": round(_overlap(message, resp_text), 3) if not err else 0.0,
            "quality": qdet,
            "recall_debug": recall_dbg,
            "error": err,
        }
        rows.append(row)
        if args.sleep > 0:
            time.sleep(args.sleep)

        if idx % 50 == 0 or idx == 1:
            qv = float(qdet.get("overall", 0.0))
            print(
                f"  {idx}/{args.count} cat={cat} d={depth_used} "
                f"lat={dt:.1f}s err={err or 'ok'} q={qv:.1f} reply_len={len(resp_text)}",
                flush=True,
            )

    total_elapsed = time.time() - t_start

    # Chunk report (progress over “time” = message index)
    chunk_n = max(1, args.count // args.chunks)
    chunk_reports = []
    for c in range(args.chunks):
        lo = c * chunk_n
        hi = min((c + 1) * chunk_n, len(rows))
        if lo >= hi:
            break
        sub = rows[lo:hi]
        m = _chunk_metrics(sub)
        m["chunk"] = c + 1
        m["i_range"] = [lo + 1, hi]
        chunk_reports.append(m)

    # First vs last chunk comparison
    first_m = chunk_reports[0] if chunk_reports else {}
    last_m = chunk_reports[-1] if chunk_reports else {}

    by_cat: Dict[str, Any] = defaultdict(list)
    for r in rows:
        if not r.get("error"):
            by_cat[r["category"]].append(r["keyword_overlap"])
    cat_summary = {
        k: {
            "n": len(v),
            "mean_overlap": round(sum(v) / max(len(v), 1), 3),
        }
        for k, v in sorted(by_cat.items())
    }

    ok_quality = [
        float(r["quality"]["overall"]) for r in rows if not r.get("error") and r.get("quality")
    ]
    summary = {
        "url": args.url,
        "count": args.count,
        "depth_uniform": args.depth,
        "schedule": args.schedule,
        "mixed_every": args.mixed_every if args.schedule == "mixed" else None,
        "debug_recall": bool(args.debug_recall),
        "elapsed_sec": round(total_elapsed, 1),
        "errors": sum(1 for r in rows if r.get("error")),
        "chunk_size": chunk_n,
        "chunks": chunk_reports,
        "mean_quality_all": (
            round(sum(ok_quality) / max(len(ok_quality), 1), 2) if ok_quality else 0.0
        ),
        "first_vs_last_chunk": {
            "mean_chars": (first_m.get("mean_chars"), last_m.get("mean_chars")),
            "unique_ratio": (first_m.get("unique_ratio"), last_m.get("unique_ratio")),
            "mean_keyword_overlap": (
                first_m.get("mean_keyword_overlap"),
                last_m.get("mean_keyword_overlap"),
            ),
            "mean_quality": (
                first_m.get("mean_quality"),
                last_m.get("mean_quality"),
            ),
            "template_expr_hits": (
                first_m.get("template_expr_hits"),
                last_m.get("template_expr_hits"),
            ),
            "clearing_hits": (first_m.get("clearing_hits"), last_m.get("clearing_hits")),
        },
        "by_category_overlap": cat_summary,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_blob = {"summary": summary, "rows": rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_blob, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(" REPORT (heuristic proxies -- not human grades)")
    print("=" * 60)
    print(f"  Wrote: {args.out}")
    print(f"  Errors/timeouts: {summary['errors']} / {args.count}")
    print(f"  Wall time: {summary['elapsed_sec']:.1f}s")
    print(
        f"\n  Mean quality (0–100, same heuristics as run_quality_iterations): "
        f"{summary['mean_quality_all']}",
        flush=True,
    )
    print("\n  Chunk trend (mean_keyword_overlap ~= user/reply token overlap):")
    for ch in chunk_reports:
        print(
            f"    ch{ch['chunk']} msgs {ch['i_range']} | "
            f"qual={ch['mean_quality']} "
            f"uniq={ch['unique_ratio']} overlap={ch['mean_keyword_overlap']} "
            f"expr_tpl={ch['template_expr_hits']} clearing={ch['clearing_hits']} "
            f"mean_chars={ch['mean_chars']}",
            flush=True,
        )
    fv = summary["first_vs_last_chunk"]
    print("\n  First chunk -> last chunk:")
    print(f"    mean_quality:           {fv['mean_quality'][0]} -> {fv['mean_quality'][1]}")
    print(f"    unique_ratio:           {fv['unique_ratio'][0]} -> {fv['unique_ratio'][1]}")
    print(
        f"    mean_keyword_overlap:   {fv['mean_keyword_overlap'][0]} -> {fv['mean_keyword_overlap'][1]}"
    )
    print(
        f"    template_expr_hits:     {fv['template_expr_hits'][0]} -> {fv['template_expr_hits'][1]} (lower often better)"
    )
    print(
        f"    clearing_hits:          {fv['clearing_hits'][0]} -> {fv['clearing_hits'][1]} (lower often saner)"
    )
    print(
        "\n  Interpretation: improvement here means rising diversity / overlap with question "
        "and fewer template-like fragments; not proof of smarter answers without human review.",
        flush=True,
    )


if __name__ == "__main__":
    main()
