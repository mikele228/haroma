"""
Print a human-readable snapshot of what each Elarion agent is doing
(ticks, intervals, buffers, background training/dreams, persona emotion, bus).

Usage:
  cd HaromaX6
  python scripts/print_agent_activity.py
  python scripts/print_agent_activity.py --watch 5
  python scripts/print_agent_activity.py --json
  python scripts/print_agent_activity.py --no-shutdown-save
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
from typing import Any, Dict, List

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _truncate(s: Any, n: int = 120) -> str:
    t = "" if s is None else str(s).replace("\n", " ").strip()
    return t if len(t) <= n else t[: n - 2] + "…"


def _fmt_emotion(e: Any) -> str:
    if not isinstance(e, dict):
        return _truncate(e, 100)
    v = e.get("valence", "?")
    ar = e.get("arousal", "?")
    dom = e.get("dominant", e.get("dominant_emotion", ""))
    return f"valence={v} arousal={ar} dominant={_truncate(dom, 48)}"


def _lines_for_agent(a: Dict[str, Any]) -> List[str]:
    tid = a.get("agent_id", "?")
    typ = a.get("agent_type", "?")
    ticks = a.get("tick_count") or 0
    alive = a.get("alive", False)
    inter = a.get("tick_interval") or "?"
    uptime = a.get("uptime_s") or 0
    head = (
        f"  [{typ:12}] {tid:12}  ticks={ticks:6}  "
        f"uptime={uptime:.0f}s  interval={inter}s  alive={alive}"
    )
    out = [head]
    if a.get("error_count"):
        out.append(f"      errors={a['error_count']} last_tick_ms={a.get('last_tick_ms')}")

    if typ == "input":
        b = a.get("buffer") or {}
        out.append(
            f"      buffer: text_pending={b.get('text_pending', '?')} "
            f"sensor_pending={b.get('sensor_pending', '?')} "
            f"response_log={b.get('response_log_size', '?')}"
        )
    elif typ == "trueself":
        out.append(
            f"      pending_delegations={a.get('pending_delegations', 0)} "
            f"fast_path_threshold={a.get('fast_path_threshold')} "
            f"delegation_timeout_ms={a.get('delegation_timeout_ms')}"
        )
    elif typ == "background":
        div = a.get("divergence")
        if div is not None:
            out.append(f"      divergence={round(float(div), 4)}")
        ld = a.get("last_training_losses") or {}
        if ld:
            bits = [f"{k}={round(float(v or 0), 4)}" for k, v in list(ld.items())[:8]]
            out.append(f"      training_losses: {' '.join(bits)}")
        dream = a.get("last_dream") or {}
        if dream:
            dn = dream.get("dream_narrative") or dream.get("narrative") or ""
            out.append(f"      last_dream: {_truncate(dn, 140)}")
        rec = a.get("last_reconcile") or {}
        if isinstance(rec, dict) and rec:
            out.append(f"      reconcile_branches={len(rec)}")
        aut = a.get("autonomy") or {}
        if aut:
            out.append(
                f"      autonomy: intentions={aut.get('intentions_recorded', 0)} "
                f"stim_q={aut.get('stimulus_queue_len', '?')}/{aut.get('stimulus_queue_max', '?')}"
            )
        wl = a.get("web_learn") or {}
        if wl.get("enabled"):
            out.append(
                f"      web_learn: queue={wl.get('queue_len', '?')} "
                f"fp={wl.get('fingerprints_cached', '?')}"
            )
    elif typ == "persona":
        out.append(
            f"      {a.get('persona_name', tid)}  cycles={a.get('persona_cycles', 0)} "
            f"default={a.get('is_default')}"
        )
        out.append(f"      emotion: {_fmt_emotion(a.get('emotion'))}")
        wm = a.get("working_memory") or {}
        if isinstance(wm, dict) and wm:
            out.append(f"      working_memory: {_truncate(wm, 100)}")
        conv = a.get("conversation") or {}
        if isinstance(conv, dict) and conv:
            out.append(f"      conversation: {_truncate(conv, 100)}")

    return out


def _collect_boot_stats() -> Dict[str, Any]:
    import mind.elarion_server_v2 as srv

    if srv.boot_agent is None:
        srv._init()
    return srv.boot_agent.stats()


def _print_snapshot(as_json: bool) -> None:
    st = _collect_boot_stats()
    if as_json:
        print(json.dumps(st, indent=2, default=str, ensure_ascii=False))
        return

    shared_cc = None
    try:
        import mind.elarion_server_v2 as srv

        if srv.boot_agent and srv.boot_agent.shared:
            shared_cc = srv.boot_agent.shared.cycle_count
    except Exception as _e:
        print(f"[PrintActivity] snapshot read error: {_e}", flush=True)

    print("\n=== Elarion agent activity ===")
    if shared_cc is not None:
        print(f"  shared.cycle_count={shared_cc}")
    mb = st.get("message_bus") or {}
    if mb:
        subs = mb.get("subscriptions") or {}
        sub_channels = len(subs) if isinstance(subs, dict) else 0
        print(
            f"  message_bus: published={mb.get('total_published', '?')} "
            f"dead_letter={mb.get('dead_letter_count', '?')} "
            f"claimed={mb.get('total_claimed', '?')} "
            f"subscription_channels={sub_channels}"
        )
    print("")
    for child in st.get("child_agents") or []:
        for line in _lines_for_agent(child):
            print(line)
        print("")
    print("=== end ===\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Print agent activity snapshot.")
    ap.add_argument(
        "--watch",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Re-print every SEC seconds (0 = once)",
    )
    ap.add_argument("--json", action="store_true", help="Full JSON (verbose)")
    ap.add_argument(
        "--no-shutdown-save",
        action="store_true",
        help="Do not register persistence save on process exit (faster for scripts)",
    )
    args = ap.parse_args()

    import mind.elarion_server_v2 as srv

    if args.no_shutdown_save:
        try:
            atexit.unregister(srv._shutdown_save)
        except Exception as _e:
            print(f"[PrintActivity] atexit unregister error: {_e}", flush=True)

    if args.watch <= 0:
        _print_snapshot(args.json)
        return

    try:
        while True:
            _print_snapshot(args.json)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\n[print_agent_activity] stopped.", flush=True)


if __name__ == "__main__":
    main()
