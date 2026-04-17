"""
Load training text from a JSON file (no user questions hardcoded here), then
repeat teach + warmup + test until the reply contains --needle.

Uses ``requests`` with separate connect/read timeouts so a stuck server cannot
hang the client indefinitely.

Usage:
  python scripts/chat_train_name_identity.py
  python scripts/chat_train_name_identity.py --messages path/to/custom.json
  python scripts/chat_train_name_identity.py --timeout 90 --connect-timeout 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

try:
    import requests
except ImportError:
    print("This script requires the 'requests' package (pip install requests).", file=sys.stderr)
    raise SystemExit(2)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MESSAGES = os.path.join(_SCRIPT_DIR, "chat_train_name_identity.messages.json")


def _load_messages(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("messages file root must be a JSON object")
    tq = data.get("test_question")
    if not isinstance(tq, str) or not tq.strip():
        raise ValueError("messages file must include non-empty string 'test_question'")
    warm = data.get("warmup_messages")
    if not isinstance(warm, list) or not all(isinstance(x, str) for x in warm):
        raise ValueError("messages file must include 'warmup_messages': [string, ...]")
    items = data.get("teach_items")
    if items is not None and not isinstance(items, list):
        raise ValueError("'teach_items' must be a list of {term, meaning} objects")
    if items:
        for it in items:
            if not isinstance(it, dict) or not str(it.get("term", "")).strip():
                raise ValueError("each teach_items entry needs non-empty 'term'")
            if not str(it.get("meaning", "")).strip():
                raise ValueError("each teach_items entry needs non-empty 'meaning'")
    return data


def _post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    connect_timeout: float,
    read_timeout: float,
) -> Tuple[int, Dict[str, Any]]:
    timeout = (connect_timeout, read_timeout)
    try:
        r = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.Timeout as e:
        raise RuntimeError(
            f"HTTP timeout (connect={connect_timeout}s read={read_timeout}s): {url} ({e})"
        ) from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTP error: {url} ({e})") from e
    try:
        body = r.json() if r.text else {}
    except json.JSONDecodeError:
        body = {"_raw": r.text[:500]}
    return r.status_code, body if isinstance(body, dict) else {"_raw": str(body)}


def _reply_text(body: Dict[str, Any]) -> str:
    return body.get("response") or body.get("reply") or body.get("text") or ""


def _check_status(base: str, connect_timeout: float, read_timeout: float) -> None:
    url = f"{base.rstrip('/')}/status"
    timeout = (connect_timeout, read_timeout)
    try:
        r = requests.get(url, timeout=timeout)
    except requests.exceptions.Timeout as e:
        raise RuntimeError(
            f"HTTP timeout (connect={connect_timeout}s read={read_timeout}s): {url} ({e})"
        ) from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTP error: {url} ({e})") from e
    if r.status_code != 200:
        raise RuntimeError(f"GET {url} -> {r.status_code}")


def _do_teach(
    base: str,
    teach_items: List[Dict[str, str]],
    connect_timeout: float,
    read_timeout: float,
) -> bool:
    teach_url = f"{base}/teach"
    try:
        status, body = _post_json(
            teach_url,
            {"items": teach_items},
            connect_timeout=connect_timeout,
            read_timeout=min(30.0, read_timeout),
        )
        print(f"[teach] {status} -> {body}")
        return 200 <= status < 300
    except RuntimeError as e:
        print(f"[teach] {e}", file=sys.stderr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Repeat teach + warmup + test until reply contains needle; "
        "prompts load from JSON (see --messages)."
    )
    ap.add_argument(
        "--messages",
        default=_DEFAULT_MESSAGES,
        help="JSON with teach_items, warmup_messages, test_question",
    )
    ap.add_argument(
        "--base",
        default="http://127.0.0.1:8193",
        help="Server root (no trailing /chat)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Per-request read timeout in seconds (default below server 120s stall)",
    )
    ap.add_argument(
        "--connect-timeout",
        type=float,
        default=8.0,
        help="TCP connect timeout in seconds",
    )
    ap.add_argument(
        "--depth",
        default="normal",
        choices=("normal",),
        help="POST /chat depth",
    )
    ap.add_argument(
        "--chat-only",
        action="store_true",
        help="Skip POST /teach (still repeats warmup + test)",
    )
    ap.add_argument(
        "--reteach-each-round",
        action="store_true",
        help="POST /teach at the start of every round (not only round 1)",
    )
    ap.add_argument(
        "--needle",
        default="Haroma",
        help="Substring that must appear in the test reply (case-insensitive)",
    )
    ap.add_argument(
        "--max-rounds",
        type=int,
        default=50,
        help="Stop after this many full training rounds without success",
    )
    ap.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="Seconds to sleep between rounds (0 = none)",
    )
    args = ap.parse_args()
    base = args.base.rstrip("/")
    needle_lower = (args.needle or "").lower()
    if not needle_lower:
        print("--needle must be non-empty", file=sys.stderr)
        return 2

    if not os.path.isfile(args.messages):
        print(f"Messages file not found: {args.messages}", file=sys.stderr)
        print(f"Create it or pass --messages (default sits beside this script).", file=sys.stderr)
        return 2

    try:
        cfg = _load_messages(args.messages)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"Invalid messages file: {e}", file=sys.stderr)
        return 2

    test_question = cfg["test_question"].strip()
    warmup_messages: List[str] = [str(m).strip() for m in cfg["warmup_messages"] if str(m).strip()]
    teach_items: List[Dict[str, str]] = list(cfg.get("teach_items") or [])

    ct = float(args.connect_timeout)
    rt = float(args.timeout)

    try:
        _check_status(base, ct, min(10.0, rt))
    except RuntimeError as e:
        print(f"Server not reachable at {base}: {e}", file=sys.stderr)
        print("Start HaromaX6 (e.g. python main.py) and retry.", file=sys.stderr)
        return 1

    chat_url = f"{base}/chat"

    for round_idx in range(1, args.max_rounds + 1):
        print(f"\n=== round {round_idx}/{args.max_rounds} ===")

        if not args.chat_only and teach_items and (round_idx == 1 or args.reteach_each_round):
            if not _do_teach(base, teach_items, ct, rt):
                return 1

        for j, msg in enumerate(warmup_messages, start=1):
            try:
                status, body = _post_json(
                    chat_url,
                    {"message": msg, "depth": args.depth},
                    connect_timeout=ct,
                    read_timeout=rt,
                )
            except RuntimeError as e:
                print(f"[warmup {j}] {e}", file=sys.stderr)
                print(
                    "Tip: adjust --timeout / --connect-timeout or fix the server.", file=sys.stderr
                )
                return 1
            if status >= 400:
                print(f"[warmup {j}] HTTP {status}: {body}", file=sys.stderr)
                return 1
            reply = _reply_text(body)
            print(f"[warmup {j}] user: {msg[:72]!r}...")
            print(f"  -> {status} reply: {reply[:400]!r}{'...' if len(reply) > 400 else ''}")

        try:
            status, body = _post_json(
                chat_url,
                {"message": test_question, "depth": args.depth},
                connect_timeout=ct,
                read_timeout=rt,
            )
        except RuntimeError as e:
            print(f"[test] {e}", file=sys.stderr)
            print("Tip: adjust --timeout / --connect-timeout or fix the server.", file=sys.stderr)
            return 1
        if status >= 400:
            print(f"[test] HTTP {status}: {body}", file=sys.stderr)
            return 1

        reply = _reply_text(body)
        print(f"[test] user: {test_question!r}")
        print(f"  -> {status} reply: {reply!r}")

        if needle_lower in reply.lower():
            print(f"\nOK: reply contains {args.needle!r} after {round_idx} round(s).")
            return 0

        print(f"  (no {args.needle!r} in reply; continuing)")
        if args.pause > 0:
            time.sleep(args.pause)

    print(
        f"\nGive up: no reply containing {args.needle!r} after {args.max_rounds} round(s).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
