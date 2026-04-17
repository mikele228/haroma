"""Measure HTTP POST /chat round-trip time (HaromaX6 must be listening).

  python scripts/benchmark_chat_response_time.py
  python scripts/benchmark_chat_response_time.py --base http://127.0.0.1:8193 --repeat 3
  python scripts/benchmark_chat_response_time.py --depth normal --timeout 900
  python scripts/benchmark_chat_response_time.py --async --poll-interval 0.5
  python scripts/benchmark_chat_response_time.py --async --sse

``--async`` POSTs with ``"async": true`` (HTTP 202), then polls ``GET /chat/result``
until the reply is ready. ``--sse`` (implies ``--async``) completes via
``GET /chat/wait`` (SSE stream) instead of polling. Reports accept time (202)
and end-to-end time.

Uses ``depth`` (``normal``), not ``mode``, to match ``elarion_server_v2``.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    raise SystemExit(2)


def _sse_read_chat_payload(base: str, rid: str, max_wait_sec: float):
    """Read first JSON ``data:`` line from ``GET /chat/wait``. Returns (kind, obj, err)."""
    cap = max(5.0, min(7200.0, float(max_wait_sec)))
    params = {"id": rid, "max_wait_sec": str(int(cap))}
    try:
        with requests.get(
            f"{base}/chat/wait",
            params=params,
            stream=True,
            timeout=(10.0, cap + 90.0),
        ) as resp:
            if resp.status_code != 200:
                return "fail", None, f"GET /chat/wait -> {resp.status_code}"
            for raw in resp.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                line = raw.strip()
                if not line.startswith("data: "):
                    continue
                try:
                    obj = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if obj.get("status") == "pending_timeout":
                    return "poll", None, None
                if obj.get("error"):
                    return "fail", None, str(obj.get("error"))
                if obj.get("status") == "pending":
                    continue
                return "done", obj, None
    except requests.RequestException as e:
        return "fail", None, str(e)
    return "fail", None, "no data line in SSE response"


def _poll_chat_until_done(base: str, rid: str, deadline_ts: float, poll_interval: float):
    poll_url = f"{base}/chat/result"
    while time.perf_counter() < deadline_ts:
        try:
            pr = requests.get(
                poll_url,
                params={"id": rid},
                timeout=(5.0, 30.0),
            )
        except requests.RequestException as e:
            return None, str(e)
        if pr.status_code != 200:
            return None, f"GET /chat/result -> {pr.status_code}"
        try:
            pdata = pr.json() or {}
        except Exception:
            pdata = {}
        if pdata.get("status") != "pending":
            return pdata, None
        time.sleep(max(0.05, float(poll_interval)))
    return None, "poll timeout"


def _complete_async_roundtrip(
    base: str,
    rid: str,
    deadline_ts: float,
    poll_interval: float,
    use_sse: bool,
):
    """Try SSE wait first when ``use_sse``; then poll until ``deadline_ts``."""
    if use_sse:
        left = max(5.0, deadline_ts - time.perf_counter())
        kind, pdata, err = _sse_read_chat_payload(base, rid, left)
        if kind == "done" and pdata is not None:
            return pdata, None
        if kind == "fail":
            return None, err or "sse failed"
    return _poll_chat_until_done(base, rid, deadline_ts, poll_interval)


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark /chat response time")
    ap.add_argument("--base", default="http://127.0.0.1:8193", help="Server root URL")
    ap.add_argument(
        "--depth",
        choices=("normal",),
        default="normal",
        help="POST depth",
    )
    ap.add_argument("--message", default="ping", help="User message body")
    ap.add_argument("--repeat", type=int, default=1, help="Number of /chat calls after warmup")
    ap.add_argument("--warmup", type=int, default=0, help="Extra warmup POSTs (not measured)")
    ap.add_argument("--status-timeout", type=float, default=15.0)
    ap.add_argument(
        "--timeout",
        type=float,
        default=240.0,
        help="Per-request read timeout for /chat",
    )
    ap.add_argument(
        "--async",
        action="store_true",
        dest="async_mode",
        help="Non-blocking chat: 202 + poll /chat/result until response (measures accept + E2E)",
    )
    ap.add_argument(
        "--poll-interval",
        type=float,
        default=0.25,
        help="Seconds between GET /chat/result polls when using --async (default 0.25)",
    )
    ap.add_argument(
        "--sse",
        action="store_true",
        help="With async handoff: use GET /chat/wait (SSE) instead of polling /chat/result (implies --async)",
    )
    args = ap.parse_args()
    if args.sse:
        args.async_mode = True
    base = args.base.rstrip("/")

    try:
        t0 = time.perf_counter()
        r = requests.get(f"{base}/status", timeout=(5.0, float(args.status_timeout)))
        status_dt = time.perf_counter() - t0
    except requests.RequestException as e:
        print(f"GET /status failed: {e}", file=sys.stderr)
        return 1

    if r.status_code != 200:
        print(f"GET /status -> {r.status_code}", file=sys.stderr)
        return 1

    print(f"GET /status -> 200 in {status_dt:.3f}s")

    for _ in range(max(0, args.warmup)):
        try:
            wj = {"message": args.message, "depth": args.depth}
            if args.async_mode:
                wj["async"] = True
            wr = requests.post(f"{base}/chat", json=wj, timeout=(10.0, float(args.timeout)))
            if args.async_mode and wr.status_code == 202:
                wid = (wr.json() or {}).get("request_id")
                if wid:
                    wdead = time.perf_counter() + float(args.timeout)
                    _complete_async_roundtrip(
                        base,
                        wid,
                        wdead,
                        float(args.poll_interval),
                        bool(args.sse),
                    )
        except requests.RequestException:
            pass

    times: list[float] = []
    accept_times: list[float] = []
    for i in range(max(1, args.repeat)):
        msg = f"{args.message} [{i + 1}]" if args.repeat > 1 else args.message
        t0 = time.perf_counter()
        try:
            if args.async_mode:
                resp = requests.post(
                    f"{base}/chat",
                    json={"message": msg, "depth": args.depth, "async": True},
                    timeout=(10.0, 30.0),
                )
            else:
                resp = requests.post(
                    f"{base}/chat",
                    json={"message": msg, "depth": args.depth},
                    timeout=(10.0, float(args.timeout)),
                )
        except requests.RequestException as e:
            print(f"POST /chat failed: {e}", file=sys.stderr)
            return 1
        t_accept = time.perf_counter()
        accept_dt = t_accept - t0
        accept_times.append(accept_dt)

        if args.async_mode:
            if resp.status_code != 202:
                print(
                    f"POST /chat async expected 202, got {resp.status_code}",
                    file=sys.stderr,
                )
                return 1
            try:
                body = resp.json() or {}
            except Exception:
                body = {}
            rid = body.get("request_id")
            if not rid:
                print("POST /chat async: missing request_id", file=sys.stderr)
                return 1
            deadline = t0 + float(args.timeout)
            pdata, cerr = _complete_async_roundtrip(
                base,
                rid,
                deadline,
                float(args.poll_interval),
                bool(args.sse),
            )
            if pdata is None:
                print(cerr or "async chat: no result", file=sys.stderr)
                return 1
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            snippet = (pdata.get("response") or "")[:80].replace("\n", " ")
            mode = "SSE" if args.sse else "poll"
            print(
                f"POST+{mode} /chat async depth={args.depth} -> "
                f"accept={accept_dt:.3f}s e2e={elapsed:.3f}s | reply[:80]={snippet!r}",
            )
        else:
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            snippet = ""
            try:
                body = resp.json()
                snippet = (body.get("response") or "")[:80].replace("\n", " ")
            except Exception:
                body = {}
            print(
                f"POST /chat depth={args.depth} -> HTTP {resp.status_code} in {elapsed:.3f}s | "
                f"reply[:80]={snippet!r}",
            )

    print("---")
    print(f"n={len(times)}  min={min(times):.3f}s  max={max(times):.3f}s  mean={statistics.mean(times):.3f}s")
    if len(times) > 1:
        print(f"  stdev={statistics.stdev(times):.3f}s")
    if args.async_mode and accept_times:
        print(
            f"accept(202) n={len(accept_times)}  "
            f"min={min(accept_times):.3f}s  max={max(accept_times):.3f}s  "
            f"mean={statistics.mean(accept_times):.3f}s",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
