#!/usr/bin/env python3
"""
Optional **terminal** chat with a small ASCII buddy (POST /chat). The main chat UI is the
**web app**: open ``http://127.0.0.1:8193/`` (or your ``HAROMA_URL``) in a browser — chat only;
no separate desktop chat GUI is required.

This script shows ``affect``, ``strategy``, ``persona_name``, optional ``llm_context.latency_ms``,
a mood trail, and wraps replies. Spinner on stderr while waiting.

Usage:
  python scripts/haroma_chat_buddy.py
  python scripts/haroma_chat_buddy.py --async-chat --width 100

**Tk desktop window** is **off by default**. Use ``--gui`` or ``HAROMA_CHAT_BUDDY_GUI=1`` only
if you explicitly want it; prefer the browser for normal chat.

**ASCII vs Unicode** buddy art is auto-detected; override with ``--ascii`` / ``--unicode`` or
``HAROMA_BUDDY_ASCII`` / ``HAROMA_BUDDY_UNICODE``.

Slash commands: /help, /status, /quit

Environment:
  HAROMA_URL               Base URL (default http://127.0.0.1:8193)
  HAROMA_HTTP_BEARER       Optional Authorization: Bearer …
  HAROMA_CHAT_BUDDY_GUI    1/true = opt-in Tk window (same as --gui)
  HAROMA_BUDDY_ASCII       1 = force ASCII art
  HAROMA_BUDDY_UNICODE     1 = force Unicode/box-drawing art
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


def _base_url() -> str:
    u = (os.environ.get("HAROMA_URL") or "http://127.0.0.1:8193").strip().rstrip("/")
    return u


def _bearer() -> str:
    return (os.environ.get("HAROMA_HTTP_BEARER") or "").strip()


def _http_json(
    method: str,
    url: str,
    body: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 600.0,
) -> Tuple[int, Any]:
    h = dict(headers or {})
    h.setdefault("Content-Type", "application/json")
    if _bearer():
        h["Authorization"] = f"Bearer {_bearer()}"
    req = urllib.request.Request(url, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            code = getattr(resp, "status", 200)
            try:
                return code, json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                return code, {"_raw": raw}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            return e.code, json.loads(raw) if raw.strip() else {"error": str(e)}
        except json.JSONDecodeError:
            return e.code, {"error": str(e), "_raw": raw}
    except OSError as e:
        return -1, {"error": str(e)}


_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_ASCII = "|/-\\"


class _Spinner:
    """Print a rotating spinner on stderr until ``stop()`` is called."""

    def __init__(self, label: str, *, ascii_only: bool, use_color: bool) -> None:
        self._label = label
        self._ascii = ascii_only
        self._color = use_color
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        seq = _SPINNER_ASCII if self._ascii else _SPINNER
        i = 0
        while not self._stop.wait(0.08):
            ch = seq[i % len(seq)]
            i += 1
            msg = f"\r\033[2K{_ansi('35', ch + ' ' + self._label, self._color)}"
            if not self._color:
                msg = f"\r\033[2K{ch} {self._label}"
            try:
                sys.stderr.write(msg)
                sys.stderr.flush()
            except OSError:
                break

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            sys.stderr.write("\r\033[2K")
            sys.stderr.flush()
        except OSError:
            pass


def _ansi(code: str, s: str, use_color: bool) -> str:
    if not use_color:
        return s
    return f"\033[{code}m{s}\033[0m"


def _emotion_color(emo: str) -> str:
    e = str(emo).lower().strip()
    return {
        "joy": "33",
        "wonder": "35",
        "curiosity": "36",
        "fear": "34",
        "sadness": "94",
        "anger": "31",
        "resolve": "32",
        "peace": "36",
        "surprise": "33",
        "neutral": "37",
    }.get(e, "37")


# Faces — EpisodeContext dominant_emotion + extras mapped to neutral-adjacent
_FACES: Dict[str, str] = {
    "neutral": "  ( · · )   ",
    "joy": "  ( ‿‿ )    ",
    "wonder": "  ( ☆‿☆ )  ",
    "curiosity": "  ( ◕‿◕ )  ",
    "fear": "  ( @_@ )   ",
    "sadness": "  ( T_T )   ",
    "anger": "  ( >_< )   ",
    "resolve": "  ( •̀ᴗ•́ ) ",
    "peace": "  ( -.- )   ",
    "surprise": "  ( O_O )   ",
    "melancholy": "  ( ‸ ‸ )  ",
}
_FACES_ASCII: Dict[str, str] = {k: v for k, v in _FACES.items()}
_FACES_ASCII.update(
    {
        "neutral": "  ( o o )   ",
        "resolve": "  ( -_- )b  ",
        "melancholy": "  ( u u )   ",
    }
)


def _face(emo: str, *, ascii_only: bool, frame: int) -> str:
    table = _FACES_ASCII if ascii_only else _FACES
    key = str(emo).lower().strip()
    if key not in table:
        key = "neutral"
    base = table[key]
    # Tiny idle shift for calm moods (alternating spaces)
    if key in ("neutral", "peace") and (frame % 4) == 2:
        return " " + base
    return base


def _strategy_label(s: str) -> str:
    s = (s or "").strip().lower()
    return {
        "": "—",
        "llm_context": "packed LLM reply",
        "inform": "inform (knowledge)",
        "inquire": "inquire (question)",
        "reflect": "reflect (memory)",
        "advance_goal": "advance goal",
        "multi_goal": "multi-goal fuse",
        "derivation": "derivation",
        "empathize": "empathize",
    }.get(s, s or "—")


def _arms(strategy: str, *, ascii_only: bool) -> str:
    s = (strategy or "").lower()
    if ascii_only:
        if "reflect" in s:
            return r"   \ | /   "
        if "inquire" in s:
            return r"   ? | ?   "
        if "llm" in s or "context" in s:
            return r"   / | \   "
        if "advance" in s or "goal" in s or "multi" in s:
            return r"  >>|<<   "
        if "inform" in s:
            return r"  > | <   "
        return r"   / | \   "
    if "reflect" in s:
        return r"   ╲ │ ╱   "
    if "inquire" in s:
        return r"   ? │ ?   "
    if "llm" in s or "context" in s:
        return r"   ╱ │ ╲   "
    if "advance" in s or "goal" in s or "multi" in s:
        return r"  ≫│≪   "
    if "inform" in s:
        return r"   ▶ │ ◀   "
    return r"   ╱ │ ╲   "


def _legs(frame: int, *, ascii_only: bool) -> str:
    phase = frame % 4
    if ascii_only:
        if phase in (0, 2):
            return r"    / \    "
        return r"    | |    "
    if phase in (0, 2):
        return r"    ╱ ╲    "
    return r"    │ │    "


def _bar(label: str, x: float, width: int, *, ascii_only: bool) -> str:
    x = max(0.0, min(1.0, float(x)))
    filled = int(round(x * width))
    if ascii_only:
        return f"{label} [{'#' * filled}{'-' * (width - filled)}] {x:.2f}"
    return f"{label} [{'█' * filled}{'░' * (width - filled)}] {x:.2f}"


def _trail(history: List[str], max_chars: int = 24) -> str:
    if not history:
        return ""
    sym = {
        "joy": "↑",
        "anger": "!",
        "fear": "?",
        "sadness": "↓",
        "neutral": "·",
        "curiosity": "c",
        "wonder": "*",
    }
    parts = [sym.get(h, (h[:1] if h else "·")) for h in history[-max_chars:]]
    return "mood trail: " + "".join(parts)


def render_buddy(
    affect: Dict[str, Any],
    strategy: str,
    cycle: Any,
    *,
    frame: int,
    use_color: bool,
    ascii_only: bool,
    persona_name: str,
    emotion_history: List[str],
    llm_latency_ms: float,
) -> str:
    emo = str(affect.get("dominant_emotion") or "neutral")
    intensity = float(affect.get("intensity") or 0.0)
    valence = float(affect.get("valence") or 0.0)
    arousal = float(affect.get("arousal") or 0.0)

    face = _face(emo, ascii_only=ascii_only, frame=frame)
    if intensity > 0.65 and emo in ("anger", "fear", "surprise"):
        face = face.rstrip() + " !"

    ec = _emotion_color(emo)
    strat_h = _strategy_label(strategy)

    box_top = "  +-- Haroma (Elarion) --+" if ascii_only else "  ╭── Haroma (Elarion) ──╮"
    who = (persona_name or "Elarion")[:14]
    box_mid = (
        f"  | {who:<14} c={str(cycle)[:8]:<8}|"
        if ascii_only
        else f"  │ {who:<14} c={str(cycle)[:8]:<8}│"
    )
    box_mo = (
        f"  | {emo[:12]:<12} {strat_h[:16]:<16}|"
        if ascii_only
        else f"  │ {emo[:12]:<12} {strat_h[:16]:<16}│"
    )
    box_bot = "  +----------------------+" if ascii_only else "  ╰──────────────────────╯"

    lat = ""
    if llm_latency_ms > 0:
        lat = f"  llm latency: {llm_latency_ms:.0f} ms"

    lines = [
        "",
        _ansi("1;36", box_top, use_color),
        _ansi("36", box_mid, use_color),
        _ansi("36", box_mo, use_color),
        _ansi("36", box_bot, use_color),
        _ansi(ec, face, use_color),
        _ansi("32", _arms(strategy, ascii_only=ascii_only), use_color),
        _ansi("32", _legs(frame, ascii_only=ascii_only), use_color),
        f"  {_bar('intensity', intensity, 12, ascii_only=ascii_only)}",
        f"  {_bar('valence  ', (valence + 1) / 2, 12, ascii_only=ascii_only)}",
        f"  {_bar('arousal  ', abs(arousal), 12, ascii_only=ascii_only)}",
        _ansi("90", f"  {_trail(emotion_history)}", use_color) if emotion_history else "",
    ]
    if lat:
        lines.append(_ansi("90", lat, use_color))
    lines.append("")
    return "\n".join(lines)


def _wrap_reply(text: str, width: int) -> str:
    t = (text or "").strip()
    if not t:
        return "(empty response)"
    if width < 20:
        width = 72
    return textwrap.fill(t, width=width, replace_whitespace=False, drop_whitespace=False)


def ping_server(base: str, timeout: float = 5.0) -> Tuple[bool, str, float]:
    t0 = time.perf_counter()
    code, data = _http_json("GET", f"{base}/status", timeout=timeout)
    dt = (time.perf_counter() - t0) * 1000
    if code == 200:
        return True, "ok", dt
    if code == -1:
        return False, str(data.get("error", data)), dt
    return False, f"HTTP {code}", dt


def chat_sync(base: str, message: str, depth: str = "normal") -> Dict[str, Any]:
    url = f"{base}/chat"
    payload = json.dumps({"message": message, "depth": depth, "async": False}).encode("utf-8")
    code, data = _http_json("POST", url, body=payload)
    if code != 200:
        return {"error": f"HTTP {code}", "body": data}
    if not isinstance(data, dict):
        return {"error": "bad_json", "body": data}
    return data


def chat_async_poll(
    base: str,
    message: str,
    depth: str = "normal",
    poll_sec: float = 0.5,
) -> Dict[str, Any]:
    url = f"{base}/chat"
    payload = json.dumps({"message": message, "depth": depth, "async": True}).encode("utf-8")
    code, data = _http_json("POST", url, body=payload, timeout=60.0)
    if code == 202 and isinstance(data, dict) and data.get("request_id"):
        rid = data["request_id"]
        result_url = f"{base}/chat/result?id={urllib.parse.quote(str(rid), safe='')}"
        for _ in range(int(600 / poll_sec)):
            time.sleep(poll_sec)
            c2, d2 = _http_json("GET", result_url, timeout=60.0)
            if c2 == 200 and isinstance(d2, dict):
                if d2.get("status") == "pending":
                    continue
                if "response" in d2:
                    return d2
            if c2 not in (200, 202):
                return {"error": f"poll HTTP {c2}", "body": d2}
        return {"error": "timeout waiting for async result"}
    if code == 200 and isinstance(data, dict):
        return data
    return {"error": f"HTTP {code}", "body": data}


def _env_wants_gui() -> bool:
    return str(os.environ.get("HAROMA_CHAT_BUDDY_GUI", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _env_rejects_gui() -> bool:
    return str(os.environ.get("HAROMA_CHAT_BUDDY_GUI", "") or "").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
        "console",
        "tty",
    )


def _should_attempt_gui(args: argparse.Namespace) -> bool:
    """True only when Tk is explicitly requested (--gui or HAROMA_CHAT_BUDDY_GUI=1)."""
    if getattr(args, "no_gui", False):
        return False
    if _env_rejects_gui():
        return False
    return bool(args.gui) or _env_wants_gui()


def _explicit_gui_required(args: argparse.Namespace) -> bool:
    """If True and GUI fails, exit non-zero (no console fallback)."""
    return bool(args.gui) or _env_wants_gui()


def _terminal_prefers_ascii() -> bool:
    """Heuristic: piped output and non-UTF-8 console encodings → safer ASCII art."""
    if not sys.stdout.isatty():
        return True
    loc = str(os.environ.get("LC_ALL") or os.environ.get("LANG") or "")
    if "utf-8" in loc.lower() or "utf8" in loc.lower():
        return False
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    if enc in ("utf-8", "utf8"):
        return False
    if enc in ("ascii", "cp437", "cp1252", "latin-1", "iso8859-1", "iso-8859-1"):
        return True
    return "utf" not in enc


def _effective_ascii(args: argparse.Namespace) -> bool:
    if getattr(args, "unicode", False):
        return False
    if args.ascii:
        return True
    if str(os.environ.get("HAROMA_BUDDY_ASCII", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return True
    if str(os.environ.get("HAROMA_BUDDY_UNICODE", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    return _terminal_prefers_ascii()


def run_gui(args: argparse.Namespace, *, quiet_fail: bool = False) -> int:
    """Small Tk window: buddy panel + transcript + input (non-blocking HTTP in a thread)."""
    try:
        import tkinter as tk
        from tkinter import scrolledtext, ttk
    except ImportError as e:
        if not quiet_fail:
            print(
                "tkinter is not available (install python3-tk on Debian/Ubuntu, or use Tcl/Tk on macOS).",
                file=sys.stderr,
            )
            print(f"  ({e})", file=sys.stderr)
        return 1

    base = (args.url or _base_url()).rstrip("/")
    use_color = False
    ascii_only = _effective_ascii(args)
    frame = [0]
    emotion_history: List[str] = []

    try:
        root = tk.Tk()
    except tk.TclError as e:
        if not quiet_fail:
            print(
                "Cannot open a Tk window (no display?). Use a desktop session, "
                "SSH -X, or run with --no-gui for terminal-only mode.",
                file=sys.stderr,
            )
            print(f"  ({e})", file=sys.stderr)
        return 1

    root.title(f"Haroma buddy — {base}")
    root.minsize(520, 480)

    status = ttk.Label(root, text="Checking server…")
    status.pack(fill=tk.X, padx=8, pady=4)

    ok, ping_msg, ping_ms = ping_server(base)
    if ok:
        status.config(text=f"Server OK ({ping_ms:.0f} ms /status)")
    else:
        status.config(text=f"Server not reachable: {ping_msg}")

    buddy_frm = ttk.LabelFrame(root, text="Buddy")
    buddy_frm.pack(fill=tk.BOTH, expand=False, padx=8, pady=4)
    mono = ("Consolas", "DejaVu Sans Mono", "Courier New", "monospace")
    buddy_txt = tk.Text(
        buddy_frm,
        height=14,
        width=44,
        font=(mono[0], 10),
        state=tk.DISABLED,
        wrap=tk.NONE,
    )
    buddy_txt.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    log = scrolledtext.ScrolledText(root, height=12, wrap=tk.WORD, font=("Segoe UI", 10))
    log.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
    log.insert(tk.END, "Type a message and press Send. /help /status /quit\n\n")
    log.config(state=tk.DISABLED)

    ent_frm = ttk.Frame(root)
    ent_frm.pack(fill=tk.X, padx=8, pady=(0, 8))
    entry = ttk.Entry(ent_frm)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
    send_btn = ttk.Button(ent_frm, text="Send")
    send_btn.pack(side=tk.RIGHT)

    result_q: queue.Queue = queue.Queue()
    busy = [False]

    def _append_log(s: str) -> None:
        log.config(state=tk.NORMAL)
        log.insert(tk.END, s)
        log.see(tk.END)
        log.config(state=tk.DISABLED)

    def _set_buddy_art(text: str) -> None:
        buddy_txt.config(state=tk.NORMAL)
        buddy_txt.delete("1.0", tk.END)
        buddy_txt.insert(tk.END, text)
        buddy_txt.config(state=tk.DISABLED)

    def _apply_response(data: Dict[str, Any]) -> None:
        if data.get("error"):
            _append_log(f"[error] {data.get('error')}\n")
            if data.get("body") is not None:
                _append_log(json.dumps(data.get("body"), indent=2)[:1200] + "\n")
            return
        affect = data.get("affect") if isinstance(data.get("affect"), dict) else {}
        strategy = str(data.get("strategy") or "")
        cycle = data.get("cycle", "—")
        reply = str(data.get("response") or "").strip()
        dom = str(affect.get("dominant_emotion") or "neutral")
        emotion_history.append(dom)
        if len(emotion_history) > 32:
            emotion_history[:] = emotion_history[-32:]
        persona_name = str(
            data.get("persona_name") or data.get("persona") or ""
        ).strip()
        llm_ctx = data.get("llm_context") if isinstance(data.get("llm_context"), dict) else {}
        lat = 0.0
        try:
            lat = float(llm_ctx.get("latency_ms") or 0.0)
        except (TypeError, ValueError):
            pass
        frame[0] += 1
        art = render_buddy(
            affect,
            strategy,
            cycle,
            frame=frame[0],
            use_color=use_color,
            ascii_only=ascii_only,
            persona_name=persona_name,
            emotion_history=emotion_history,
            llm_latency_ms=lat,
        )
        _set_buddy_art(art)
        term_w = max(40, min(100, root.winfo_screenwidth() // 8 or 72))
        _append_log("\nElarion:\n" + _wrap_reply(reply, term_w) + "\n\n")

    def _worker(line: str) -> None:
        try:
            if args.async_chat:
                data = chat_async_poll(base, line, depth=args.depth)
            else:
                data = chat_sync(base, line, depth=args.depth)
            result_q.put(("chat", data))
        except Exception as e:
            result_q.put(("err", str(e)))

    def _send() -> None:
        if busy[0]:
            return
        line = entry.get().strip()
        if not line:
            return
        low = line.lower()
        if low in ("/q", "/quit", "/exit"):
            root.destroy()
            return
        if low == "/help":
            _append_log(
                "  /help /quit  ·  Server must be running.\n"
                "  Use --async-chat if first reply is very slow on CPU.\n\n"
            )
            entry.delete(0, tk.END)
            return
        if low == "/status":
            ok2, msg, ms = ping_server(base)
            _append_log(f"  /status: {'ok' if ok2 else msg} ({ms:.0f} ms)\n\n")
            entry.delete(0, tk.END)
            return

        entry.delete(0, tk.END)
        _append_log(f"You: {line}\n")
        busy[0] = True
        send_btn.config(state=tk.DISABLED)
        status.config(text="Waiting for Elarion…")
        threading.Thread(target=_worker, args=(line,), daemon=True).start()

    def _poll_queue() -> None:
        try:
            while True:
                kind, payload = result_q.get_nowait()
                if kind == "err":
                    _append_log(f"[error] {payload}\n\n")
                elif kind == "chat":
                    _apply_response(payload if isinstance(payload, dict) else {})
                busy[0] = False
                send_btn.config(state=tk.NORMAL)
                ok3, _, ms3 = ping_server(base, timeout=3.0)
                status.config(
                    text=f"Ready ({ms3:.0f} ms /status)" if ok3 else "Server issue"
                )
        except queue.Empty:
            pass
        root.after(120, _poll_queue)

    send_btn.config(command=_send)
    entry.bind("<Return>", lambda e: _send())
    root.after(100, _poll_queue)

    def _on_close() -> None:
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    entry.focus_set()
    root.mainloop()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Optional terminal chat buddy. Main UI is the web app at the server URL (browser)."
    )
    ap.add_argument("--url", default=None, help="Base URL (default HAROMA_URL or http://127.0.0.1:8193)")
    ap.add_argument(
        "--gui",
        action="store_true",
        help="Open optional Tk window (default: off; use the web chatbox in a browser instead)",
    )
    ap.add_argument(
        "--no-gui",
        action="store_true",
        help="Never open Tk (default). Use if HAROMA_CHAT_BUDDY_GUI=1 is set but you want terminal only",
    )
    ap.add_argument("--async-chat", action="store_true", help="Use async /chat + poll (for long LLM loads)")
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    art = ap.add_mutually_exclusive_group()
    art.add_argument(
        "--ascii",
        action="store_true",
        help="Force ASCII-only buddy art (overrides auto-detect)",
    )
    art.add_argument(
        "--unicode",
        action="store_true",
        help="Force Unicode/box-drawing buddy art (overrides auto-detect)",
    )
    ap.add_argument("--no-spinner", action="store_true", help="Disable waiting spinner on stderr")
    ap.add_argument("--width", type=int, default=0, help="Wrap reply text (0 = terminal width)")
    ap.add_argument("--depth", default="normal", choices=("normal",), help="Chat depth (same as HTTP API)")
    args = ap.parse_args()

    if _should_attempt_gui(args):
        rc = run_gui(args, quiet_fail=not _explicit_gui_required(args))
        if rc == 0:
            return 0
        if _explicit_gui_required(args):
            return rc
        print(
            "[haroma_chat_buddy] Tk GUI unavailable — using terminal buddy.",
            file=sys.stderr,
            flush=True,
        )

    base = (args.url or _base_url()).rstrip("/")
    use_color = sys.stdout.isatty() and not args.no_color
    ascii_only = _effective_ascii(args)
    frame = 0
    emotion_history: List[str] = []

    term_w = shutil.get_terminal_size((88, 24)).columns
    wrap_w = int(args.width) if args.width and args.width > 0 else max(40, term_w - 4)

    print(
        _ansi("1;35", "Haroma console buddy — messages, /help, empty line or Ctrl+C to quit.", use_color),
        flush=True,
    )
    print(
        _ansi("90", f"Web chat UI (browser): {base}/", use_color),
        flush=True,
    )
    ok, ping_msg, ping_ms = ping_server(base)
    if ok:
        print(
            _ansi("32", f"Server: {base}  ({ping_ms:.0f} ms /status)", use_color),
            flush=True,
        )
    else:
        print(
            _ansi("33", f"Server: {base}  (not reachable: {ping_msg} — start main.py first?)", use_color),
            flush=True,
        )
    print(flush=True)

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0
        if not line:
            print("Bye.")
            return 0

        low = line.lower()
        if low in ("/q", "/quit", "/exit"):
            print("Bye.")
            return 0
        if low == "/help":
            print(
                "  /help /quit  ·  empty line exits\n"
                "  Server must be running (python main.py). "
                "Use --async-chat if the first reply takes minutes on CPU.\n",
                flush=True,
            )
            continue
        if low == "/status":
            ok2, msg, ms = ping_server(base)
            print(f"  /status: {'ok' if ok2 else msg} ({ms:.0f} ms)\n", flush=True)
            continue

        frame += 1
        spinner: Optional[_Spinner] = None
        if not args.no_spinner and sys.stderr.isatty():
            spinner = _Spinner("Elarion is thinking…", ascii_only=ascii_only, use_color=use_color)
            spinner.start()

        try:
            if args.async_chat:
                data = chat_async_poll(base, line, depth=args.depth)
            else:
                data = chat_sync(base, line, depth=args.depth)
        finally:
            if spinner:
                spinner.stop()

        if data.get("error"):
            print(_ansi("31", f"[error] {data.get('error')}", use_color), flush=True)
            if data.get("body") is not None:
                print(json.dumps(data.get("body"), indent=2)[:800], flush=True)
            continue

        affect = data.get("affect") if isinstance(data.get("affect"), dict) else {}
        strategy = str(data.get("strategy") or "")
        cycle = data.get("cycle", "—")
        reply = str(data.get("response") or "").strip()
        dom = str(affect.get("dominant_emotion") or "neutral")
        emotion_history.append(dom)
        if len(emotion_history) > 32:
            emotion_history[:] = emotion_history[-32:]

        persona_name = str(
            data.get("persona_name") or data.get("persona") or ""
        ).strip()
        llm_ctx = data.get("llm_context") if isinstance(data.get("llm_context"), dict) else {}
        lat = 0.0
        try:
            lat = float(llm_ctx.get("latency_ms") or 0.0)
        except (TypeError, ValueError):
            pass

        print(
            render_buddy(
                affect,
                strategy,
                cycle,
                frame=frame,
                use_color=use_color,
                ascii_only=ascii_only,
                persona_name=persona_name,
                emotion_history=emotion_history,
                llm_latency_ms=lat,
            ),
            flush=True,
        )
        print(_ansi("1;37", "Elarion:", use_color), flush=True)
        print(_wrap_reply(reply, wrap_w), flush=True)
        print(flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
