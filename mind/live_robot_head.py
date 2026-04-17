"""
Live robot head alongside the HTTP server: GUI popup on Windows / macOS / X11 / Wayland,
otherwise a compact ASCII face on stderr (updates when affect / inflight changes).

Disabled when ``main.py --debug`` (server blocks on main thread; no extra UI).

Override: ``HAROMA_LIVE_ROBOT_HEAD=0`` to disable.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Callable, Dict

from mind.robot_face_presets import ascii_face_line, emotion_intensity_boost, normalize_emotion_label

BootGetter = Callable[[], Any]


def _env_disabled() -> bool:
    return str(os.environ.get("HAROMA_LIVE_ROBOT_HEAD", "") or "").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    )


def _gui_display_available() -> bool:
    if sys.platform == "win32":
        return True
    if sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _read_emotion(shared: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "dominant_emotion": "neutral",
        "intensity": 0.0,
    }
    if shared is None:
        return out
    try:
        em = getattr(shared, "emotion", None)
        if em is not None and hasattr(em, "summarize"):
            s = em.summarize()
            if isinstance(s, dict):
                dom = s.get("dominant_emotion") or s.get("dominant") or s.get("current_emotion")
                if dom:
                    out["dominant_emotion"] = str(dom)
                try:
                    out["intensity"] = float(s.get("intensity", 0.0) or 0.0)
                except (TypeError, ValueError):
                    out["intensity"] = 0.0
    except Exception:
        pass
    return out


def _read_inflight(shared: Any) -> int:
    if shared is None:
        return 0
    try:
        # ``http_chat_inflight`` is a property on SharedResources — do not call it.
        return int(getattr(shared, "http_chat_inflight", 0) or 0)
    except Exception:
        pass
    return 0


def _read_input_pipeline_busy(shared: Any) -> bool:
    if shared is None:
        return False
    try:
        from mind.chat_priority import input_pipeline_busy

        return bool(input_pipeline_busy(shared, None))
    except Exception:
        return _read_inflight(shared) > 0


def _run_tk_loop(get_boot: BootGetter) -> None:
    import tkinter as tk

    root = tk.Tk()
    root.title("Haroma — Elarion")
    root.resizable(False, False)
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    # Avoid tk.StringVar: its __del__ can call Tcl from the wrong thread during GC
    # after mainloop ends (Python 3.10+ on Windows), causing RuntimeError / Tcl_AsyncDelete.
    frm = tk.Frame(root, padx=12, pady=10, bg="#161b22")
    frm.pack(fill="both", expand=True)
    face_lbl = tk.Label(
        frm,
        text=ascii_face_line("neutral", 0, ascii_only=False),
        font=("Segoe UI", 16) if sys.platform == "win32" else ("Menlo", 15),
        fg="#58a6ff",
        bg="#161b22",
        justify="center",
    )
    face_lbl.pack()
    sub_lbl = tk.Label(
        frm,
        text="neutral  ·  intensity=0.00  ·  chat_inflight=0",
        font=("Segoe UI", 10),
        fg="#8b949e",
        bg="#161b22",
    )
    sub_lbl.pack(pady=(6, 0))

    stop = threading.Event()
    frame_ctr = [0]
    after_id: list[int | None] = [None]

    def tick() -> None:
        if stop.is_set():
            return
        try:
            if not root.winfo_exists():
                return
        except tk.TclError:
            return
        frame_ctr[0] += 1
        frame = frame_ctr[0]
        ba = get_boot()
        shared = getattr(ba, "shared", None) if ba is not None else None
        aff = _read_emotion(shared)
        emo = normalize_emotion_label(aff.get("dominant_emotion"))
        try:
            inten = float(aff.get("intensity") or 0.0)
        except (TypeError, ValueError):
            inten = 0.0
        inflight = _read_inflight(shared)
        pip_busy = _read_input_pipeline_busy(shared)
        line = ascii_face_line(emo, frame, ascii_only=False)
        line = emotion_intensity_boost(emo, inten, line)
        sub = f"{emo}  ·  intensity={inten:.2f}  ·  http_inflight={inflight}  ·  input_pipeline_busy={int(pip_busy)}"
        try:
            face_lbl.config(text=line)
            sub_lbl.config(text=sub)
            after_id[0] = root.after(180, tick)
        except tk.TclError:
            return

    tick()

    def on_close() -> None:
        stop.set()
        aid = after_id[0]
        if aid is not None:
            try:
                root.after_cancel(aid)
            except (tk.TclError, ValueError):
                pass
            after_id[0] = None
        try:
            root.destroy()
        except tk.TclError:
            pass

    root.protocol("WM_DELETE_WINDOW", on_close)
    try:
        root.mainloop()
    finally:
        stop.set()
        try:
            root.destroy()
        except tk.TclError:
            pass


def _stderr_ascii_loop(get_boot: BootGetter) -> None:
    ascii_only = str(os.environ.get("HAROMA_LIVE_ROBOT_ASCII", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    frame = 0
    last_state: tuple | None = None
    last_print = 0.0

    while True:
        time.sleep(0.2)
        frame += 1
        ba = get_boot()
        shared = getattr(ba, "shared", None) if ba is not None else None
        aff = _read_emotion(shared)
        emo = normalize_emotion_label(aff.get("dominant_emotion"))
        try:
            inten = float(aff.get("intensity") or 0.0)
        except (TypeError, ValueError):
            inten = 0.0
        inflight = _read_inflight(shared)
        pip_busy = _read_input_pipeline_busy(shared)
        line = ascii_face_line(emo, frame, ascii_only=ascii_only)
        line = emotion_intensity_boost(emo, inten, line)
        state = (emo, round(inten, 2), inflight, pip_busy)
        now = time.time()
        if state == last_state and (now - last_print) < 1.1:
            continue
        last_state = state
        last_print = now
        msg = f"[Elarion robot] {line}  {emo}  http_inflight={inflight} input_pipeline_busy={int(pip_busy)}\n"
        try:
            sys.stderr.write(msg)
            sys.stderr.flush()
        except OSError:
            break


def attach_live_robot_head(*, get_boot: BootGetter, debug: bool) -> None:
    """Start GUI or stderr ASCII robot; no-op if disabled or debug server mode."""
    if debug or _env_disabled():
        return
    if _gui_display_available():
        try:
            import tkinter  # noqa: F401
        except ImportError:
            print(
                "[Elarion-v2] Live robot head: tkinter missing; using stderr ASCII",
                flush=True,
            )
            threading.Thread(target=_stderr_ascii_loop, args=(get_boot,), daemon=True).start()
            return
        print(
            "[Elarion-v2] Live robot head: GUI popup (close window to continue; server keeps running)",
            flush=True,
        )
        try:
            _run_tk_loop(get_boot)
        except Exception as e:
            print(f"[Elarion-v2] Live robot GUI failed ({e}); falling back to stderr ASCII.", flush=True)
            t = threading.Thread(target=_stderr_ascii_loop, args=(get_boot,), daemon=True)
            t.start()
        return
    print("[Elarion-v2] Live robot head: ASCII on stderr (set DISPLAY for GUI)", flush=True)
    t = threading.Thread(target=_stderr_ascii_loop, args=(get_boot,), daemon=True)
    t.start()
