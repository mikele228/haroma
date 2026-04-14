#!/usr/bin/env python3
"""Benchmark GGUF models: load time + chat reply to \"Hi, how are you?\"

Runs one model per subprocess by default so VRAM/RAM is released between loads.

From repo root (HaromaX6)::

    python scripts/benchmark_gguf_models.py --scan
    python scripts/benchmark_gguf_models.py --scan --models-dir models
    python scripts/benchmark_gguf_models.py --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf
    python scripts/benchmark_gguf_models.py --scan --json

Environment: same as production — ``HAROMA_N_GPU_LAYERS``, optional tier via ``n_ctx``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from contextlib import redirect_stdout
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GREETING = "Hi, how are you?"


def _collect_ggufs(directory: str, *, recursive: bool) -> List[str]:
    directory = os.path.abspath(os.path.expanduser(directory))
    if not os.path.isdir(directory):
        return []
    out: List[str] = []
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(directory):
            parts = dirpath.replace("\\", "/").split("/")
            if ".cache" in parts:
                continue
            for fn in filenames:
                if fn.lower().endswith(".gguf"):
                    out.append(os.path.join(dirpath, fn))
    else:
        for fn in sorted(os.listdir(directory)):
            if fn.lower().endswith(".gguf"):
                out.append(os.path.join(directory, fn))
    out.sort(key=lambda p: os.path.basename(p).lower())
    return out


def _build_worker_argv(
    model_path: str,
    base: argparse.Namespace,
) -> List[str]:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--model",
        model_path,
        "--prompt",
        base.prompt,
        "--max-tokens",
        str(base.max_tokens),
        "--temperature",
        str(base.temperature),
    ]
    if base.n_ctx is not None:
        cmd.extend(["--n-ctx", str(base.n_ctx)])
    if base.n_gpu_layers is not None:
        cmd.extend(["--n-gpu-layers", str(base.n_gpu_layers)])
    cmd.append("--json")
    return cmd


def _run_single(model_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Benchmark one GGUF in this process; return a result dict."""
    os.environ.setdefault("HAROMA_LLM_WARMUP", "0")

    path = os.path.abspath(os.path.expanduser(model_path))
    result: Dict[str, Any] = {
        "model_path": path,
        "model_file": os.path.basename(path),
        "size_mb": None,
        "load_s": None,
        "chat_s": None,
        "response_preview": None,
        "ok": False,
        "error": None,
    }

    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    if not os.path.isfile(path):
        result["error"] = "not a file"
        return result

    result["size_mb"] = round(os.path.getsize(path) / (1024 * 1024), 2)

    n_ctx = args.n_ctx
    if n_ctx is None:
        try:
            from engine.ResourceAdaptiveConfig import detect_resources

            rc = detect_resources()
            n_ctx = int(rc.llm.get("n_ctx", 4096))
        except Exception:
            n_ctx = 4096

    n_gpu = args.n_gpu_layers
    if n_gpu is None:
        try:
            n_gpu = int(os.environ.get("HAROMA_N_GPU_LAYERS", "0") or "0")
        except ValueError:
            n_gpu = 0

    try:
        from engine.LLMBackend import LLMBackend

        t0 = time.perf_counter()
        backend = LLMBackend(
            model_path=path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu,
            verbose=False,
        )
        result["load_s"] = round(time.perf_counter() - t0, 3)
    except Exception as exc:
        result["error"] = f"LLMBackend init: {exc}"
        return result

    if backend.backend_type != "local" or not getattr(backend, "_model", None):
        result["error"] = (
            f"no local llama (backend_type={backend.backend_type!r}) — install llama-cpp-python?"
        )
        return result

    messages = [{"role": "user", "content": args.prompt or GREETING}]
    mt = max(1, int(args.max_tokens))
    try:
        t1 = time.perf_counter()
        text = backend.generate_chat(
            messages,
            max_tokens=mt,
            temperature=float(args.temperature),
            top_p=0.9,
        )
        result["chat_s"] = round(time.perf_counter() - t1, 3)
        if text:
            result["response_preview"] = text.strip().replace("\n", " ")[:200]
        else:
            result["response_preview"] = ""
    except Exception as exc:
        result["error"] = f"generate_chat: {exc}"
        return result
    finally:
        try:
            del backend
        except Exception:
            pass
        try:
            import gc

            gc.collect()
        except Exception:
            pass

    result["ok"] = True
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark GGUF load + greeting chat latency",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single .gguf path (for one shot or subprocess worker)",
    )
    ap.add_argument(
        "--scan",
        action="store_true",
        help="Run every .gguf under --models-dir (one subprocess each)",
    )
    ap.add_argument(
        "--models-dir",
        type=str,
        default=os.path.join(_ROOT, "models"),
        help="Directory to scan when using --scan",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Include subfolders (skips paths containing .cache)",
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default=GREETING,
        help="User message for generate_chat",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max new tokens for the assistant reply",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for chat",
    )
    ap.add_argument("--n-ctx", type=int, default=None)
    ap.add_argument("--n-gpu-layers", type=int, default=None)
    ap.add_argument(
        "--json",
        action="store_true",
        help="Print JSON only (one object, or one line per model with --scan)",
    )
    ap.add_argument(
        "--no-subprocess",
        action="store_true",
        help="Load all models in one process (faster to start; may OOM on GPU)",
    )
    args = ap.parse_args()

    if args.scan:
        paths = _collect_ggufs(args.models_dir, recursive=args.recursive)
        if not paths:
            print(
                f"No .gguf files under {args.models_dir!r}",
                file=sys.stderr,
            )
            return 1
        results: List[Dict[str, Any]] = []
        scan_t0 = time.perf_counter()
        for i, p in enumerate(paths):
            if args.no_subprocess:
                r = _run_single(p, args)
                results.append(r)
                if args.json:
                    print(json.dumps(r, ensure_ascii=False))
                continue
            cmd = _build_worker_argv(p, args)
            proc = subprocess.run(
                cmd,
                cwd=_ROOT,
                capture_output=True,
                text=True,
                timeout=None,
            )
            stdout_lines = (proc.stdout or "").strip().splitlines()
            raw = ""
            for line in reversed(stdout_lines):
                s = line.strip()
                if s.startswith("{") and s.endswith("}"):
                    raw = s
                    break
            if proc.returncode != 0 or not raw:
                err = (proc.stderr or "").strip() or f"exit {proc.returncode}"
                results.append(
                    {
                        "model_path": os.path.abspath(p),
                        "model_file": os.path.basename(p),
                        "size_mb": round(os.path.getsize(p) / (1024 * 1024), 2)
                        if os.path.isfile(p)
                        else None,
                        "load_s": None,
                        "chat_s": None,
                        "response_preview": None,
                        "ok": False,
                        "error": err[:500],
                    }
                )
                if args.json:
                    print(json.dumps(results[-1], ensure_ascii=False))
                continue
            try:
                r = json.loads(raw)
            except json.JSONDecodeError:
                r = {
                    "model_file": os.path.basename(p),
                    "ok": False,
                    "error": "invalid worker JSON",
                }
            results.append(r)
            if args.json:
                print(json.dumps(r, ensure_ascii=False))
            elif not args.json:
                print(
                    f"[{i + 1}/{len(paths)}] {r.get('model_file')} …",
                    flush=True,
                )

        scan_total = time.perf_counter() - scan_t0
        if not args.json:
            print(f"\nPrompt: {args.prompt!r}\n", flush=True)
            hdr = f"{'file':<42} {'MB':>8} {'load_s':>8} {'chat_s':>8} ok"
            print(hdr, flush=True)
            print("-" * len(hdr), flush=True)
            for r in results:
                ok = "yes" if r.get("ok") else "no"
                print(
                    f"{str(r.get('model_file', '')):<42} "
                    f"{str(r.get('size_mb', '')):>8} "
                    f"{str(r.get('load_s', '')):>8} "
                    f"{str(r.get('chat_s', '')):>8} {ok}",
                    flush=True,
                )
            print(f"\nWall time (all models): {scan_total:.1f}s", flush=True)
            for r in results:
                if not r.get("ok"):
                    print(
                        f"  FAIL {r.get('model_file')}: {r.get('error')}",
                        file=sys.stderr,
                    )
                elif r.get("response_preview"):
                    prev = r["response_preview"]
                    if len(prev) > 72:
                        prev = prev[:69] + "…"
                    print(f"  • {r.get('model_file')}: {prev!r}", flush=True)
        return 0 if all(r.get("ok") for r in results) else 3

    if not args.model:
        ap.print_help()
        print("\nProvide --model PATH or --scan", file=sys.stderr)
        return 1

    if args.json:
        buf = io.StringIO()
        with redirect_stdout(buf):
            r = _run_single(args.model, args)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    else:
        r = _run_single(args.model, args)
        print(f"File: {r['model_path']}", flush=True)
        print(f"Size: {r.get('size_mb')} MB", flush=True)
        print(f"load_s: {r.get('load_s')}", flush=True)
        print(f"chat_s: {r.get('chat_s')}", flush=True)
        print(f"preview: {r.get('response_preview')!r}", flush=True)
        if r.get("error"):
            print(f"error: {r['error']}", file=sys.stderr)
    return 0 if r.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
