#!/usr/bin/env python3
"""Measure local GGUF load time and first / second decode (HaromaX6).

Run from repo root::

    python scripts/benchmark_llm_load_time.py
    python scripts/benchmark_llm_load_time.py --model models/gemma-4-e2b-it-Q8_0.gguf
    python scripts/benchmark_llm_load_time.py --n-ctx 8192 --n-gpu-layers 0
    python scripts/benchmark_llm_load_time.py --n-ctx 8192 --n-gpu-layers -1

**Reading the numbers**

* ``LLMBackend()`` includes **MiniLM (reward/SBERT)** init and **Llama mmap** — not
  mmap alone. Logs like ``[ModelCache] 'sbert-minilm'`` are inside that timer.
* **GPU**: ``--n-gpu-layers -1`` usually cuts load + first decode sharply vs CPU-only
  (``0``). Match production with env ``HAROMA_N_GPU_LAYERS`` or tier ``n_gpu_layers``.
* **Chat** uses a **large packed prompt** (memories, identity, JSON). First real
  ``generate_chat`` is often **much slower** than this tiny ``generate`` probe — keep
  ``HAROMA_LLM_CONTEXT_TIMEOUT_SEC`` above that, not only above this script.

Uses the same ``LLMBackend`` / ``llama_cpp`` path as the live server.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _resolve_model_path(explicit: str | None) -> str | None:
    if explicit:
        p = os.path.abspath(os.path.expanduser(explicit.strip()))
        return p if os.path.isfile(p) else None
    for key in ("HAROMA_LOCAL_GGUF", "ELARION_LOCAL_GGUF"):
        raw = (os.environ.get(key) or "").strip()
        if raw and os.path.isfile(raw):
            return os.path.abspath(raw)
    agents = os.path.join(_ROOT, "soul", "agents.json")
    if os.path.isfile(agents):
        try:
            with open(agents, encoding="utf-8") as f:
                cfg = json.load(f)
            llm = cfg.get("llm") or {}
            rel = (llm.get("local_gguf") or "").strip()
            if rel:
                p = os.path.normpath(os.path.join(_ROOT, rel))
                if os.path.isfile(p):
                    return p
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    from engine.LLMBackend import _find_gguf

    models_dir = os.path.join(_ROOT, "models")
    found = _find_gguf(models_dir)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="Time GGUF load + first decode")
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to .gguf (else env HAROMA_LOCAL_GGUF, soul agents.json, or models/)",
    )
    ap.add_argument(
        "--n-ctx", type=int, default=None, help="Context size (default: from tier or 4096)"
    )
    ap.add_argument(
        "--n-gpu-layers",
        type=int,
        default=None,
        help="llama.cpp GPU layers (default: env HAROMA_N_GPU_LAYERS or 0)",
    )
    ap.add_argument(
        "--warmup-tokens",
        type=int,
        default=8,
        help="Max new tokens for first/second timed decode",
    )
    ap.add_argument(
        "--second-decode",
        action="store_true",
        help="Also time a second decode (should be faster after warmup)",
    )
    args = ap.parse_args()

    path = _resolve_model_path(args.model)
    if not path:
        print(
            "No GGUF found. Set --model, HAROMA_LOCAL_GGUF, or place a .gguf under models/",
            file=sys.stderr,
        )
        return 1

    n_ctx = args.n_ctx
    t_resource = 0.0
    if n_ctx is None:
        try:
            from engine.ResourceAdaptiveConfig import detect_resources

            _tr0 = time.perf_counter()
            rc = detect_resources()
            t_resource = time.perf_counter() - _tr0
            n_ctx = int(rc.llm.get("n_ctx", 4096))
        except Exception:
            n_ctx = 4096

    n_gpu = args.n_gpu_layers
    if n_gpu is None:
        try:
            n_gpu = int(os.environ.get("HAROMA_N_GPU_LAYERS", "0") or "0")
        except ValueError:
            n_gpu = 0

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"File: {path}", flush=True)
    print(f"Size: {size_mb:.1f} MB | n_ctx={n_ctx} n_gpu_layers={n_gpu}", flush=True)
    if t_resource > 0.01:
        print(f"detect_resources(): {t_resource:.2f}s", flush=True)

    _ti0 = time.perf_counter()
    from engine.LLMBackend import LLMBackend

    t_import = time.perf_counter() - _ti0
    print(f"import LLMBackend: {t_import:.2f}s", flush=True)

    # Do not trigger SharedResources warmup; we time explicitly.
    os.environ.setdefault("HAROMA_LLM_WARMUP", "0")

    t0 = time.perf_counter()
    backend = LLMBackend(
        model_path=path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu,
        verbose=False,
    )
    t_load = time.perf_counter() - t0

    if backend.backend_type != "local" or backend._model is None:
        print(
            f"LLMBackend.__init__: {t_load:.2f}s (backend_type={backend.backend_type!r})",
            flush=True,
        )
        print(
            "llama-cpp did not load a local model (missing llama-cpp-python or load error).",
            file=sys.stderr,
        )
        return 2

    print(
        f"LLMBackend.__init__: {t_load:.2f}s  (reward/SBERT + Llama mmap + llama init)",
        flush=True,
    )

    mt = max(1, min(64, int(args.warmup_tokens)))
    prompt = "USER: .\nASSISTANT:"

    t1 = time.perf_counter()
    r1 = backend.generate(prompt, max_tokens=mt, temperature=0.0, top_p=1.0)
    t_dec1 = time.perf_counter() - t1
    print(f"First decode ({mt} tok max): {t_dec1:.2f}s  | tail: {(r1 or '')[:80]!r}", flush=True)

    if args.second_decode:
        t2 = time.perf_counter()
        r2 = backend.generate(prompt, max_tokens=mt, temperature=0.0, top_p=1.0)
        t_dec2 = time.perf_counter() - t2
        print(
            f"Second decode ({mt} tok max): {t_dec2:.2f}s  | tail: {(r2 or '')[:80]!r}",
            flush=True,
        )

    total = t_load + t_dec1
    print(f"Total (LLMBackend init + first decode): {total:.2f}s", flush=True)
    print(
        "\nTips:\n"
        "  • HAROMA_LLM_CONTEXT_TIMEOUT_SEC should exceed **packed chat** latency, "
        "not only this tiny prompt (often several× longer).\n"
        "  • If you have a GPU, try HAROMA_N_GPU_LAYERS=-1 (or --n-gpu-layers -1 here) "
        "to match a fast run.\n"
        "  • HF_TOKEN reduces Hub rate-limit warnings when MiniLM downloads.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
