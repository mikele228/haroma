#!/usr/bin/env python3
"""Print GPU / CPU signals relevant to local Gemma GGUF (HaromaX6).

Run from repo root: python scripts/check_llm_hw_env.py
"""

from __future__ import annotations

import subprocess
import sys


def _main() -> int:
    print("=== HaromaX6 LLM hardware check ===", flush=True)

    cuda_torch = False
    try:
        import torch

        cuda_torch = bool(torch.cuda.is_available())
        print(f"torch.cuda.is_available: {cuda_torch}", flush=True)
        if cuda_torch:
            print(f"  device: {torch.cuda.get_device_name(0)}", flush=True)
    except Exception as exc:
        print(f"torch: not available ({exc})", flush=True)

    try:
        from llama_cpp import llama_cpp

        # llama.cpp build may expose offload support
        sup = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        if callable(sup):
            print(f"llama_cpp.llama_supports_gpu_offload(): {bool(sup())}", flush=True)
    except Exception as exc:
        print(f"llama_cpp GPU probe: {exc}", flush=True)

    try:
        r = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            print("nvidia-smi -L:", flush=True)
            print(r.stdout.strip(), flush=True)
        else:
            print("nvidia-smi: not found or no GPUs listed", flush=True)
    except Exception as exc:
        print(f"nvidia-smi: {exc}", flush=True)

    print("", flush=True)
    print("Recommended env (CPU-only, packed Gemma slow):", flush=True)
    print('  POST /chat with {"depth": "normal"}', flush=True)
    print(
        "  HAROMA_CHAT_TIMEOUT > HAROMA_LLM_CONTEXT_TIMEOUT_SEC (defaults aligned in server)",
        flush=True,
    )
    print("", flush=True)
    if cuda_torch:
        print(
            "CUDA detected: server boot may set HAROMA_FAST_LLM=1 when local GGUF uses GPU layers.",
            flush=True,
        )
    else:
        print(
            "No CUDA via PyTorch: expect CPU inference; use Q4 GGUF or API for faster replies.",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(_main())
