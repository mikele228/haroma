"""GGUF auto-selection prefers smaller files when quant rank ties."""

import os

from engine.LLMBackend import _find_gguf
from engine.ResourceAdaptiveConfig import _pick_auto_gguf


def test_find_gguf_prefers_smallest(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "big.gguf").write_bytes(b"x" * 2000)
    (d / "small.gguf").write_bytes(b"x" * 100)
    chosen = _find_gguf(str(d))
    assert chosen.endswith("small.gguf")


def test_pick_auto_gguf_prefers_smaller_disk_on_same_quant():
    models = [
        {"name": "qwen-3b-instruct-q4_k_m.gguf", "path": "/a.gguf", "size_gb": 2.0},
        {"name": "qwen-1.5b-instruct-q4_k_m.gguf", "path": "/b.gguf", "size_gb": 1.0},
    ]
    best = _pick_auto_gguf(models)
    assert best["name"] == "qwen-1.5b-instruct-q4_k_m.gguf"


def test_pick_auto_gguf_prefers_gemma4_when_present():
    models = [
        {"name": "qwen2.5-1.5b-instruct-q4_k_m.gguf", "path": "/q.gguf", "size_gb": 1.0},
        {"name": "gemma-4-e2b-it-Q4_K_M.gguf", "path": "/g.gguf", "size_gb": 1.2},
    ]
    best = _pick_auto_gguf(models)
    assert best["name"] == "gemma-4-e2b-it-Q4_K_M.gguf"
