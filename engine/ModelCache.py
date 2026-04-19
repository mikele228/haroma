"""Process-wide model cache to avoid redundant heavy model loads.

Every transformer backbone is loaded once and shared read-only.
Thread-safe: uses a module-level lock.
"""

import threading
import time
from typing import Any, Dict

_lock = threading.Lock()
_cache: Dict[str, Any] = {}
_load_times: Dict[str, float] = {}


def get_or_load(key: str, loader, timeout: float = 5.0):
    """Return cached model or load it (once). Prints timing."""
    with _lock:
        if key in _cache:
            return _cache[key]

    t0 = time.time()
    print(f"  [ModelCache] Loading '{key}'...", flush=True)
    result = loader()
    elapsed = time.time() - t0
    print(f"  [ModelCache] '{key}' loaded in {elapsed:.1f}s", flush=True)

    with _lock:
        if key not in _cache:
            _cache[key] = result
            _load_times[key] = elapsed
        return _cache[key]


def get_sbert():
    """Get shared SentenceTransformer('all-MiniLM-L6-v2')."""

    def _load():
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("all-MiniLM-L6-v2")

    return get_or_load("sbert-minilm", _load)


DEFAULT_SEMANTIC_ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"


def get_hf_semantic_encoder(model_id: str):
    """Shared Hugging Face encoder for :class:`engine.NeuralEncoder` (mean-pooled text embeddings).

    Set env ``HAROMA_SEMANTIC_ENCODER`` to the same model id to pick a different backbone
    (e.g. ``sentence-transformers/all-mpnet-base-v2``). Must be ``AutoModel``-compatible.
    """
    mid = (model_id or DEFAULT_SEMANTIC_ENCODER_ID).strip()
    key = f"hf-semantic:{mid}"

    def _load():
        from transformers import AutoTokenizer, AutoModel

        tok = AutoTokenizer.from_pretrained(mid)
        mdl = AutoModel.from_pretrained(mid)
        mdl.eval()
        return tok, mdl

    return get_or_load(key, _load)


def get_hf_minilm():
    """Legacy alias — same as ``get_hf_semantic_encoder(DEFAULT_SEMANTIC_ENCODER_ID)``."""

    return get_hf_semantic_encoder(DEFAULT_SEMANTIC_ENCODER_ID)


def get_hf_encoder(model_id: str):
    """Get shared Hugging Face **encoder** (BERT family: DistilBERT, MobileBERT, etc.).

    Used by :class:`engine.EmotionEngine` when ``HAROMA_EMOTION_ENCODER`` is set.
    Not a generative LLM — chat still uses GGUF / API backends in :class:`engine.LLMBackend`.
    """
    mid = (model_id or "distilbert-base-uncased").strip()
    key = f"hf-encoder:{mid}"

    def _load():
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(mid)
        mdl = AutoModel.from_pretrained(mid)
        mdl.eval()
        return tok, mdl

    return get_or_load(key, _load)


def get_distilbert():
    """Get shared DistilBERT tokenizer + model (legacy alias)."""

    return get_hf_encoder("distilbert-base-uncased")


def stats() -> Dict[str, Any]:
    with _lock:
        return {"cached_models": list(_cache.keys()), "load_times": dict(_load_times)}
