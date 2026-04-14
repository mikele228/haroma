#!/usr/bin/env python3
"""
Prefetch BERT-family checkpoints into the Hugging Face Hub cache.

Uses ``huggingface_hub.snapshot_download`` (files only — no PyTorch load). That avoids
noisy "UNEXPECTED" weight reports you can see when loading MLM checkpoints with bare
``AutoModel`` (extra ``cls.*`` head weights in the file vs encoder-only architecture).

Default set:

* **DistilBERT** — ``distilbert-base-uncased`` (same family as :class:`engine.EmotionEngine` default).
* **MobileBERT** — ``google/mobilebert-uncased``.
* **Tiny / small BERT** — ``gaunernst/bert-tiny-uncased`` (compact encoder; common for TinyBERT-style experiments).

Requires: ``pip install transformers`` (already in project ``requirements.txt``).

Usage::

    python scripts/download_bert_models.py
    python scripts/download_bert_models.py --only distilbert mobilebert
    python scripts/download_bert_models.py --list
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# id, short label
DEFAULT_MODELS: dict[str, tuple[str, str]] = {
    "distilbert": (
        "distilbert-base-uncased",
        "DistilBERT base (EmotionEngine / ModelCache.get_distilbert)",
    ),
    "mobilebert": (
        "google/mobilebert-uncased",
        "MobileBERT (Google)",
    ),
    "bert_tiny": (
        "gaunernst/bert-tiny-uncased",
        "BERT-tiny ~4M params (Turc et al.; TinyBERT-class)",
    ),
}


def _download_one(model_id: str, label: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "Install huggingface_hub: pip install huggingface_hub",
            "(usually pulled in with transformers)",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"  Downloading {label} ({model_id})...")
    snapshot_download(repo_id=model_id)
    print(f"  OK — snapshot in HF cache for {model_id}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Download BERT-family HF models into cache")
    ap.add_argument(
        "--only",
        nargs="+",
        choices=sorted(DEFAULT_MODELS.keys()),
        help="Subset of keys (default: all)",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="Print default model IDs and exit",
    )
    args = ap.parse_args()

    if args.list:
        for k, (mid, desc) in sorted(DEFAULT_MODELS.items()):
            print(f"{k:12}  {mid:40}  # {desc}")
        return 0

    keys = args.only if args.only else list(DEFAULT_MODELS.keys())
    print("Downloading / warming Hugging Face cache for BERT-family models...\n")
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    print(f"(Typical cache root: {hf_home})\n")

    failed = []
    for k in keys:
        mid, desc = DEFAULT_MODELS[k]
        try:
            _download_one(mid, desc)
        except Exception as e:
            print(f"  FAILED {mid}: {e}", flush=True)
            failed.append((mid, str(e)))
        print()

    if failed:
        print(f"{len(failed)} model(s) failed (network, auth, or incompatible env).")
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
