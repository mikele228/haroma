"""Soul identity lexicon merge for bind-time config (no utterance routing)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.identity_slot_reply import merge_identity_query_lexicon


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_essence() -> dict:
    path = os.path.join(_repo_root(), "soul", "essence.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_overlay() -> dict:
    path = os.path.join(_repo_root(), "config", "identity_slot_overlay.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_merge_identity_query_lexicon_overlay_wins_keys():
    essence = _load_essence()
    overlay = _load_overlay()
    soul_lex = essence.get("identity_query_lexicon") or {}
    merged = merge_identity_query_lexicon(soul_lex, overlay)
    assert "essence_name" in merged
    assert isinstance(merged["essence_name"], dict)
    assert "tokens" in merged["essence_name"]
