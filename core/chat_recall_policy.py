"""Recall tuning for conversational / teaching-style user turns."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from core.Memory import MemoryForest, MemoryNode


_PHRASES = (
    "teach me",
    "explain ",
    "explain:",
    "why does",
    "why do ",
    "how does",
    "how do ",
    "what is ",
    "what are ",
    "difference between",
    "difference among",
    "learn ",
    "define ",
    "describe ",
    "summarize",
    "in simple terms",
    "intuitively",
)


def is_teaching_turn(text: str, nlu: Dict[str, Any]) -> bool:
    """Detect explanation / pedagogy style prompts (boosts recall + web_learn)."""
    t = (text or "").strip().lower()
    if not t:
        return False
    if any(p in t for p in _PHRASES):
        return True
    return False


def web_learn_inject_mode() -> str:
    """Env ``HAROMA_WEB_LEARN_INJECT_MODE``: ``teaching_only`` | ``factual_heuristic`` | ``always``."""
    m = str(os.environ.get("HAROMA_WEB_LEARN_INJECT_MODE", "teaching_only") or "").strip().lower()
    if m in ("teaching_only", "factual_heuristic", "always"):
        return m
    return "teaching_only"


def web_learn_inject_max() -> int:
    """Env ``HAROMA_WEB_LEARN_INJECT_MAX`` (default ``3``, cap ``12``)."""
    try:
        return max(0, min(12, int(os.environ.get("HAROMA_WEB_LEARN_INJECT_MAX", "3"))))
    except (TypeError, ValueError):
        return 3


def is_factual_heuristic_turn(text: str, nlu: Dict[str, Any]) -> bool:
    """Lightweight factual / lookup cues (question mark, WH-words, NLU entities)."""
    t = (text or "").strip().lower()
    if not t:
        return False
    if "?" in t:
        return True
    hints = (
        "who ",
        "when ",
        "where ",
        "which ",
        "how many ",
        "how much ",
    )
    if any(h in t for h in hints):
        return True
    if isinstance(nlu, dict):
        ents = nlu.get("entities")
        if isinstance(ents, list) and len(ents) > 0:
            return True
        rels = nlu.get("relations")
        if isinstance(rels, list) and len(rels) > 0:
            return True
    return False


def should_merge_web_learn(
    text: str,
    nlu: Dict[str, Any],
    *,
    teaching: bool,
) -> bool:
    """Whether to prepend ``web_learn`` memories for this turn (see ``web_learn_inject_mode``)."""
    mode = web_learn_inject_mode()
    if mode == "always":
        return True
    if mode == "teaching_only":
        return bool(teaching)
    return bool(teaching or is_factual_heuristic_turn(text, nlu))


def merge_web_learn_tail(
    forest: "MemoryForest",
    recalled: List["MemoryNode"],
    recall_limit: int,
    *,
    max_inject: int = 3,
) -> List["MemoryNode"]:
    """Prepend recent ``web_learn`` nodes (deduped) for teaching-style turns."""
    if max_inject <= 0 or recall_limit <= 0:
        return recalled
    try:
        nodes = forest.get_nodes("thought_tree", "web_learn")
        tail = nodes[-max_inject:] if len(nodes) > max_inject else nodes
        if not tail:
            return recalled
        seen = {n.moment_id for n in recalled}
        prefix = [n for n in reversed(tail) if n.moment_id not in seen]
        if not prefix:
            return recalled
        merged: List["MemoryNode"] = list(prefix) + list(recalled)
        return merged[:recall_limit]
    except Exception:
        return recalled
