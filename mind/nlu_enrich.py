"""Lightweight NLU enrichment for KnowledgeGraph integration.

When the perception NLU returns few entities, derive simple CONCEPT mentions
from surface text (quoted phrases, capitalization patterns) so ``integrate``
runs more often without replacing the main NLU pipeline.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_STOPWORDS = frozenset(
    {
        "about",
        "above",
        "after",
        "again",
        "along",
        "already",
        "also",
        "always",
        "among",
        "another",
        "around",
        "based",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "bring",
        "built",
        "called",
        "could",
        "doing",
        "don't",
        "during",
        "each",
        "either",
        "enough",
        "every",
        "first",
        "found",
        "given",
        "going",
        "great",
        "happen",
        "having",
        "hello",
        "here",
        "how's",
        "human",
        "isn't",
        "just",
        "known",
        "large",
        "later",
        "learn",
        "least",
        "let's",
        "level",
        "light",
        "long",
        "looks",
        "makes",
        "maybe",
        "means",
        "memory",
        "might",
        "model",
        "month",
        "more",
        "most",
        "much",
        "never",
        "night",
        "often",
        "only",
        "order",
        "other",
        "ought",
        "place",
        "point",
        "pretty",
        "probably",
        "quite",
        "question",
        "rather",
        "reach",
        "ready",
        "really",
        "right",
        "shall",
        "she's",
        "short",
        "since",
        "small",
        "something",
        "sometimes",
        "should",
        "start",
        "state",
        "still",
        "story",
        "stuff",
        "such",
        "taken",
        "thank",
        "thanks",
        "that's",
        "their",
        "them",
        "there",
        "these",
        "they",
        "thing",
        "things",
        "think",
        "those",
        "three",
        "through",
        "times",
        "today",
        "together",
        "total",
        "truly",
        "turns",
        "under",
        "until",
        "using",
        "value",
        "wants",
        "wasn't",
        "watch",
        "water",
        "well",
        "what",
        "what's",
        "when",
        "where",
        "which",
        "while",
        "whole",
        "whose",
        "world",
        "would",
        "write",
        "years",
        "yours",
    }
)


def enrich_nlu_for_kg(nlu: Optional[Dict[str, Any]], raw_text: str) -> Dict[str, Any]:
    """Return a copy of *nlu* with optional extra ``entities`` (cap ~8)."""
    base: Dict[str, Any] = dict(nlu) if nlu else {}
    entities: List[Dict[str, Any]] = list(base.get("entities") or [])
    text = (raw_text or "").strip()
    if len(text) < 4:
        base["entities"] = entities
        return base

    existing_lower = {
        str(e.get("text", "")).lower().strip()
        for e in entities
        if isinstance(e, dict) and e.get("text")
    }

    found: List[str] = []

    for m in re.finditer(r'"([^"]{2,48})"|\'([^\']{2,48})\'', text):
        chunk = (m.group(1) or m.group(2) or "").strip()
        if chunk:
            found.append(chunk)

    for m in re.finditer(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
        chunk = m.group(0).strip()
        if len(chunk) >= 3 and chunk not in found:
            found.append(chunk)

    for token in re.findall(r"\b[A-Za-z]{5,}\b", text.lower()):
        if token in _STOPWORDS:
            continue
        w = token[:1].upper() + token[1:] if token else ""
        if w and w.lower() not in existing_lower and w not in found:
            found.append(w)
        if len(found) >= 12:
            break

    added = 0
    for chunk in found:
        cl = chunk.lower().strip()
        if not cl or cl in existing_lower:
            continue
        entities.append({"text": chunk, "type": "CONCEPT", "role": "enriched"})
        existing_lower.add(cl)
        added += 1
        if added >= 8 or len(entities) >= 24:
            break

    base["entities"] = entities
    return base
