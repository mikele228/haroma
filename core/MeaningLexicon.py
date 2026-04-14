"""
Taught word meanings — explicit term→gloss pairs the user (or /teach) registers.

Unlike LearnedLexicon (emotion polarity), this stores human-readable definitions
so cycles can ground language in what you said a word *means*.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, List


class MeaningLexicon:
    """Thread-safe term → definition store. Keys normalized to lowercase."""

    def __init__(self, max_terms: int = 8000):
        self._max_terms = max_terms
        self._lock = threading.RLock()
        self._terms: Dict[str, str] = {}

    def register(self, term: str, meaning: str) -> None:
        t = (term or "").strip().lower()
        m = (meaning or "").strip()
        if not t or not m:
            return
        with self._lock:
            if t not in self._terms and len(self._terms) >= self._max_terms:
                return
            self._terms[t] = m[:2000]

    def get(self, term: str) -> str:
        with self._lock:
            return self._terms.get((term or "").strip().lower(), "")

    def match_in_text(self, text: str, max_hits: int = 5) -> List[Dict[str, str]]:
        """Return up to *max_hits* {term, meaning} for terms appearing as whole words."""
        if not text or not self._terms:
            return []
        with self._lock:
            snapshot = list(self._terms.items())
        # Longest terms first so e.g. "machine learning" wins over "learning"
        snapshot.sort(key=lambda x: len(x[0]), reverse=True)
        hits: List[Dict[str, str]] = []
        seen_terms = set()
        low = text.lower()
        for term, meaning in snapshot:
            if term in seen_terms:
                continue
            try:
                pat = re.compile(r"\b" + re.escape(term) + r"\b", re.I)
            except re.error:
                continue
            if pat.search(low):
                hits.append({"term": term, "meaning": meaning})
                seen_terms.add(term)
            if len(hits) >= max_hits:
                break
        return hits

    def __len__(self) -> int:
        with self._lock:
            return len(self._terms)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {"terms": dict(self._terms)}

    def from_dict(self, data: Dict[str, Any]) -> None:
        if not data:
            return
        raw = data.get("terms") if isinstance(data, dict) else None
        if not isinstance(raw, dict):
            return
        with self._lock:
            self._terms.clear()
            for k, v in raw.items():
                if isinstance(k, str) and isinstance(v, str):
                    kk = k.strip().lower()
                    if kk:
                        self._terms[kk] = v[:2000]
                if len(self._terms) >= self._max_terms:
                    break
