"""Symbolic law store — tag-overlap compliance checks for the cognitive loop.

Each law binds a set of tags. When `check_compliance` is called with active
tags from perception (or explicit `law_tags` in input), any non-empty
intersection with a law's tag set counts as a violation. This matches
\"forbidden pattern\" constraints; extend later for obligation-style rules.

**Sources**
  * ``internal`` — from the soul (essence, identity); non-negotiable self-law.
  * ``external`` — from society (norms, policy, KG, HTTP config); social law.
"""

from __future__ import annotations

from typing import Any, Dict, List

LAW_SOURCE_INTERNAL = "internal"
LAW_SOURCE_EXTERNAL = "external"


class LawEngine:
    def __init__(self) -> None:
        self.laws: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def normalize_source(source: str | None) -> str:
        s = (source or LAW_SOURCE_EXTERNAL).strip().lower()
        if s == LAW_SOURCE_INTERNAL:
            return LAW_SOURCE_INTERNAL
        return LAW_SOURCE_EXTERNAL

    def declare_law(
        self,
        law_id: str,
        description: str,
        tags: List[str],
        severity: float = 1.0,
        source: str = LAW_SOURCE_EXTERNAL,
    ) -> None:
        src = self.normalize_source(source)
        self.laws[law_id] = {
            "id": law_id,
            "description": description,
            "tags": [str(t).lower() for t in (tags or [])],
            "severity": float(severity),
            "source": src,
        }

    def revoke_law(self, law_id: str) -> None:
        self.laws.pop(law_id, None)

    def check_compliance(self, active_tags: List[str]) -> List[Dict[str, Any]]:
        if not self.laws:
            return []
        tag_set = {str(t).lower() for t in (active_tags or []) if t is not None}
        if not tag_set:
            return []
        violations: List[Dict[str, Any]] = []
        for lid, law in self.laws.items():
            forbidden = set(law.get("tags") or [])
            if not forbidden:
                continue
            overlap = tag_set & forbidden
            if overlap:
                violations.append(
                    {
                        "law_id": lid,
                        "description": law.get("description", ""),
                        "matched_tags": sorted(overlap),
                        "severity": law.get("severity", 1.0),
                        "source": law.get("source", LAW_SOURCE_EXTERNAL),
                    }
                )
        return violations

    def get_law(self, law_id: str) -> Dict[str, Any]:
        law = self.laws.get(law_id)
        return dict(law) if law else {}

    def summarize(self) -> Dict[str, Any]:
        internal = 0
        external = 0
        for row in self.laws.values():
            if row.get("source") == LAW_SOURCE_INTERNAL:
                internal += 1
            else:
                external += 1
        return {
            "available": True,
            "count": len(self.laws),
            "ids": list(self.laws.keys()),
            "internal_count": internal,
            "external_count": external,
        }

    def reset(self) -> None:
        self.laws.clear()
