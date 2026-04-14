"""
PersonalityProfile — stable dispositional traits that bias cognition.

Seeded from soul alignment values, nudged slowly by lived experience,
and queried by emotion, action, gating, and appraisal systems to shape
how the agent reacts, decides, and grows.

Trait model (Big-Five inspired + two Elarion-specific):
    openness, conscientiousness, extraversion, agreeableness,
    neuroticism, resilience, assertiveness
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional


class PersonalityProfile:
    TRAITS: List[str] = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
        "resilience",
        "assertiveness",
    ]
    DRIFT_RATE: float = 0.01
    DRIFT_DECAY: float = 0.995

    def __init__(
        self,
        seed: Dict[str, float],
        persona_variation: float = 0.0,
    ):
        self._baseline: Dict[str, float] = {}
        self._traits: Dict[str, float] = {}
        for t in self.TRAITS:
            base = _clamp(seed.get(t, 0.5))
            jitter = random.gauss(0.0, persona_variation) if persona_variation else 0.0
            self._baseline[t] = base
            self._traits[t] = _clamp(base + jitter)

        self._history: List[Dict[str, Any]] = []
        self._nudge_count: int = 0

    def get(self, trait: str) -> float:
        return self._traits.get(trait, 0.5)

    def nudge(self, trait: str, delta: float) -> None:
        if trait not in self._traits:
            return
        capped = max(-self.DRIFT_RATE, min(self.DRIFT_RATE, delta))
        self._traits[trait] = _clamp(self._traits[trait] + capped)
        self._nudge_count += 1

    def decay_toward_baseline(self, strength: float = 0.0) -> None:
        """Mild gravity pull toward soul-seeded baseline."""
        if strength <= 0.0:
            return
        for t in self.TRAITS:
            diff = self._baseline[t] - self._traits[t]
            self._traits[t] = _clamp(self._traits[t] + diff * strength)

    def apply_drift_decay(self) -> None:
        """Slight regression toward 0.5 — prevents runaway extremes."""
        for t in self.TRAITS:
            val = self._traits[t]
            self._traits[t] = _clamp(val + (0.5 - val) * (1.0 - self.DRIFT_DECAY))

    def summarize(self) -> Dict[str, float]:
        return {t: round(self._traits[t], 4) for t in self.TRAITS}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "traits": {t: round(v, 6) for t, v in self._traits.items()},
            "baseline": {t: round(v, 6) for t, v in self._baseline.items()},
            "nudge_count": self._nudge_count,
            "timestamp": time.time(),
        }

    def load_snapshot(self, data: Dict[str, Any]) -> None:
        traits = data.get("traits", {})
        for t in self.TRAITS:
            if t in traits:
                self._traits[t] = _clamp(float(traits[t]))
        baseline = data.get("baseline", {})
        for t in self.TRAITS:
            if t in baseline:
                self._baseline[t] = _clamp(float(baseline[t]))
        self._nudge_count = data.get("nudge_count", 0)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))
