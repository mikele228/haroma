"""
HomeostaticDrives — internal needs for HaromaX6 (Phase 3 → Phase 11).

Five drives that rise and fall cyclically, creating genuine
motivational states beyond curiosity:

  Understanding  — need to reduce prediction error
  Coherence      — need for stable identity
  Expression     — need to act and produce outcomes
  Rest           — need to consolidate and recover
  Connection     — need for external input / interaction

Phase 11 upgrade: Drive dynamics (rise_rate, decay_rate,
urgency_threshold) are no longer fixed.  An EMA-based adaptation
system tracks which drive satisfactions correlate with good outcomes
and slowly adjusts parameters to optimize motivational balance.
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class Drive:
    name: str
    level: float = 0.0
    decay_rate: float = 0.05
    rise_rate: float = 0.03
    urgency_threshold: float = 0.7

    base_decay_rate: float = 0.0
    base_rise_rate: float = 0.0
    base_urgency_threshold: float = 0.0

    def __post_init__(self):
        if self.base_decay_rate == 0.0:
            self.base_decay_rate = self.decay_rate
        if self.base_rise_rate == 0.0:
            self.base_rise_rate = self.rise_rate
        if self.base_urgency_threshold == 0.0:
            self.base_urgency_threshold = self.urgency_threshold

    @property
    def is_urgent(self) -> bool:
        return self.level >= self.urgency_threshold

    def rise(self, amount: float = 0.0):
        self.level = min(1.0, self.level + (amount or self.rise_rate))

    def decay(self, amount: float = 0.0):
        self.level = max(0.0, self.level - (amount or self.decay_rate))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "level": round(self.level, 3),
            "urgent": self.is_urgent,
            "rise_rate": round(self.rise_rate, 4),
            "decay_rate": round(self.decay_rate, 4),
            "urgency_threshold": round(self.urgency_threshold, 3),
        }


class DriveAdaptation:
    """Tracks outcome correlation per-drive and adapts parameters."""

    _EMA_ALPHA = 0.05
    _ADJUSTMENT_RATE = 0.002
    _MIN_SAMPLES = 20

    def __init__(self):
        self._satisfaction_outcomes: Dict[str, List[float]] = {}
        self._deprivation_outcomes: Dict[str, List[float]] = {}
        self._adaptation_steps = 0
        self._history_cap = 200

    def record(self, drive_name: str, was_satisfied: bool, outcome_score: float):
        if was_satisfied:
            buf = self._satisfaction_outcomes.setdefault(drive_name, [])
            buf.append(outcome_score)
            if len(buf) > self._history_cap:
                self._satisfaction_outcomes[drive_name] = buf[-self._history_cap :]
        else:
            buf = self._deprivation_outcomes.setdefault(drive_name, [])
            buf.append(outcome_score)
            if len(buf) > self._history_cap:
                self._deprivation_outcomes[drive_name] = buf[-self._history_cap :]

    def adapt(self, drive: Drive):
        sat = self._satisfaction_outcomes.get(drive.name, [])
        dep = self._deprivation_outcomes.get(drive.name, [])

        if len(sat) < self._MIN_SAMPLES or len(dep) < self._MIN_SAMPLES:
            return

        avg_sat = sum(sat[-50:]) / len(sat[-50:])
        avg_dep = sum(dep[-50:]) / len(dep[-50:])

        if avg_sat > avg_dep + 0.05:
            drive.decay_rate = min(
                drive.base_decay_rate * 1.5, drive.decay_rate + self._ADJUSTMENT_RATE
            )
            drive.urgency_threshold = max(0.5, drive.urgency_threshold - self._ADJUSTMENT_RATE)
        elif avg_dep > avg_sat + 0.05:
            drive.decay_rate = max(
                drive.base_decay_rate * 0.5, drive.decay_rate - self._ADJUSTMENT_RATE
            )
            drive.urgency_threshold = min(0.9, drive.urgency_threshold + self._ADJUSTMENT_RATE)

        if avg_sat > 0.6:
            drive.rise_rate = min(
                drive.base_rise_rate * 1.5, drive.rise_rate + self._ADJUSTMENT_RATE * 0.5
            )
        elif avg_sat < 0.4:
            drive.rise_rate = max(
                drive.base_rise_rate * 0.5, drive.rise_rate - self._ADJUSTMENT_RATE * 0.5
            )

    def stats(self) -> Dict[str, Any]:
        return {
            "adaptation_steps": self._adaptation_steps,
            "tracked_drives": list(self._satisfaction_outcomes.keys()),
            "satisfaction_samples": {k: len(v) for k, v in self._satisfaction_outcomes.items()},
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "satisfaction_outcomes": self._satisfaction_outcomes,
            "deprivation_outcomes": self._deprivation_outcomes,
            "adaptation_steps": self._adaptation_steps,
        }

    def from_dict(self, data: Dict[str, Any]):
        self._satisfaction_outcomes = data.get("satisfaction_outcomes", {})
        self._deprivation_outcomes = data.get("deprivation_outcomes", {})
        self._adaptation_steps = data.get("adaptation_steps", 0)


class HomeostaticSystem:
    def __init__(self):
        self.drives: List[Drive] = [
            Drive("understanding", rise_rate=0.04, decay_rate=0.06, urgency_threshold=0.7),
            Drive("coherence", rise_rate=0.03, decay_rate=0.05, urgency_threshold=0.65),
            Drive("expression", rise_rate=0.03, decay_rate=0.07, urgency_threshold=0.7),
            Drive("rest", rise_rate=0.02, decay_rate=0.15, urgency_threshold=0.8),
            Drive("connection", rise_rate=0.04, decay_rate=0.10, urgency_threshold=0.7),
        ]
        self._drive_map = {d.name: d for d in self.drives}
        self.adaptation = DriveAdaptation()

    def get(self, name: str) -> Drive:
        return self._drive_map[name]

    def update(
        self,
        episode_payload: Dict[str, Any],
        outcome: Dict[str, Any],
        is_dream_cycle: bool = False,
        has_external_input: bool = False,
    ) -> Dict[str, Any]:

        pred_error = episode_payload.get("curiosity", {}).get("uncertainty_score", 0.5)
        drift = episode_payload.get("drift_score", 0.0)
        action_score = outcome.get("score", 0.5)
        action_type = episode_payload.get("action", {}).get("action_type", "")

        understanding = self.get("understanding")
        if pred_error > 0.5:
            understanding.rise(pred_error * 0.06)
        else:
            understanding.decay(0.04)

        coherence = self.get("coherence")
        if drift > 0.2:
            coherence.rise(drift * 0.08)
        else:
            coherence.decay(0.04)

        expression = self.get("expression")
        if action_type and action_score > 0.5:
            expression.decay(0.10)
        else:
            expression.rise(0.03)

        rest = self.get("rest")
        if is_dream_cycle:
            rest.decay(0.20)
        else:
            rest.rise(0.02)

        connection = self.get("connection")
        if has_external_input:
            connection.decay(0.15)
        else:
            connection.rise(0.04)

        score = outcome.get("score", 0.5)
        for drive in self.drives:
            was_satisfied = not drive.is_urgent
            self.adaptation.record(drive.name, was_satisfied, score)

        self.adaptation._adaptation_steps += 1
        if self.adaptation._adaptation_steps % 10 == 0:
            for drive in self.drives:
                self.adaptation.adapt(drive)

        dominant = max(self.drives, key=lambda d: d.level)
        total_urgency = sum(1 for d in self.drives if d.is_urgent)

        return {
            "drives": {d.name: round(d.level, 3) for d in self.drives},
            "dominant_drive": dominant.name,
            "dominant_level": round(dominant.level, 3),
            "urgency": total_urgency / len(self.drives),
            "urgent_drives": [d.name for d in self.drives if d.is_urgent],
        }

    def bias_goals(self, drive_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        goals: List[Dict[str, Any]] = []
        urgent = drive_state.get("urgent_drives", [])

        if "rest" in urgent:
            goals.append(
                {
                    "goal_id": "drive_rest",
                    "description": "Initiate dream consolidation -- need rest",
                    "priority": self.get("rest").level,
                    "source": "homeostasis",
                }
            )

        if "understanding" in urgent:
            goals.append(
                {
                    "goal_id": "drive_understanding",
                    "description": "Seek explanatory input -- need understanding",
                    "priority": self.get("understanding").level,
                    "source": "homeostasis",
                }
            )

        if "coherence" in urgent:
            goals.append(
                {
                    "goal_id": "drive_coherence",
                    "description": "Reflect on identity to restore coherence",
                    "priority": self.get("coherence").level,
                    "source": "homeostasis",
                }
            )

        if "expression" in urgent:
            goals.append(
                {
                    "goal_id": "drive_expression",
                    "description": "Generate a meaningful response -- need expression",
                    "priority": self.get("expression").level,
                    "source": "homeostasis",
                }
            )

        if "connection" in urgent:
            goals.append(
                {
                    "goal_id": "drive_connection",
                    "description": "Seek external input -- need connection",
                    "priority": self.get("connection").level,
                    "source": "homeostasis",
                }
            )

        return goals

    def should_dream(self) -> bool:
        return self.get("rest").is_urgent

    def summarize(self) -> Dict[str, Any]:
        dominant = max(self.drives, key=lambda d: d.level)
        return {
            "drives": {d.name: round(d.level, 3) for d in self.drives},
            "dominant": dominant.name,
            "urgent": [d.name for d in self.drives if d.is_urgent],
            "adaptation": self.adaptation.stats(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drives": {
                d.name: {
                    "level": d.level,
                    "rise_rate": d.rise_rate,
                    "decay_rate": d.decay_rate,
                    "urgency_threshold": d.urgency_threshold,
                    "base_rise_rate": d.base_rise_rate,
                    "base_decay_rate": d.base_decay_rate,
                    "base_urgency_threshold": d.base_urgency_threshold,
                }
                for d in self.drives
            },
            "adaptation": self.adaptation.to_dict(),
        }

    def from_dict(self, data: Dict[str, Any]):
        drive_data = data.get("drives", {})
        for d in self.drives:
            if d.name in drive_data:
                dd = drive_data[d.name]
                d.level = dd.get("level", d.level)
                d.rise_rate = dd.get("rise_rate", d.rise_rate)
                d.decay_rate = dd.get("decay_rate", d.decay_rate)
                d.urgency_threshold = dd.get("urgency_threshold", d.urgency_threshold)
                d.base_rise_rate = dd.get("base_rise_rate", d.base_rise_rate)
                d.base_decay_rate = dd.get("base_decay_rate", d.base_decay_rate)
                d.base_urgency_threshold = dd.get(
                    "base_urgency_threshold", d.base_urgency_threshold
                )

        adapt_data = data.get("adaptation")
        if adapt_data:
            self.adaptation.from_dict(adapt_data)
