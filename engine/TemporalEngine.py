"""
TemporalEngine — temporal binding and sequence awareness for HaromaX6.

Gives the system a sense of time: how long the session has lasted,
what sequence of events occurred, whether patterns are repeating,
and an arc summary of recent experience.

Predictive layer (Markov on logged summaries): compares the current dominant
emotion to the distribution implied by the last timeline step, and surfaces
a coarse counterfactual when an alternative action type would have shifted
the likely next emotion.
"""

from typing import Dict, Any, List, Optional
from collections import Counter
import math
import time


class TemporalEngine:
    def __init__(self, max_timeline: int = 100):
        self.episode_timeline: List[Dict[str, Any]] = []
        self.max_timeline = max_timeline
        self._session_start: float = time.time()
        self._last_external_cycle: int = 0

    def record(self, episode_summary: Dict[str, Any]):
        snapshot = {
            "cycle_id": episode_summary.get("cycle_id", 0),
            "timestamp": time.time(),
            "emotion": episode_summary.get("emotion", "neutral"),
            "action_type": episode_summary.get("action_type", "none"),
            "salience": episode_summary.get("salience", 0.0),
            "dominant_drive": episode_summary.get("dominant_drive", ""),
            "role": episode_summary.get("role", "observer"),
            "drift": episode_summary.get("drift", 0.0),
        }
        self.episode_timeline.append(snapshot)
        if len(self.episode_timeline) > self.max_timeline:
            self.episode_timeline = self.episode_timeline[-self.max_timeline :]

        if snapshot["role"] != "idle":
            self._last_external_cycle = snapshot["cycle_id"]

    def get_recent_sequence(self, n: int = 10) -> List[Dict[str, Any]]:
        return self.episode_timeline[-n:]

    def detect_repetition(self) -> Optional[str]:
        if len(self.episode_timeline) < 4:
            return None

        recent = self.episode_timeline[-6:]
        emotions = [s["emotion"] for s in recent]
        actions = [s["action_type"] for s in recent]

        emo_count = Counter(emotions)
        most_common_emo, emo_freq = emo_count.most_common(1)[0]
        if emo_freq >= 4:
            return (
                f"Emotional repetition: '{most_common_emo}' "
                f"for {emo_freq} of the last {len(recent)} cycles"
            )

        action_count = Counter(actions)
        most_common_act, act_freq = action_count.most_common(1)[0]
        if act_freq >= 5:
            return (
                f"Action repetition: '{most_common_act}' "
                f"for {act_freq} of the last {len(recent)} cycles"
            )

        if len(recent) >= 4:
            pairs = [(recent[i]["emotion"], recent[i]["action_type"]) for i in range(len(recent))]
            pair_count = Counter(pairs)
            top_pair, pair_freq = pair_count.most_common(1)[0]
            if pair_freq >= 3:
                return (
                    f"Pattern repetition: ({top_pair[0]}, {top_pair[1]}) repeated {pair_freq} times"
                )

        return None

    def compute_session_duration(self) -> float:
        return time.time() - self._session_start

    def summarize_arc(self, n: int = 20) -> str:
        recent = self.episode_timeline[-n:]
        if not recent:
            return "No experience yet."

        if len(recent) < 3:
            return f"Just beginning -- {len(recent)} cycles so far."

        emotions = [s["emotion"] for s in recent]
        thirds = len(emotions) // 3
        if thirds == 0:
            thirds = 1

        early_emo = Counter(emotions[:thirds]).most_common(1)[0][0]
        mid_emo = Counter(emotions[thirds : thirds * 2]).most_common(1)[0][0]
        late_emo = Counter(emotions[thirds * 2 :]).most_common(1)[0][0]

        if early_emo == mid_emo == late_emo:
            arc = f"a sustained sense of {early_emo}"
        elif mid_emo == late_emo:
            arc = f"a shift from {early_emo} to {late_emo}"
        else:
            arc = f"a journey from {early_emo} through {mid_emo} to {late_emo}"

        duration = self.compute_session_duration()
        mins = int(duration // 60)
        secs = int(duration % 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

        return f"Over {len(recent)} cycles ({time_str}), I experienced {arc}."

    def time_since_last_external_input(self, current_cycle: int) -> int:
        if self._last_external_cycle == 0:
            return current_cycle
        return current_cycle - self._last_external_cycle

    def get_rhythm(self) -> Dict[str, Any]:
        if len(self.episode_timeline) < 2:
            return {"avg_cycle_duration": 0.0, "emotion_variety": 0, "action_variety": 0}

        durations = []
        for i in range(1, len(self.episode_timeline)):
            dt = self.episode_timeline[i]["timestamp"] - self.episode_timeline[i - 1]["timestamp"]
            durations.append(dt)

        emotions = set(s["emotion"] for s in self.episode_timeline[-20:])
        actions = set(s["action_type"] for s in self.episode_timeline[-20:])

        return {
            "avg_cycle_duration": round(sum(durations) / len(durations), 2) if durations else 0.0,
            "emotion_variety": len(emotions),
            "action_variety": len(actions),
            "timeline_length": len(self.episode_timeline),
        }

    def _next_emotion_distribution(
        self,
        from_emotion: str,
        from_action: str,
    ) -> Dict[str, float]:
        counts: Counter = Counter()
        for i in range(len(self.episode_timeline) - 1):
            s0 = self.episode_timeline[i]
            if s0["emotion"] == from_emotion and s0["action_type"] == from_action:
                nxt = self.episode_timeline[i + 1]["emotion"]
                counts[nxt] += 1
        total = sum(counts.values())
        if total == 0:
            return {}
        return {e: c / total for e, c in counts.items()}

    def _next_emotion_marginal(self, from_emotion: str) -> Dict[str, float]:
        counts: Counter = Counter()
        for i in range(len(self.episode_timeline) - 1):
            if self.episode_timeline[i]["emotion"] == from_emotion:
                counts[self.episode_timeline[i + 1]["emotion"]] += 1
        total = sum(counts.values())
        if total == 0:
            return {}
        return {e: c / total for e, c in counts.items()}

    def _alternate_actions(self, emotion: str, current_action: str) -> List[str]:
        seen: List[str] = []
        for s in self.episode_timeline[-60:]:
            act = s["action_type"]
            if s["emotion"] == emotion and act != current_action and act not in seen:
                seen.append(act)
        return seen[:6]

    def _counterfactual_hint(
        self,
        emotion: str,
        current_action: str,
        default_predicted: str,
    ) -> Optional[str]:
        base = self._next_emotion_distribution(emotion, current_action)
        if not base:
            base = self._next_emotion_marginal(emotion)
        base_top = max(base, key=base.get) if base else default_predicted
        for alt in self._alternate_actions(emotion, current_action):
            dist = self._next_emotion_distribution(emotion, alt)
            if not dist:
                continue
            alt_top = max(dist, key=dist.get)
            if alt_top != base_top and dist[alt_top] >= 0.34:
                return (
                    f"[counterfactual] after similar affect, "
                    f"`{alt}` tended toward {alt_top} (not {base_top})."
                )
        return None

    def predictive_state(self, actual_emotion: str) -> Dict[str, Any]:
        """Likely next emotion from the last logged step vs *this* step's emotion.

        ``surprise`` is high when the observed emotion was unlikely under the
        empirical transition model (capped 0..1).
        """
        result: Dict[str, Any] = {
            "available": False,
            "predicted_emotion": None,
            "probability": 0.0,
            "surprise": 0.0,
            "counterfactual": None,
            "distribution": {},
        }
        if len(self.episode_timeline) < 2:
            return result

        last = self.episode_timeline[-1]
        dist = self._next_emotion_distribution(last["emotion"], last["action_type"])
        if not dist:
            dist = self._next_emotion_marginal(last["emotion"])
        if not dist:
            return result

        top_e = max(dist, key=dist.get)
        p_obs = float(dist.get(actual_emotion, 0.0))
        # Surprise: complement of probability mass on observed label
        surprise = min(1.0, max(0.0, 1.0 - p_obs))
        ent = 0.0
        for p in dist.values():
            if p > 0:
                ent -= p * math.log(p + 1e-9)
        ent_norm = ent / math.log(len(dist) + 1e-9) if len(dist) > 1 else 0.0

        result["available"] = True
        result["predicted_emotion"] = top_e
        result["probability"] = round(float(dist[top_e]), 3)
        result["surprise"] = round(surprise, 3)
        result["entropy_norm"] = round(float(ent_norm), 3)
        result["distribution"] = {
            k: round(v, 3) for k, v in sorted(dist.items(), key=lambda x: -x[1])[:6]
        }
        if surprise >= 0.35:
            result["counterfactual"] = self._counterfactual_hint(
                last["emotion"], last["action_type"], top_e
            )
        return result

    def get_temporal_position(
        self,
        current_cycle: int,
        actual_emotion: Optional[str] = None,
    ) -> Dict[str, Any]:
        pos: Dict[str, Any] = {
            "session_duration": round(self.compute_session_duration(), 1),
            "cycles_since_input": self.time_since_last_external_input(current_cycle),
            "repetition_detected": self.detect_repetition(),
            "timeline_length": len(self.episode_timeline),
        }
        if actual_emotion:
            pos["world_prediction"] = self.predictive_state(actual_emotion)
        return pos

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeline": self.episode_timeline[-self.max_timeline :],
            "session_start": self._session_start,
            "last_external_cycle": self._last_external_cycle,
            "max_timeline": self.max_timeline,
        }

    def from_dict(self, data: Dict[str, Any]):
        self.max_timeline = data.get("max_timeline", self.max_timeline)
        self.episode_timeline = data.get("timeline", [])
        self._session_start = data.get("session_start", time.time())
        self._last_external_cycle = data.get("last_external_cycle", 0)

    def summarize(self) -> Dict[str, Any]:
        return {
            "timeline_length": len(self.episode_timeline),
            "session_duration": round(self.compute_session_duration(), 1),
            "rhythm": self.get_rhythm(),
            "repetition": self.detect_repetition(),
        }
