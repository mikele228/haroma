"""
AppraisalEngine — Cognitive appraisal for HaromaX6 (Phase 7 → Phase 11).

Evaluates each cognitive cycle across four Scherer-inspired dimensions:
  Relevance         — does this event matter to active goals?
  Implication        — does it help or hinder those goals?
  Coping Potential   — can the system handle the event?
  Norm Compatibility — does it align with identity and values?

Phase 11 upgrade: Fixed weights and thresholds are replaced by a small
PyTorch MLP that learns optimal dimension weighting from outcome
feedback.  The hard-coded formulas remain as a baseline; the learned
weights blend in via the standard earned-weight ramp.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import random

from utils.module_base import ModuleBase

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except (ImportError, OSError):
    _TORCH = False

from engine.ComputeFabric import get_fabric as _get_fabric


_APPRAISAL_INPUT_DIM = 15
_APPRAISAL_OUTPUT_DIM = 7

if _TORCH:

    class _AppraisalNet(nn.Module):
        """Maps raw context features to appraisal dimension scores.

        Input  (12-d): goal_signal, baseline_emotion, nlu_token_count,
                       polarity, goal_relevant, kg_gain, inter_polarity,
                       coverage, avg_action_score, wm_headroom,
                       coherence, no_drift
        Output (7-d):  relevance, implication, coping, norm_compat,
                       valence, arousal, intensity
        """

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(_APPRAISAL_INPUT_DIM, 24),
                nn.ReLU(),
                nn.Linear(24, 16),
                nn.ReLU(),
                nn.Linear(16, _APPRAISAL_OUTPUT_DIM),
                nn.Tanh(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


class AppraisalEngine(ModuleBase):
    RELEVANCE_THRESHOLD = 0.4
    _RAMP_STEPS = 150
    _MAX_WEIGHT = 0.9

    def __init__(self, history_cap: int = 200):
        super().__init__("AppraisalEngine")
        self._history: List[Dict[str, Any]] = []
        self._history_cap = history_cap

        self.available = _TORCH
        self._train_steps = 0
        self._train_buffer: List[Tuple[List[float], List[float]]] = []
        self._buffer_cap = 256
        self._rolling_accuracy: List[float] = []
        self._accuracy_window = 50

        if _TORCH:
            self._net = _AppraisalNet()
            _fab = _get_fabric()
            if _fab:
                self._net = _fab.register("appraisal_net", self._net)
            self._optim = torch.optim.Adam(self._net.parameters(), lr=5e-4)
        else:
            self._net = None
            self._optim = None

    @property
    def learned_weight(self) -> float:
        progress = min(1.0, self._train_steps / max(self._RAMP_STEPS, 1))
        return progress * self._MAX_WEIGHT

    def evaluate(
        self,
        *,
        nlu_result: Dict[str, Any],
        active_goals: List[Dict[str, Any]],
        knowledge_summary: Dict[str, Any],
        knowledge_diff: Dict[str, Any],
        identity_summary: Dict[str, Any],
        emotion_summary: Dict[str, Any],
        drift_score: float,
        action_memory_stats: Dict[str, Any],
        working_memory_load: float,
        interlocutor: Optional[Dict[str, Any]] = None,
        personality: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:

        raw = self._extract_raw_features(
            nlu_result,
            active_goals,
            knowledge_summary,
            knowledge_diff,
            identity_summary,
            emotion_summary,
            drift_score,
            action_memory_stats,
            working_memory_load,
            interlocutor,
        )
        raw.extend(
            [
                personality.get("assertiveness", 0.5) if personality else 0.5,
                personality.get("neuroticism", 0.5) if personality else 0.5,
                personality.get("agreeableness", 0.5) if personality else 0.5,
            ]
        )

        baseline = self._baseline_evaluate(
            nlu_result,
            active_goals,
            knowledge_summary,
            knowledge_diff,
            identity_summary,
            emotion_summary,
            drift_score,
            action_memory_stats,
            working_memory_load,
            interlocutor,
        )

        learned = self._learned_evaluate(raw)
        result = self._blend(baseline, learned)

        self._history.append(result)
        if len(self._history) > self._history_cap:
            self._history = self._history[-self._history_cap :]

        result["_raw_features"] = raw
        return result

    def _extract_raw_features(
        self,
        nlu,
        goals,
        knowledge_summary,
        knowledge_diff,
        identity_summary,
        emotion_summary,
        drift_score,
        action_memory_stats,
        working_memory_load,
        interlocutor,
    ) -> List[float]:
        entities = {e.get("text", "").lower() for e in nlu.get("entities", []) if e.get("text")}
        relations = nlu.get("relations", [])
        rel_words = set()
        for r in relations:
            for key in ("subject", "predicate", "object"):
                val = r.get(key, "")
                if isinstance(val, str):
                    rel_words.update(val.lower().split())
        nlu_tokens = entities | rel_words

        affected = 0
        priority_sum = 0.0
        for goal in goals:
            desc = goal.get("description", goal.get("goal_id", ""))
            goal_words = set(desc.lower().replace("_", " ").split())
            if nlu_tokens & goal_words:
                affected += 1
                priority_sum += goal.get("priority", 0.5)
        goal_signal = min(1.0, affected / max(len(goals), 1) + priority_sum * 0.1)
        baseline_emotion = emotion_summary.get("intensity", 0.0)

        polarity = nlu.get("sentiment", {}).get("polarity", 0.0)
        goal_relevant = (
            1.0
            if any(
                any(
                    e in goal.get("description", goal.get("goal_id", "")).lower()
                    for e in entities
                    if e
                )
                for goal in goals
            )
            else 0.0
        )
        kg_gain = (
            knowledge_diff.get("knowledge_gain", 0.0) if knowledge_diff.get("changed") else 0.0
        )
        inter_polarity = (interlocutor or {}).get("polarity", 0.0)

        known_count = knowledge_summary.get("entity_count", 0)
        entity_list = [e.get("text", "").lower() for e in nlu.get("entities", []) if e.get("text")]
        if entity_list:
            coverage = min(1.0, known_count / max(len(entity_list) * 3, 1))
        else:
            coverage = min(1.0, known_count * 0.05) if known_count else 0.3
        avg_score = action_memory_stats.get("avg_score", 0.5)
        wm_headroom = 1.0 - working_memory_load

        coherence = identity_summary.get("coherence", 1.0)
        no_drift = 1.0 - min(1.0, drift_score)

        return [
            goal_signal,
            baseline_emotion,
            min(1.0, len(nlu_tokens) * 0.1),
            polarity,
            goal_relevant,
            min(1.0, kg_gain),
            inter_polarity,
            coverage,
            avg_score,
            wm_headroom,
            coherence,
            no_drift,
        ]

    def _learned_evaluate(self, raw: List[float]) -> Optional[Dict[str, Any]]:
        if not self.available or self._net is None:
            return None
        w = self.learned_weight
        if w < 0.01:
            return None

        try:
            with torch.no_grad():
                _fab = _get_fabric()
                x = _fab.tensor([raw]) if _fab else torch.tensor([raw], dtype=torch.float32)
                out = self._net(x).squeeze(0)
        except RuntimeError:
            return None

        relevance = (out[0].item() + 1.0) / 2.0
        implication = out[1].item()
        coping = (out[2].item() + 1.0) / 2.0
        norm_compat = (out[3].item() + 1.0) / 2.0
        valence = out[4].item()
        arousal = (out[5].item() + 1.0) / 2.0
        intensity = (out[6].item() + 1.0) / 2.0

        emotion = self._map_emotion(relevance, implication, coping, norm_compat)

        return {
            "relevance": round(relevance, 3),
            "implication": round(implication, 3),
            "coping": round(coping, 3),
            "norm_compatibility": round(norm_compat, 3),
            "emotion": emotion,
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
            "intensity": round(intensity, 3),
            "overrides": relevance > self.RELEVANCE_THRESHOLD,
        }

    def _blend(self, baseline: Dict[str, Any], learned: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if learned is None:
            return baseline

        w = self.learned_weight
        result = {}
        for key in (
            "relevance",
            "implication",
            "coping",
            "norm_compatibility",
            "valence",
            "arousal",
            "intensity",
        ):
            b = baseline.get(key, 0.0)
            l = learned.get(key, 0.0)
            result[key] = round((1.0 - w) * b + w * l, 3)

        result["emotion"] = learned["emotion"] if w > 0.5 else baseline["emotion"]
        result["overrides"] = result["relevance"] > self.RELEVANCE_THRESHOLD
        result["blend_weight"] = round(w, 3)
        return result

    def record_outcome(
        self,
        raw_features: List[float],
        actual_emotion: str,
        actual_valence: float,
        actual_arousal: float,
        outcome_score: float,
    ):
        if not self.available:
            return

        target_relevance = min(1.0, outcome_score)
        target_implication = actual_valence
        target_coping = min(1.0, max(0.0, outcome_score * 0.8 + 0.2))
        target_norm = min(1.0, max(0.0, outcome_score * 0.5 + 0.5))
        target_valence = actual_valence
        target_arousal = actual_arousal
        target_intensity = min(1.0, abs(actual_valence) * 0.5 + actual_arousal * 0.5)

        targets = [
            target_relevance * 2.0 - 1.0,
            target_implication,
            target_coping * 2.0 - 1.0,
            target_norm * 2.0 - 1.0,
            target_valence,
            target_arousal * 2.0 - 1.0,
            target_intensity * 2.0 - 1.0,
        ]

        self._train_buffer.append((raw_features, targets))
        if len(self._train_buffer) > self._buffer_cap:
            self._train_buffer = self._train_buffer[-self._buffer_cap :]

    def train_step(self) -> Optional[float]:
        if not self.available or not self._train_buffer or self._net is None:
            return None
        if len(self._train_buffer) < 8:
            return None

        batch_size = min(32, len(self._train_buffer))
        batch = random.sample(self._train_buffer, batch_size)

        _fab = _get_fabric()
        features = (
            _fab.tensor([b[0] for b in batch])
            if _fab
            else torch.tensor([b[0] for b in batch], dtype=torch.float32)
        )
        targets = (
            _fab.tensor([b[1] for b in batch])
            if _fab
            else torch.tensor([b[1] for b in batch], dtype=torch.float32)
        )

        predictions = self._net(features)
        loss = nn.functional.mse_loss(predictions, targets)

        self._optim.zero_grad()
        if _fab:
            _fab.scale_loss(loss).backward()
            nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
            _fab.scaler_step(self._optim)
            _fab.scaler_update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
            self._optim.step()

        self._train_steps += 1

        loss_val = loss.item()
        accuracy = 1.0 - min(1.0, loss_val)
        self._rolling_accuracy.append(accuracy)
        if len(self._rolling_accuracy) > self._accuracy_window:
            self._rolling_accuracy = self._rolling_accuracy[-self._accuracy_window :]

        return loss_val

    # ------------------------------------------------------------------
    # Original baseline (Phase 7 formulas, unchanged)
    # ------------------------------------------------------------------

    def _baseline_evaluate(
        self,
        nlu,
        goals,
        knowledge_summary,
        knowledge_diff,
        identity_summary,
        emotion_summary,
        drift_score,
        action_memory_stats,
        working_memory_load,
        interlocutor,
    ) -> Dict[str, Any]:
        relevance = self._compute_relevance(nlu, goals, emotion_summary)
        implication = self._compute_implication(nlu, goals, knowledge_diff, interlocutor)
        coping = self._compute_coping(
            nlu, knowledge_summary, action_memory_stats, working_memory_load
        )
        norm_compat = self._compute_norm_compatibility(identity_summary, drift_score)

        emotion = self._map_emotion(relevance, implication, coping, norm_compat)
        valence = self._compute_valence(implication, norm_compat, coping)
        arousal = self._compute_arousal(relevance, implication, coping)
        intensity = self._compute_intensity(relevance, implication, arousal)

        return {
            "relevance": round(relevance, 3),
            "implication": round(implication, 3),
            "coping": round(coping, 3),
            "norm_compatibility": round(norm_compat, 3),
            "emotion": emotion,
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
            "intensity": round(intensity, 3),
            "overrides": relevance > self.RELEVANCE_THRESHOLD,
        }

    def _compute_relevance(self, nlu, goals, emotion_summary) -> float:
        if not goals:
            baseline = emotion_summary.get("intensity", 0.0) * 0.3
            return min(1.0, baseline)

        entities = {e.get("text", "").lower() for e in nlu.get("entities", []) if e.get("text")}
        relations = nlu.get("relations", [])
        rel_words = set()
        for r in relations:
            for key in ("subject", "predicate", "object"):
                val = r.get(key, "")
                if isinstance(val, str):
                    rel_words.update(val.lower().split())
        nlu_tokens = entities | rel_words

        affected = 0
        priority_sum = 0.0
        for goal in goals:
            desc = goal.get("description", goal.get("goal_id", ""))
            goal_words = set(desc.lower().replace("_", " ").split())
            if nlu_tokens & goal_words:
                affected += 1
                priority_sum += goal.get("priority", 0.5)

        goal_signal = min(1.0, affected / max(len(goals), 1) + priority_sum * 0.1)

        baseline = emotion_summary.get("intensity", 0.0) * 0.2
        return min(1.0, goal_signal * 0.7 + baseline + len(nlu_tokens) * 0.02)

    def _compute_implication(self, nlu, goals, knowledge_diff, interlocutor) -> float:
        sentiment = nlu.get("sentiment", {})
        polarity = sentiment.get("polarity", 0.0)

        entities = {e.get("text", "").lower() for e in nlu.get("entities", []) if e.get("text")}
        goal_relevant = False
        for goal in goals:
            desc = goal.get("description", goal.get("goal_id", ""))
            if any(e in desc.lower() for e in entities if e):
                goal_relevant = True
                break

        base = polarity * 0.5
        if goal_relevant:
            base += polarity * 0.3

        if knowledge_diff.get("changed"):
            kg_gain = knowledge_diff.get("knowledge_gain", 0.0)
            base += min(0.2, kg_gain * 0.15)

        if interlocutor and interlocutor.get("known"):
            inter_polarity = interlocutor.get("polarity", 0.0)
            base += inter_polarity * 0.1

        return max(-1.0, min(1.0, base))

    def _compute_coping(
        self, nlu, knowledge_summary, action_memory_stats, working_memory_load
    ) -> float:
        entities = [e.get("text", "").lower() for e in nlu.get("entities", []) if e.get("text")]
        known_count = knowledge_summary.get("entity_count", 0)

        if entities:
            coverage = min(1.0, known_count / max(len(entities) * 3, 1))
        else:
            coverage = min(1.0, known_count * 0.05) if known_count else 0.3

        avg_score = action_memory_stats.get("avg_score", 0.5)
        wm_headroom = 1.0 - working_memory_load

        return min(1.0, coverage * 0.4 + avg_score * 0.35 + wm_headroom * 0.25)

    def _compute_norm_compatibility(self, identity_summary, drift_score) -> float:
        coherence = identity_summary.get("coherence", 1.0)
        no_drift = 1.0 - min(1.0, drift_score)
        return min(1.0, coherence * 0.6 + no_drift * 0.4)

    def _map_emotion(self, relevance, implication, coping, norm_compat) -> str:
        if relevance <= self.RELEVANCE_THRESHOLD:
            return "neutral"
        if implication > 0.3 and coping > 0.5:
            return "joy"
        if implication < -0.3 and coping < 0.4:
            return "fear"
        if implication < -0.3 and coping >= 0.4:
            return "anger"
        if implication > 0.2 and norm_compat < 0.4:
            return "surprise"
        if implication > 0.1:
            return "wonder"
        if implication < -0.1:
            return "sadness"
        return "curiosity"

    @staticmethod
    def _compute_valence(implication, norm_compat, coping) -> float:
        v = implication * 0.6 + norm_compat * 0.4 - (1.0 - coping) * 0.2
        return max(-1.0, min(1.0, v))

    @staticmethod
    def _compute_arousal(relevance, implication, coping) -> float:
        a = relevance * 0.5 + abs(implication) * 0.3 + (1.0 - coping) * 0.2
        return max(0.0, min(1.0, a))

    @staticmethod
    def _compute_intensity(relevance, implication, arousal) -> float:
        return max(0.0, min(1.0, relevance * 0.4 + abs(implication) * 0.3 + arousal * 0.3))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "history": self._history[-self._history_cap :],
            "train_steps": self._train_steps,
        }
        if self.available and self._net is not None:
            data["net_state"] = {k: v.tolist() for k, v in self._net.state_dict().items()}
        return data

    def from_dict(self, data: Dict[str, Any]):
        self._history = data.get("history", [])
        self._train_steps = data.get("train_steps", 0)

        if self.available and self._net is not None:
            net_state = data.get("net_state")
            if net_state:
                try:
                    state = {k: torch.tensor(v) for k, v in net_state.items()}
                    self._net.load_state_dict(state, strict=True)
                except Exception:
                    print("[AppraisalEngine] Skipping net restore (dim mismatch)", flush=True)

    def summarize(self) -> Dict[str, Any]:
        if not self._history:
            return {"history_length": 0}
        last = self._history[-1]
        return {
            "history_length": len(self._history),
            "last_relevance": last.get("relevance", 0.0),
            "last_emotion": last.get("emotion", "neutral"),
            "last_overrides": last.get("overrides", False),
            "learned_weight": round(self.learned_weight, 3),
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "history_length": len(self._history),
            "available": self.available,
            "train_steps": self._train_steps,
            "learned_weight": round(self.learned_weight, 3),
            "accuracy": round(sum(self._rolling_accuracy) / max(len(self._rolling_accuracy), 1), 3),
        }
