"""
MentalSimulator — Theory of Mind for HaromaX6 (Phase 15).

Maintains per-agent mental models (beliefs, goals, emotional state,
exposure history) and predicts agent behavior.  Enables perspective-
taking by answering "what does agent X believe about Y?" and
detecting knowledge gaps between self and other agents.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except (ImportError, OSError):
    _TORCH = False

from engine.ComputeFabric import get_fabric as _get_fabric

_BEHAVIOR_TYPES = [
    "speak",
    "question",
    "agree",
    "disagree",
    "move",
    "take",
    "give",
    "wait",
    "emote",
    "unknown",
]
_N_BEHAVIORS = len(_BEHAVIOR_TYPES)
_MENTAL_DIM = 32


if _TORCH:

    class _BehaviorPredictorNet(nn.Module):
        """Predicts action-type distribution for an agent given state features."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(_MENTAL_DIM, 24),
                nn.ReLU(),
                nn.Linear(24, 16),
                nn.ReLU(),
                nn.Linear(16, _N_BEHAVIORS),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


@dataclass
class AgentMentalModel:
    agent_id: str
    beliefs: List[Dict[str, Any]] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    emotional_state: Dict[str, float] = field(
        default_factory=lambda: {"valence": 0.0, "arousal": 0.0}
    )
    exposure_history: List[str] = field(default_factory=list)
    predicted_next: str = "unknown"
    interaction_count: int = 0
    last_seen_cycle: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "beliefs": self.beliefs[-30:],
            "goals": self.goals[-20:],
            "emotional_state": self.emotional_state,
            "exposure_history": self.exposure_history[-50:],
            "predicted_next": self.predicted_next,
            "interaction_count": self.interaction_count,
            "last_seen_cycle": self.last_seen_cycle,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentMentalModel":
        m = cls(agent_id=d.get("agent_id", "unknown"))
        m.beliefs = d.get("beliefs", [])
        m.goals = d.get("goals", [])
        m.emotional_state = d.get("emotional_state", {"valence": 0.0, "arousal": 0.0})
        m.exposure_history = d.get("exposure_history", [])
        m.predicted_next = d.get("predicted_next", "unknown")
        m.interaction_count = d.get("interaction_count", 0)
        m.last_seen_cycle = d.get("last_seen_cycle", 0)
        return m


class MentalSimulator:
    """Maintains per-agent belief/goal/emotion models and predicts behavior."""

    def __init__(self, max_agents: int = 30):
        self._models: Dict[str, AgentMentalModel] = {}
        self._max_agents = max_agents
        self._train_steps = 0
        self._train_buffer: List[Dict[str, Any]] = []
        self._buffer_cap = 256

        self.available = _TORCH
        if _TORCH:
            self._predictor = _BehaviorPredictorNet()
            _fab = _get_fabric()
            if _fab:
                self._predictor = _fab.register("behavior_predictor", self._predictor)
            self._optim = torch.optim.Adam(self._predictor.parameters(), lr=1e-3)
        else:
            self._predictor = None
            self._optim = None

    def update_model(
        self,
        agent_id: str,
        observed_action: Optional[str] = None,
        discourse_frames: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        cycle_id: int = 0,
    ):
        """Update beliefs, goals, emotion from observed behavior."""
        model = self._get_or_create(agent_id)
        model.interaction_count += 1
        model.last_seen_cycle = cycle_id

        if discourse_frames:
            for frame in discourse_frames:
                frame_hash = (
                    f"{frame.get('frame_type', '')}|"
                    f"{frame.get('agent', '')}|"
                    f"{frame.get('action', '')}|"
                    f"{frame.get('patient', '')}"
                )
                existing_hashes = set(
                    f"{b.get('frame_type', '')}|{b.get('agent', '')}|"
                    f"{b.get('action', '')}|{b.get('patient', '')}"
                    for b in model.beliefs
                )
                if frame_hash not in existing_hashes:
                    model.beliefs.append(frame)
                    if len(model.beliefs) > 50:
                        model.beliefs = model.beliefs[-50:]

        if context:
            sentiment = context.get("sentiment", {})
            pol = sentiment.get("polarity", 0.0)
            model.emotional_state["valence"] = (
                model.emotional_state.get("valence", 0.0) * 0.7 + pol * 0.3
            )
            content = context.get("content", "")
            if content and content not in model.exposure_history:
                model.exposure_history.append(content[:80])
                if len(model.exposure_history) > 100:
                    model.exposure_history = model.exposure_history[-100:]

        if observed_action:
            self._record_observation(agent_id, observed_action, model)
            model.predicted_next = observed_action

    def predict_behavior(
        self, agent_id: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Predict what this agent will likely do/say next."""
        model = self._models.get(agent_id)
        if not model:
            return {"predicted_action": "unknown", "confidence": 0.0}

        features = self._encode_mental_state(model, context)

        if self.available and self._predictor is not None and self._train_steps > 5:
            with torch.no_grad():
                _fab = _get_fabric()
                x = (
                    _fab.tensor([features], dtype=torch.float32)
                    if _fab
                    else torch.tensor([features], dtype=torch.float32)
                )
                logits = self._predictor(x).squeeze(0)
                probs = torch.softmax(logits, dim=-1)
            best_idx = probs.argmax().item()
            return {
                "predicted_action": _BEHAVIOR_TYPES[best_idx],
                "confidence": round(probs[best_idx].item(), 3),
                "distribution": {
                    _BEHAVIOR_TYPES[i]: round(probs[i].item(), 3)
                    for i in range(_N_BEHAVIORS)
                    if probs[i].item() > 0.05
                },
            }

        return {
            "predicted_action": model.predicted_next or "unknown",
            "confidence": min(0.5, model.interaction_count * 0.05),
        }

    def query_belief(self, agent_id: str, topic: str) -> List[Dict[str, Any]]:
        """What does this agent believe about topic?"""
        model = self._models.get(agent_id)
        if not model:
            return []
        topic_lower = topic.lower()
        matching: List[Dict[str, Any]] = []
        for belief in model.beliefs:
            b_text = (
                f"{belief.get('agent', '')} {belief.get('action', '')} {belief.get('patient', '')}"
            ).lower()
            if topic_lower in b_text:
                matching.append(belief)
        return matching

    def perspective_gap(self, agent_id: str, system_knowledge: List[str]) -> List[str]:
        """What does the system know that this agent doesn't?"""
        model = self._models.get(agent_id)
        if not model:
            return system_knowledge[:5]
        agent_exposure = set(e.lower() for e in model.exposure_history)
        gaps: List[str] = []
        for item in system_knowledge:
            if item.lower() not in agent_exposure:
                gaps.append(item)
            if len(gaps) >= 5:
                break
        return gaps

    def _get_or_create(self, agent_id: str) -> AgentMentalModel:
        if agent_id not in self._models:
            if len(self._models) >= self._max_agents:
                oldest = min(self._models.values(), key=lambda m: m.last_seen_cycle)
                del self._models[oldest.agent_id]
            self._models[agent_id] = AgentMentalModel(agent_id=agent_id)
        return self._models[agent_id]

    def _encode_mental_state(
        self, model: AgentMentalModel, context: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        vec: List[float] = [
            model.emotional_state.get("valence", 0.0),
            model.emotional_state.get("arousal", 0.0),
            min(1.0, model.interaction_count / 20.0),
            min(1.0, len(model.beliefs) / 30.0),
            min(1.0, len(model.goals) / 10.0),
            min(1.0, len(model.exposure_history) / 50.0),
        ]
        vec.extend([0.0] * (_MENTAL_DIM - len(vec)))
        return vec[:_MENTAL_DIM]

    def _record_observation(self, agent_id: str, action: str, model: AgentMentalModel):
        """Record for training the predictor."""
        if not self.available:
            return
        features = self._encode_mental_state(model)
        action_idx = (
            _BEHAVIOR_TYPES.index(action) if action in _BEHAVIOR_TYPES else _N_BEHAVIORS - 1
        )
        self._train_buffer.append(
            {
                "features": features,
                "target": action_idx,
            }
        )
        if len(self._train_buffer) > self._buffer_cap:
            self._train_buffer = self._train_buffer[-self._buffer_cap :]

    def train_step(self) -> Optional[float]:
        if not self.available or len(self._train_buffer) < 8:
            return None
        import random

        batch_size = min(16, len(self._train_buffer))
        batch = random.sample(self._train_buffer, batch_size)

        _fab = _get_fabric()
        features = (
            _fab.tensor([b["features"] for b in batch], dtype=torch.float32)
            if _fab
            else torch.tensor([b["features"] for b in batch], dtype=torch.float32)
        )
        targets = (
            _fab.tensor([b["target"] for b in batch], dtype=torch.long)
            if _fab
            else torch.tensor([b["target"] for b in batch], dtype=torch.long)
        )

        probs = self._predictor(features)
        loss = nn.functional.cross_entropy(probs, targets)

        self._optim.zero_grad()
        if _fab:
            _fab.scale_loss(loss).backward()
            _fab.scaler_step(self._optim)
            _fab.scaler_update()
        else:
            loss.backward()
            self._optim.step()

        self._train_steps += 1
        return loss.item()

    def stats(self) -> Dict[str, Any]:
        return {
            "tracked_agents": len(self._models),
            "train_steps": self._train_steps,
            "buffer_size": len(self._train_buffer),
            "available": self.available,
            "agent_ids": list(self._models.keys()),
        }

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "models": {aid: m.to_dict() for aid, m in self._models.items()},
            "train_steps": self._train_steps,
        }
        if self.available and self._predictor is not None:
            data["predictor_state"] = {
                k: v.tolist() for k, v in self._predictor.state_dict().items()
            }
        return data

    def from_dict(self, data: Dict[str, Any]):
        self._train_steps = data.get("train_steps", 0)
        self._models.clear()
        for aid, md in data.get("models", {}).items():
            self._models[aid] = AgentMentalModel.from_dict(md)
        if self.available and self._predictor is not None:
            ps = data.get("predictor_state")
            if ps:
                try:
                    state = {k: torch.tensor(v) for k, v in ps.items()}
                    self._predictor.load_state_dict(state)
                except Exception as _e:
                    print(f"[MentalSimulator] from_dict predictor load failed: {_e}", flush=True)
