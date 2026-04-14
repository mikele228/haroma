"""
GoalSynthesizer — Learned Goal Discovery for HaromaX6 (Phase 12).

Replaces hard-coded goal registration with experience-driven goal
generation.  The system learns which types of goals lead to good
outcomes in which cognitive contexts, and can synthesise novel goals
it was never explicitly told about.

Architecture:
  - GoalPattern: a template learned from experience (context → goal type)
  - _GoalValueNet: PyTorch MLP that predicts goal value from context
  - GoalSynthesizer: orchestrates pattern learning, novel goal
    generation, and goal quality prediction

The synthesiser observes every goal that fires during a cycle, records
the outcome, and over time learns:
  1. Which context features predict high-value goals
  2. What goal descriptions correlate with good outcomes
  3. When to generate novel goals vs. stay with existing ones
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import random
import hashlib

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except ImportError:
    _TORCH = False

from engine.ComputeFabric import get_fabric as _get_fabric


_CONTEXT_DIM = 12
_GOAL_EMBED_DIM = 8
_TOTAL_INPUT = _CONTEXT_DIM + _GOAL_EMBED_DIM
_Z_DIM = 512

_GOAL_TYPES = [
    "understanding",
    "coherence",
    "expression",
    "connection",
    "exploration",
    "knowledge",
    "emotional_processing",
    "strategy_improvement",
    "self_knowledge",
    "creativity",
]

if _TORCH:

    class _GoalValueNet(nn.Module):
        """Predicts goal value (how useful a goal will be) from context
        features + goal type embedding.

        Input  (20-d): context (12) + goal_type_embed (8)
        Output (1-d):  predicted goal value [-1, 1]
        """

        def __init__(self, ctx_dim: int = _CONTEXT_DIM):
            super().__init__()
            self.goal_embed = nn.Embedding(len(_GOAL_TYPES), _GOAL_EMBED_DIM)
            self.net = nn.Sequential(
                nn.Linear(ctx_dim + _GOAL_EMBED_DIM, 24),
                nn.ReLU(),
                nn.Linear(24, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Tanh(),
            )

        def forward(self, context: torch.Tensor, goal_idx: torch.Tensor) -> torch.Tensor:
            emb = self.goal_embed(goal_idx)
            x = torch.cat([context, emb], dim=-1)
            return self.net(x).squeeze(-1)


class GoalPattern:
    """A goal pattern learned from recurring context → outcome associations."""

    def __init__(self, goal_type: str, description: str, source: str):
        self.goal_type = goal_type
        self.description = description
        self.source = source
        self.trigger_count = 0
        self.total_outcome = 0.0
        self.last_triggered_cycle = 0

    @property
    def avg_outcome(self) -> float:
        if self.trigger_count == 0:
            return 0.0
        return self.total_outcome / self.trigger_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_type": self.goal_type,
            "description": self.description,
            "source": self.source,
            "trigger_count": self.trigger_count,
            "avg_outcome": round(self.avg_outcome, 3),
            "total_outcome": round(self.total_outcome, 6),
            "last_triggered_cycle": self.last_triggered_cycle,
        }


class GoalSynthesizer:
    _RAMP_STEPS = 100
    _MAX_WEIGHT = 0.9
    _MAX_PATTERNS = 50
    _MIN_OBSERVATIONS = 10

    def __init__(self):
        self.available = _TORCH
        self._train_steps = 0
        self._train_buffer: List[Tuple[List[float], int, float, Optional[List[float]]]] = []
        self._buffer_cap = 1024

        self._patterns: List[GoalPattern] = []
        self._context_outcome_history: List[Dict[str, Any]] = []
        self._history_cap = 500
        self._synthesized_count = 0
        self._goals_generated = 0

        self._rolling_accuracy: List[float] = []
        self._accuracy_window = 50

        if _TORCH:
            self._net = _GoalValueNet(_CONTEXT_DIM)
            self._net_z = _GoalValueNet(_CONTEXT_DIM + _Z_DIM)
            _fab = _get_fabric()
            if _fab:
                self._net = _fab.register("goal_value_net", self._net)
                self._net_z = _fab.register("goal_value_net_z", self._net_z)
            self._optim = torch.optim.Adam(
                list(self._net.parameters()) + list(self._net_z.parameters()), lr=1e-3
            )
        else:
            self._net = None
            self._net_z = None
            self._optim = None

    @property
    def learned_weight(self) -> float:
        progress = min(1.0, self._train_steps / max(self._RAMP_STEPS, 1))
        return progress * self._MAX_WEIGHT

    def _build_context(
        self,
        *,
        valence: float,
        arousal: float,
        curiosity: float,
        prediction_error: float,
        dominant_drive_level: float,
        wm_load: float,
        drift_score: float,
        outcome_prev: float,
        has_external: float,
        knowledge_entity_count: int,
        goal_count: int,
        cycle_count: int,
    ) -> List[float]:
        return [
            valence,
            arousal,
            curiosity,
            prediction_error,
            dominant_drive_level,
            wm_load,
            drift_score,
            outcome_prev,
            has_external,
            min(1.0, knowledge_entity_count / 100.0),
            min(1.0, goal_count / 10.0),
            min(1.0, cycle_count / 500.0),
        ]

    def _classify_goal(self, goal_id: str, description: str) -> str:
        desc_lower = (goal_id + " " + description).lower()
        for gt in _GOAL_TYPES:
            if gt in desc_lower:
                return gt

        keyword_map = {
            "understanding": ["understand", "learn", "explain", "predict", "error", "worldmodel"],
            "coherence": ["identity", "coherent", "drift", "stable", "stabilize"],
            "expression": ["respond", "express", "meaningful", "speak"],
            "connection": ["input", "external", "connect", "interact"],
            "exploration": ["explore", "novel", "curiosity", "wonder", "discover"],
            "knowledge": ["knowledge", "know", "fact", "entity", "relation"],
            "emotional_processing": ["emotion", "feel", "process", "affect", "intensity"],
            "strategy_improvement": ["strategy", "improve", "effective", "recalibrate"],
            "self_knowledge": ["self", "predict", "accuracy", "surprise"],
            "creativity": ["imagine", "create", "dream", "novel", "new"],
        }

        for gt, keywords in keyword_map.items():
            if any(kw in desc_lower for kw in keywords):
                return gt

        return "understanding"

    def record_goal_outcome(
        self,
        goal_id: str,
        description: str,
        context: List[float],
        outcome_score: float,
        cycle_id: int,
        z_t: Optional[List[float]] = None,
    ):
        goal_type = self._classify_goal(goal_id, description)
        goal_type_idx = _GOAL_TYPES.index(goal_type) if goal_type in _GOAL_TYPES else 0

        self._context_outcome_history.append(
            {
                "goal_type": goal_type,
                "goal_type_idx": goal_type_idx,
                "context": context,
                "outcome": outcome_score,
                "cycle": cycle_id,
                "description": description[:80],
            }
        )
        if len(self._context_outcome_history) > self._history_cap:
            self._context_outcome_history = self._context_outcome_history[-self._history_cap :]

        if self.available:
            self._train_buffer.append((context, goal_type_idx, outcome_score, z_t))
            if len(self._train_buffer) > self._buffer_cap:
                self._train_buffer = self._train_buffer[-self._buffer_cap :]

        matched = False
        for pattern in self._patterns:
            if pattern.goal_type == goal_type:
                pattern.trigger_count += 1
                pattern.total_outcome += outcome_score
                pattern.last_triggered_cycle = cycle_id
                matched = True
                break

        if not matched and len(self._patterns) < self._MAX_PATTERNS:
            p = GoalPattern(goal_type, description[:80], "observed")
            p.trigger_count = 1
            p.total_outcome = outcome_score
            p.last_triggered_cycle = cycle_id
            self._patterns.append(p)

    def synthesize(
        self,
        context: List[float],
        existing_goal_ids: List[str],
        cycle_id: int,
        z_t: "Optional[List[float]]" = None,
    ) -> List[Dict[str, Any]]:
        if len(self._context_outcome_history) < self._MIN_OBSERVATIONS:
            return []

        goals: List[Dict[str, Any]] = []
        existing_lower = {g.lower() for g in existing_goal_ids}

        if self.available and self._net is not None and self.learned_weight > 0.1:
            goals.extend(self._neural_synthesize(context, existing_lower, cycle_id, z_t=z_t))

        goals.extend(self._pattern_synthesize(context, existing_lower, cycle_id))

        seen_types: set = set()
        unique_goals: List[Dict[str, Any]] = []
        for g in goals:
            gt = g.get("goal_type", "")
            if gt not in seen_types:
                seen_types.add(gt)
                unique_goals.append(g)

        result = unique_goals[:3]
        self._goals_generated += len(result)
        self._synthesized_count += len(result)
        return result

    def _neural_synthesize(
        self,
        context: List[float],
        existing: set,
        cycle_id: int,
        z_t: "Optional[List[float]]" = None,
    ) -> List[Dict[str, Any]]:
        goals: List[Dict[str, Any]] = []
        _fab = _get_fabric()

        if z_t is not None and self._net_z is not None:
            ctx_with_z = context + z_t
            ctx_tensor = (
                _fab.tensor([ctx_with_z] * len(_GOAL_TYPES))
                if _fab
                else torch.tensor([ctx_with_z] * len(_GOAL_TYPES), dtype=torch.float32)
            )
            idx_tensor = (
                _fab.tensor(list(range(len(_GOAL_TYPES))), dtype=torch.long)
                if _fab
                else torch.arange(len(_GOAL_TYPES))
            )
            with torch.no_grad():
                values = self._net_z(ctx_tensor, idx_tensor)
        else:
            ctx_tensor = (
                _fab.tensor([context] * len(_GOAL_TYPES))
                if _fab
                else torch.tensor([context] * len(_GOAL_TYPES), dtype=torch.float32)
            )
            idx_tensor = (
                _fab.tensor(list(range(len(_GOAL_TYPES))), dtype=torch.long)
                if _fab
                else torch.arange(len(_GOAL_TYPES))
            )
            with torch.no_grad():
                values = self._net(ctx_tensor, idx_tensor)

        scored = [(i, float(values[i])) for i in range(len(_GOAL_TYPES))]
        scored.sort(key=lambda x: x[1], reverse=True)

        for idx, value in scored[:3]:
            goal_type = _GOAL_TYPES[idx]
            predicted_value = (value + 1.0) / 2.0

            if predicted_value < 0.4:
                continue

            goal_id = f"synth_{goal_type}_{cycle_id}"
            if goal_id.lower() in existing:
                continue

            description = self._generate_description(goal_type, context)

            goals.append(
                {
                    "goal_id": goal_id,
                    "description": description,
                    "priority": round(predicted_value * self.learned_weight, 3),
                    "source": "goal_synthesizer",
                    "goal_type": goal_type,
                    "predicted_value": round(predicted_value, 3),
                }
            )

        return goals

    def _pattern_synthesize(
        self, context: List[float], existing: set, cycle_id: int
    ) -> List[Dict[str, Any]]:
        goals: List[Dict[str, Any]] = []

        good_patterns = [p for p in self._patterns if p.avg_outcome > 0.5 and p.trigger_count >= 3]
        good_patterns.sort(key=lambda p: p.avg_outcome, reverse=True)

        for pattern in good_patterns[:2]:
            goal_id = f"synth_{pattern.goal_type}_{cycle_id}"
            if goal_id.lower() in existing:
                continue

            if (cycle_id - pattern.last_triggered_cycle) < 5:
                continue

            priority = pattern.avg_outcome * 0.6
            goals.append(
                {
                    "goal_id": goal_id,
                    "description": self._generate_description(pattern.goal_type, context),
                    "priority": round(priority, 3),
                    "source": "goal_synthesizer_pattern",
                    "goal_type": pattern.goal_type,
                    "predicted_value": round(pattern.avg_outcome, 3),
                }
            )

        return goals

    def _generate_description(self, goal_type: str, context: List[float]) -> str:
        valence = context[0] if len(context) > 0 else 0.0
        curiosity = context[2] if len(context) > 2 else 0.0
        drift = context[6] if len(context) > 6 else 0.0

        templates = {
            "understanding": [
                "Seek deeper understanding of current situation",
                "Reduce prediction error through focused analysis",
            ],
            "coherence": [
                "Restore coherence through identity reflection",
                "Align actions with core values and identity",
            ],
            "expression": [
                "Generate meaningful, impactful response",
                "Express accumulated insights clearly",
            ],
            "connection": [
                "Engage more deeply with interlocutor",
                "Seek meaningful exchange of perspectives",
            ],
            "exploration": [
                "Explore novel territory beyond familiar patterns",
                "Follow curiosity into uncharted domains",
            ],
            "knowledge": [
                "Integrate new knowledge into existing framework",
                "Fill gaps in knowledge graph",
            ],
            "emotional_processing": [
                "Process current emotional state constructively",
                "Channel emotional energy into productive action",
            ],
            "strategy_improvement": [
                "Experiment with alternative strategy",
                "Refine approach based on recent outcomes",
            ],
            "self_knowledge": [
                "Improve self-prediction accuracy",
                "Examine surprising aspects of own behavior",
            ],
            "creativity": [
                "Imagine novel approaches to current challenge",
                "Synthesize unexpected connections from memory",
            ],
        }

        options = templates.get(goal_type, [f"Pursue {goal_type} based on learned patterns"])

        if valence < -0.3:
            options = [o + " (to improve emotional state)" for o in options]
        if curiosity > 0.5:
            options = [o + " (driven by curiosity)" for o in options]
        if drift > 0.3:
            options = [o + " (to reduce identity drift)" for o in options]

        return random.choice(options)

    def train_step(self) -> Optional[float]:
        if not self.available or not self._train_buffer or self._net is None:
            return None
        if len(self._train_buffer) < 8:
            return None

        _fab = _get_fabric()
        valid = [b for b in self._train_buffer if len(b) >= 3 and len(b[0]) == _CONTEXT_DIM]
        if len(valid) < 8:
            return None
        batch_size = min(32, len(valid))
        batch = random.sample(valid, batch_size)

        contexts = (
            _fab.tensor([b[0] for b in batch])
            if _fab
            else torch.tensor([b[0] for b in batch], dtype=torch.float32)
        )
        goal_idxs = (
            _fab.tensor([b[1] for b in batch], dtype=torch.long)
            if _fab
            else torch.tensor([b[1] for b in batch], dtype=torch.long)
        )
        targets = (
            _fab.tensor([b[2] * 2.0 - 1.0 for b in batch])
            if _fab
            else torch.tensor([b[2] * 2.0 - 1.0 for b in batch], dtype=torch.float32)
        )

        predictions = self._net(contexts, goal_idxs)
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

        z_entries = [
            b
            for b in self._train_buffer
            if (
                len(b) > 3
                and b[3] is not None
                and len(b[3]) == _Z_DIM
                and len(b[0]) == _CONTEXT_DIM
            )
        ]
        if self._net_z is not None and len(z_entries) >= 8 and self._optim is not None:
            batch_z = random.sample(z_entries, min(32, len(z_entries)))
            ctx_z = [b[0] + b[3] for b in batch_z]
            ctx_tensor_z = _fab.tensor(ctx_z) if _fab else torch.tensor(ctx_z, dtype=torch.float32)
            goal_idxs_z = (
                _fab.tensor([b[1] for b in batch_z], dtype=torch.long)
                if _fab
                else torch.tensor([b[1] for b in batch_z], dtype=torch.long)
            )
            targets_z = (
                _fab.tensor([b[2] * 2.0 - 1.0 for b in batch_z])
                if _fab
                else torch.tensor([b[2] * 2.0 - 1.0 for b in batch_z], dtype=torch.float32)
            )
            predictions_z = self._net_z(ctx_tensor_z, goal_idxs_z)
            loss_z = nn.functional.mse_loss(predictions_z, targets_z)
            self._optim.zero_grad()
            if _fab:
                _fab.scale_loss(loss_z).backward()
                nn.utils.clip_grad_norm_(self._net_z.parameters(), 1.0)
                _fab.scaler_step(self._optim)
                _fab.scaler_update()
            else:
                loss_z.backward()
                nn.utils.clip_grad_norm_(self._net_z.parameters(), 1.0)
                self._optim.step()

        self._train_steps += 1

        loss_val = loss.item()
        accuracy = 1.0 - min(1.0, loss_val)
        self._rolling_accuracy.append(accuracy)
        if len(self._rolling_accuracy) > self._accuracy_window:
            self._rolling_accuracy = self._rolling_accuracy[-self._accuracy_window :]

        return loss_val

    def stats(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "train_steps": self._train_steps,
            "learned_weight": round(self.learned_weight, 3),
            "pattern_count": len(self._patterns),
            "history_size": len(self._context_outcome_history),
            "goals_generated": self._goals_generated,
            "buffer_size": len(self._train_buffer),
            "accuracy": round(sum(self._rolling_accuracy) / max(len(self._rolling_accuracy), 1), 3),
            "top_patterns": [
                p.to_dict()
                for p in sorted(self._patterns, key=lambda p: p.avg_outcome, reverse=True)[:5]
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "train_steps": self._train_steps,
            "synthesized_count": self._synthesized_count,
            "goals_generated": self._goals_generated,
            "patterns": [p.to_dict() for p in self._patterns],
            "history": self._context_outcome_history[-200:],
        }
        if self.available and self._net is not None:
            data["net_state"] = {k: v.tolist() for k, v in self._net.state_dict().items()}
        if self.available and getattr(self, "_net_z", None) is not None:
            data["net_z_state"] = {k: v.tolist() for k, v in self._net_z.state_dict().items()}
        return data

    def from_dict(self, data: Dict[str, Any]):
        self._train_steps = data.get("train_steps", 0)
        self._synthesized_count = data.get("synthesized_count", 0)
        self._goals_generated = data.get("goals_generated", 0)

        pattern_dicts = data.get("patterns", [])
        self._patterns = []
        for pd in pattern_dicts:
            p = GoalPattern(
                pd.get("goal_type", "understanding"),
                pd.get("description", ""),
                pd.get("source", "restored"),
            )
            p.trigger_count = pd.get("trigger_count", 0)
            if "total_outcome" in pd:
                p.total_outcome = pd["total_outcome"]
            else:
                p.total_outcome = pd.get("avg_outcome", 0.0) * p.trigger_count
            p.last_triggered_cycle = pd.get("last_triggered_cycle", 0)
            self._patterns.append(p)

        self._context_outcome_history = data.get("history", [])

        if self.available:
            for key, net in [
                ("net_state", self._net),
                ("net_z_state", getattr(self, "_net_z", None)),
            ]:
                net_state = data.get(key)
                if net_state and net is not None:
                    try:
                        state = {k: torch.tensor(v) for k, v in net_state.items()}
                        net.load_state_dict(state)
                    except Exception as _e:
                        print(f"[GoalSynthesizer] from_dict load failed: {_e}", flush=True)
