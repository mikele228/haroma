"""
LearnedAttention — adaptive salience scoring for HaromaX6 (Phase 11).

Replaces fixed, hard-coded salience values with a self-grown neural
scorer that learns which cognitive sources deserve conscious access
based on experiential outcome feedback.

Two learned systems:
  1. Source salience scorer — a small MLP that predicts optimal salience
     for each workspace coalition source given current cognitive context.
  2. Episode salience weights — adaptive weights for EpisodeContext's
     compute_salience() that learn which factors best predict important
     moments.

Both start from the existing hard-coded values as a baseline and
gradually earn influence via the standard earned-weight ramp.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import math
import random

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except ImportError:
    _TORCH = False

from engine.ComputeFabric import get_fabric as _get_fabric


_KNOWN_SOURCES = [
    "perception",
    "memory",
    "emotion",
    "knowledge",
    "curiosity",
    "goals",
    "reasoning",
    "counterfactual",
    "metacognition",
    "drives",
]

_DEFAULT_SALIENCE_WEIGHTS = {
    "affect_intensity": 0.22,
    "goal_urgency": 0.13,
    "novelty_score": 0.13,
    "drift_score": 0.10,
    "memory_influence": 0.09,
    "integration_density": 0.13,
    "workspace_fill": 0.10,
    "appraisal_relevance": 0.05,
    "self_surprise": 0.05,
}

_CONTEXT_DIM = 8
_SOURCE_COUNT = len(_KNOWN_SOURCES)
_Z_DIM = 512


if _TORCH:

    class _AttentionScorerNet(nn.Module):
        """Context (8-d) + source one-hot (10-d) → salience adjustment."""

        def __init__(self, input_dim: int = _CONTEXT_DIM + _SOURCE_COUNT):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Tanh(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)


class LearnedAttention:
    _RAMP_STEPS = 120
    _MAX_WEIGHT = 0.9

    def __init__(self):
        self.available = _TORCH
        self._train_steps = 0
        self._train_buffer: List[Tuple[List[float], float, Optional[List[float]]]] = []
        self._buffer_cap = 1024

        self._salience_weights = dict(_DEFAULT_SALIENCE_WEIGHTS)
        self._salience_accum: Dict[str, List[float]] = {k: [] for k in _DEFAULT_SALIENCE_WEIGHTS}
        self._salience_train_steps = 0

        self._rolling_accuracy: List[float] = []
        self._accuracy_window = 50

        if _TORCH:
            self._net = _AttentionScorerNet(_CONTEXT_DIM + _SOURCE_COUNT)
            self._net_z = _AttentionScorerNet(_CONTEXT_DIM + _SOURCE_COUNT + _Z_DIM)
            _fab = _get_fabric()
            if _fab:
                self._net = _fab.register("attention_scorer", self._net)
                self._net_z = _fab.register("attention_scorer_z", self._net_z)
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

    @property
    def salience_learned_weight(self) -> float:
        progress = min(1.0, self._salience_train_steps / max(self._RAMP_STEPS, 1))
        return progress * self._MAX_WEIGHT

    def _build_context(
        self,
        valence: float,
        arousal: float,
        curiosity: float,
        wm_load: float,
        dominant_drive_level: float,
        outcome_prev: float,
        cycle_count: int,
        has_external: float,
    ) -> List[float]:
        return [
            valence,
            arousal,
            curiosity,
            wm_load,
            dominant_drive_level,
            outcome_prev,
            min(1.0, cycle_count / 500.0),
            has_external,
        ]

    def _source_onehot(self, source: str) -> List[float]:
        vec = [0.0] * _SOURCE_COUNT
        if source in _KNOWN_SOURCES:
            vec[_KNOWN_SOURCES.index(source)] = 1.0
        return vec

    def adjust_salience(
        self,
        source: str,
        base_salience: float,
        context: List[float],
        z_t: "Optional[List[float]]" = None,
    ) -> float:
        if not self.available or self._net is None:
            return base_salience

        w = self.learned_weight
        if w < 0.01:
            return base_salience

        features = context + self._source_onehot(source)
        _fab = _get_fabric()
        with torch.no_grad():
            if z_t is not None and self._net_z is not None:
                features_z = features + z_t
                x = (
                    _fab.tensor([features_z])
                    if _fab
                    else torch.tensor([features_z], dtype=torch.float32)
                )
                adjustment = self._net_z(x).item()
            else:
                x = (
                    _fab.tensor([features])
                    if _fab
                    else torch.tensor([features], dtype=torch.float32)
                )
                adjustment = self._net(x).item()

        learned = base_salience + adjustment * 0.3
        blended = (1.0 - w) * base_salience + w * learned
        return max(0.0, min(1.0, blended))

    def record_outcome(
        self,
        source_saliences: Dict[str, float],
        context: List[float],
        outcome_score: float,
        dominant_sources: List[str],
        z_t: "Optional[List[float]]" = None,
    ):
        if not self.available:
            return

        for source, salience in source_saliences.items():
            was_dominant = source in dominant_sources
            target = outcome_score if was_dominant else outcome_score * 0.3
            features = context + self._source_onehot(source)
            self._train_buffer.append((features, target - salience, z_t))

        if len(self._train_buffer) > self._buffer_cap:
            self._train_buffer = self._train_buffer[-self._buffer_cap :]

    def train_step(self) -> Optional[float]:
        if not self.available or not self._train_buffer or self._net is None:
            return None
        if len(self._train_buffer) < 8:
            return None

        _fab = _get_fabric()
        batch_size = min(32, len(self._train_buffer))
        batch = random.sample(self._train_buffer, batch_size)

        base_batch = [b for b in batch if len(b) <= 2 or b[2] is None]
        loss = None
        if base_batch:
            features = (
                _fab.tensor([b[0] for b in base_batch])
                if _fab
                else torch.tensor([b[0] for b in base_batch], dtype=torch.float32)
            )
            targets = (
                _fab.tensor([b[1] for b in base_batch])
                if _fab
                else torch.tensor([b[1] for b in base_batch], dtype=torch.float32)
            )

            predictions = self._net(features)
            loss = nn.functional.mse_loss(predictions, targets)

            self._optim.zero_grad()
            if _fab:
                _fab.scale_loss(loss).backward()
                nn.utils.clip_grad_norm_(
                    list(self._net.parameters()) + list(self._net_z.parameters()), 1.0
                )
                _fab.scaler_step(self._optim)
                _fab.scaler_update()
            else:
                loss.backward()
                self._optim.step()

        # Train _net_z on samples with z_t
        z_batch = [b for b in batch if len(b) > 2 and b[2] is not None]
        if z_batch and self._net_z is not None:
            self._net_z.train()
            z_features = (
                _fab.tensor([b[0] + b[2] for b in z_batch])
                if _fab
                else torch.tensor([b[0] + b[2] for b in z_batch], dtype=torch.float32)
            )
            z_targets = (
                _fab.tensor([b[1] for b in z_batch])
                if _fab
                else torch.tensor([b[1] for b in z_batch], dtype=torch.float32)
            )
            z_preds = self._net_z(z_features)
            z_loss = nn.functional.mse_loss(z_preds, z_targets)
            self._optim.zero_grad()
            if _fab:
                _fab.scale_loss(z_loss).backward()
                nn.utils.clip_grad_norm_(
                    list(self._net.parameters()) + list(self._net_z.parameters()), 1.0
                )
                _fab.scaler_step(self._optim)
                _fab.scaler_update()
            else:
                z_loss.backward()
                self._optim.step()
            self._net_z.eval()

        self._train_steps += 1

        loss_val = loss.item() if loss is not None else 0.0
        accuracy = 1.0 - min(1.0, loss_val)
        self._rolling_accuracy.append(accuracy)
        if len(self._rolling_accuracy) > self._accuracy_window:
            self._rolling_accuracy = self._rolling_accuracy[-self._accuracy_window :]

        return loss_val

    def record_salience_outcome(
        self, episode_factors: Dict[str, float], was_recalled_later: bool, recall_quality: float
    ):
        for key in _DEFAULT_SALIENCE_WEIGHTS:
            if key not in self._salience_accum:
                self._salience_accum[key] = []

            factor_val = episode_factors.get(key, 0.0)
            if was_recalled_later and factor_val > 0.1:
                self._salience_accum[key].append(recall_quality)
            elif not was_recalled_later and factor_val > 0.5:
                self._salience_accum[key].append(-0.1)

            if len(self._salience_accum[key]) > 100:
                self._salience_accum[key] = self._salience_accum[key][-100:]

    def update_salience_weights(self):
        updated = False
        for key in _DEFAULT_SALIENCE_WEIGHTS:
            samples = self._salience_accum.get(key, [])
            if len(samples) < 10:
                continue

            avg = sum(samples) / len(samples)
            default = _DEFAULT_SALIENCE_WEIGHTS[key]
            adjustment = avg * 0.02
            new_val = default + adjustment
            new_val = max(0.01, min(0.40, new_val))
            self._salience_weights[key] = new_val
            updated = True

        if updated:
            total = sum(self._salience_weights.values())
            if total > 0:
                self._salience_weights = {k: v / total for k, v in self._salience_weights.items()}
            self._salience_train_steps += 1

    def get_salience_weights(self) -> Dict[str, float]:
        w = self.salience_learned_weight
        if w < 0.01:
            return dict(_DEFAULT_SALIENCE_WEIGHTS)

        blended = {}
        for key in _DEFAULT_SALIENCE_WEIGHTS:
            default = _DEFAULT_SALIENCE_WEIGHTS[key]
            learned = self._salience_weights.get(key, default)
            blended[key] = (1.0 - w) * default + w * learned
        return blended

    @property
    def overall_accuracy(self) -> float:
        if not self._rolling_accuracy:
            return 0.0
        return sum(self._rolling_accuracy) / len(self._rolling_accuracy)

    def stats(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "train_steps": self._train_steps,
            "learned_weight": round(self.learned_weight, 3),
            "salience_train_steps": self._salience_train_steps,
            "salience_learned_weight": round(self.salience_learned_weight, 3),
            "buffer_size": len(self._train_buffer),
            "accuracy": round(self.overall_accuracy, 3),
            "salience_weights": {k: round(v, 4) for k, v in self._salience_weights.items()},
        }

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "train_steps": self._train_steps,
            "salience_train_steps": self._salience_train_steps,
            "salience_weights": self._salience_weights,
        }
        if self.available and self._net is not None:
            data["net_state"] = {k: v.tolist() for k, v in self._net.state_dict().items()}
        if self.available and self._net_z is not None:
            data["net_z_state"] = {k: v.tolist() for k, v in self._net_z.state_dict().items()}
        return data

    def from_dict(self, data: Dict[str, Any]):
        self._train_steps = data.get("train_steps", 0)
        self._salience_train_steps = data.get("salience_train_steps", 0)

        saved_weights = data.get("salience_weights")
        if saved_weights and isinstance(saved_weights, dict):
            self._salience_weights = saved_weights

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
                    except Exception as _la_exc:
                        print(
                            f"[LearnedAttention] from_dict load_state_dict failed: {_la_exc}",
                            flush=True,
                        )
