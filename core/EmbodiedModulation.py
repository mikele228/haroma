"""
EmbodiedModulation — emotion reshapes processing parameters
(Phase 7 → Phase 11).

Rather than emotion only colouring output (tone, strategy scoring),
embodied modulation lets affect reshape the *processing itself*:

  workspace_capacity  — high arousal narrows attention (fewer slots)
  recall_limit        — threat widens memory scan
  inference_cap       — high arousal shortens reasoning depth
  novelty_bias        — positive valence encourages exploration
  curiosity_damping   — fear dampens exploration, wonder amplifies

Phase 11 upgrade: The closed-form formulas are preserved as a baseline.
A small PyTorch MLP learns optimal modulation mappings from outcome
feedback and blends in via the standard earned-weight ramp.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import random

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except (ImportError, OSError):
    _TORCH = False

from engine.ComputeFabric import get_fabric as _get_fabric


_DEFAULTS = {
    "workspace_capacity": 5,
    "recall_limit": 5,
    "inference_cap": 8,
    "novelty_bias": 0.0,
    "curiosity_damping": 1.0,
}

_PARAM_RANGES = {
    "workspace_capacity": (3.0, 7.0),
    "recall_limit": (3.0, 8.0),
    "inference_cap": (2.0, 10.0),
    "novelty_bias": (-0.2, 0.2),
    "curiosity_damping": (0.4, 1.4),
}

_INPUT_DIM = 2
_OUTPUT_DIM = 5
_Z_DIM = 512

if _TORCH:

    class _ModulationNet(nn.Module):
        """Maps (valence, arousal [+ optional z_t]) → 5 processing parameters."""

        def __init__(self, input_dim: int = _INPUT_DIM):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 12),
                nn.ReLU(),
                nn.Linear(12, _OUTPUT_DIM),
                nn.Tanh(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


class EmbodiedModulation:
    _RAMP_STEPS = 120
    _MAX_WEIGHT = 0.9

    def __init__(self):
        self.available = _TORCH
        self._train_steps = 0
        self._train_buffer: List[Dict[str, Any]] = []
        self._buffer_cap = 1024
        self._rolling_accuracy: List[float] = []
        self._accuracy_window = 50

        if _TORCH:
            self._net = _ModulationNet(_INPUT_DIM)
            self._net_z = _ModulationNet(_INPUT_DIM + _Z_DIM)
            _fab = _get_fabric()
            if _fab:
                self._net = _fab.register("modulation_net", self._net)
                self._net_z = _fab.register("modulation_net_z", self._net_z)
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

    def compute(
        self, emotion_summary: Dict[str, Any], z_t: "Optional[List[float]]" = None
    ) -> Dict[str, Any]:
        valence = emotion_summary.get("valence", 0.0)
        arousal = emotion_summary.get("arousal", 0.0)

        baseline = self._baseline(valence, arousal)
        learned = self._learned(valence, arousal, z_t=z_t)
        result = self._blend(baseline, learned)
        result["valence_input"] = round(valence, 3)
        result["arousal_input"] = round(arousal, 3)
        result["blend_weight"] = round(self.learned_weight, 3)
        return result

    def _baseline(self, valence: float, arousal: float) -> Dict[str, float]:
        workspace_capacity = round(5.0 - arousal * 2.0 + (1.0 - arousal) * 2.0)
        workspace_capacity = max(3, min(7, workspace_capacity))

        recall_limit = round(5.0 + arousal * 2.0 + max(0.0, -valence))
        recall_limit = max(3, min(8, recall_limit))

        inference_cap = round(6.0 - arousal * 3.0 + (1.0 - arousal) * 2.0)
        inference_cap = max(2, min(10, inference_cap))

        novelty_bias = valence * 0.15
        novelty_bias = max(-0.2, min(0.2, novelty_bias))

        curiosity_damping = 1.0 + valence * 0.3 - max(0.0, arousal - 0.7) * 0.4
        curiosity_damping = max(0.4, min(1.4, curiosity_damping))

        return {
            "workspace_capacity": float(workspace_capacity),
            "recall_limit": float(recall_limit),
            "inference_cap": float(inference_cap),
            "novelty_bias": round(novelty_bias, 3),
            "curiosity_damping": round(curiosity_damping, 3),
        }

    def _learned(
        self, valence: float, arousal: float, z_t: "Optional[List[float]]" = None
    ) -> Optional[Dict[str, float]]:
        if not self.available or self._net is None:
            return None
        w = self.learned_weight
        if w < 0.01:
            return None

        _fab = _get_fabric()
        with torch.no_grad():
            if z_t is not None and self._net_z is not None:
                feats = [valence, arousal] + z_t
                x = _fab.tensor([feats]) if _fab else torch.tensor([feats], dtype=torch.float32)
                out = self._net_z(x).squeeze(0)
            else:
                x = (
                    _fab.tensor([[valence, arousal]])
                    if _fab
                    else torch.tensor([[valence, arousal]], dtype=torch.float32)
                )
                out = self._net(x).squeeze(0)

        params = {}
        names = list(_PARAM_RANGES.keys())
        for i, name in enumerate(names):
            lo, hi = _PARAM_RANGES[name]
            raw = (out[i].item() + 1.0) / 2.0
            params[name] = lo + raw * (hi - lo)

        return params

    def _blend(
        self, baseline: Dict[str, float], learned: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        if learned is None:
            result = {}
            for key in _PARAM_RANGES:
                val = baseline[key]
                lo, hi = _PARAM_RANGES[key]
                if key in ("workspace_capacity", "recall_limit", "inference_cap"):
                    result[key] = round(max(lo, min(hi, val)))
                else:
                    result[key] = round(max(lo, min(hi, val)), 3)
            return result

        w = self.learned_weight
        result = {}
        for key in _PARAM_RANGES:
            b = baseline[key]
            l = learned[key]
            blended = (1.0 - w) * b + w * l
            lo, hi = _PARAM_RANGES[key]
            if key in ("workspace_capacity", "recall_limit", "inference_cap"):
                result[key] = round(max(lo, min(hi, blended)))
            else:
                result[key] = round(max(lo, min(hi, blended)), 3)
        return result

    def record_outcome(
        self,
        valence: float,
        arousal: float,
        modulation_used: Dict[str, Any],
        outcome_score: float,
        z_t: Optional[List[float]] = None,
    ):
        if not self.available:
            return

        names = list(_PARAM_RANGES.keys())
        targets = []
        for name in names:
            lo, hi = _PARAM_RANGES[name]
            actual = modulation_used.get(name, _DEFAULTS[name])
            if outcome_score > 0.6:
                target = actual
            else:
                target = _DEFAULTS[name]
            normalized = ((target - lo) / max(hi - lo, 1e-6)) * 2.0 - 1.0
            targets.append(normalized)

        self._train_buffer.append(
            {
                "features": [valence, arousal],
                "targets": targets,
                "z_t": z_t,
            }
        )
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

        features = (
            _fab.tensor([b["features"] for b in batch])
            if _fab
            else torch.tensor([b["features"] for b in batch], dtype=torch.float32)
        )
        targets = (
            _fab.tensor([b["targets"] for b in batch])
            if _fab
            else torch.tensor([b["targets"] for b in batch], dtype=torch.float32)
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
        z_batch = [b for b in batch if b.get("z_t") is not None]
        if z_batch and self._net_z is not None:
            self._net_z.train()
            z_features = (
                _fab.tensor([b["features"] + b["z_t"] for b in z_batch])
                if _fab
                else torch.tensor([b["features"] + b["z_t"] for b in z_batch], dtype=torch.float32)
            )
            z_targets = (
                _fab.tensor([b["targets"] for b in z_batch])
                if _fab
                else torch.tensor([b["targets"] for b in z_batch], dtype=torch.float32)
            )
            z_preds = self._net_z(z_features)
            z_loss = nn.functional.mse_loss(z_preds, z_targets)
            self._optim.zero_grad()
            if _fab:
                _fab.scale_loss(z_loss).backward()
                nn.utils.clip_grad_norm_(list(self._net_z.parameters()), 1.0)
                _fab.scaler_step(self._optim)
                _fab.scaler_update()
            else:
                z_loss.backward()
                self._optim.step()
            self._net_z.eval()

        self._train_steps += 1

        loss_val = loss.item()
        accuracy = 1.0 - min(1.0, loss_val)
        self._rolling_accuracy.append(accuracy)
        if len(self._rolling_accuracy) > self._accuracy_window:
            self._rolling_accuracy = self._rolling_accuracy[-self._accuracy_window :]

        return loss_val

    @staticmethod
    def defaults() -> Dict[str, Any]:
        return dict(_DEFAULTS)

    def stats(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "train_steps": self._train_steps,
            "learned_weight": round(self.learned_weight, 3),
            "buffer_size": len(self._train_buffer),
            "accuracy": round(sum(self._rolling_accuracy) / max(len(self._rolling_accuracy), 1), 3),
        }

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "train_steps": self._train_steps,
        }
        if self.available and self._net is not None:
            data["net_state"] = {k: v.tolist() for k, v in self._net.state_dict().items()}
        if self.available and getattr(self, "_net_z", None) is not None:
            data["net_z_state"] = {k: v.tolist() for k, v in self._net_z.state_dict().items()}
        return data

    def from_dict(self, data: Dict[str, Any]):
        self._train_steps = data.get("train_steps", 0)

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
                    except Exception as _em_exc:
                        print(
                            f"[EmbodiedModulation] from_dict load_state_dict({key}) failed: {_em_exc}",
                            flush=True,
                        )
