"""
ProcessGate — Dynamic Processing Allocation for HaromaX6 (Phase 13).

A learned gate that decides which optional cognitive steps to run or skip
each cycle.  Mandatory steps (perception, embedding, emotion, action,
evaluation, learning) always execute.  The gate learns which optional
steps actually improve outcomes via a REINFORCE-style update.

If PyTorch is unavailable the class degrades to "run everything".
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import math
import threading

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    _TORCH_AVAILABLE = False

from engine.ComputeFabric import get_fabric as _get_fabric

GATABLE_STEPS = [
    "self_prediction",
    "interlocutor_model",
    "kg_integration",
    "identity_update",
    "law_value_myth_fusion",
    "dream_consolidation",
    "reflection_diagnose",
    "curiosity",
    "reasoning",
    "counterfactual",
    "metacognition",
    "temporal_bind",
    "imagination",
    "goal_synthesis",
    "narrative_update",
    "phrase_extraction",
]

N_GATABLE = len(GATABLE_STEPS)
_STEP_IDX = {name: i for i, name in enumerate(GATABLE_STEPS)}
_INPUT_DIM = 22


_Z_DIM = 512


class _ProcessGateNet(nn.Module if _TORCH_AVAILABLE else object):
    """MLP (20 + optional 128-d z_t) -> 32 -> 16 -> N_GATABLE (sigmoid per step)."""

    def __init__(self, input_dim: int = _INPUT_DIM):
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        self._input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, N_GATABLE)
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.fc3.bias.fill_(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        return torch.sigmoid(self.fc3(h))


class ProcessGate:
    """Decides which optional cognitive steps to run per cycle."""

    def __init__(self, pending_lock: Optional[threading.Lock] = None):
        self._available = _TORCH_AVAILABLE
        self._train_steps: int = 0
        self._pending: List[Dict[str, Any]] = []
        self._run_threshold = 0.5
        self._biases: Dict[str, float] = {}
        self._has_backbone = False
        self._lock = pending_lock if pending_lock is not None else threading.Lock()

        if self._available:
            self._net = _ProcessGateNet(_INPUT_DIM)
            self._net_z = _ProcessGateNet(_INPUT_DIM + _Z_DIM)
            _fab = _get_fabric()
            if _fab:
                self._net = _fab.register("process_gate_net", self._net)
                self._net_z = _fab.register("process_gate_net_z", self._net_z)
            self._net.eval()
            self._net_z.eval()
            self._optimizer = torch.optim.Adam(self._net.parameters(), lr=0.0005)
            self._optimizer_z = torch.optim.Adam(self._net_z.parameters(), lr=0.0005)
        else:
            self._net = None
            self._net_z = None
            self._optimizer = None
            self._optimizer_z = None

    @property
    def learned_weight(self) -> float:
        if not self._available:
            return 0.0
        return min(0.6, self._train_steps / 300.0)

    def set_bias(self, step_name: str, bias_value: float):
        """Offset the gate sigmoid for a specific step (positive = more likely)."""
        if step_name in _STEP_IDX:
            self._biases[step_name] = bias_value

    def clear_bias(self, step_name: str):
        self._biases.pop(step_name, None)

    # Steps safe to skip during fast conversant cycles
    _CONVERSANT_SKIP = frozenset(
        {
            "dream_consolidation",
            "imagination",
            "counterfactual",
            "temporal_bind",
            "goal_synthesis",
            "reflection_diagnose",
        }
    )

    # Extra skips when POST /chat uses {"depth": "fast"} (latency-critical path)
    _CHAT_FAST_EXTRA_SKIP = frozenset(
        {
            "self_prediction",
            "interlocutor_model",
            "kg_integration",
            "identity_update",
            "curiosity",
            "reasoning",
            "metacognition",
            "narrative_update",
            "phrase_extraction",
            "law_value_myth_fusion",
        }
    )

    def decide(
        self,
        features: List[float],
        z_t: "Optional[List[float]]" = None,
        force_off: "Optional[frozenset]" = None,
    ) -> Dict[str, bool]:
        """Return {step_name: should_run} for each gatable step.

        If *force_off* is provided, those steps are unconditionally False
        regardless of the learned gate output.
        """
        lw = self.learned_weight
        if lw < 0.01 or not self._available or self._net is None:
            decisions = {name: True for name in GATABLE_STEPS}
        else:
            try:
                _fab = _get_fabric()
                with self._lock, torch.no_grad():
                    if z_t is not None and self._net_z is not None:
                        combined = features + z_t
                        x = (
                            _fab.tensor(combined)
                            if _fab
                            else torch.tensor(combined, dtype=torch.float32)
                        )
                        probs = self._net_z(x)
                        self._has_backbone = True
                    else:
                        x = (
                            _fab.tensor(features)
                            if _fab
                            else torch.tensor(features, dtype=torch.float32)
                        )
                        probs = self._net(x)

                decisions = {}
                for i, name in enumerate(GATABLE_STEPS):
                    learned_prob = probs[i].item()
                    bias = self._biases.get(name, 0.0)
                    learned_prob = max(0.0, min(1.0, learned_prob + bias))
                    blended = (1.0 - lw) * 1.0 + lw * learned_prob
                    decisions[name] = blended >= self._run_threshold
            except RuntimeError:
                decisions = {name: True for name in GATABLE_STEPS}

        if force_off:
            for name in force_off:
                if name in decisions:
                    decisions[name] = False

        return decisions

    def record_outcome(
        self,
        steps_run: Dict[str, bool],
        features: List[float],
        outcome_score: float,
        z_t: "Optional[List[float]]" = None,
    ):
        with self._lock:
            self._pending.append(
                {
                    "features": features,
                    "steps_run": {name: 1.0 if ran else 0.0 for name, ran in steps_run.items()},
                    "outcome": outcome_score,
                    "z_t": z_t,
                }
            )
            if len(self._pending) > 512:
                self._pending = self._pending[-512:]

    def train_step(self):
        if not self._available or self._net is None:
            return
        with self._lock:
            if len(self._pending) < 4:
                return
            batch = list(self._pending[-32:])

        _fab = _get_fabric()
        avg_outcome = sum(s["outcome"] for s in batch) / len(batch)

        if not batch:
            return

        base_samples = [s for s in batch if s.get("z_t") is None]
        z_samples = [s for s in batch if s.get("z_t") is not None]

        with self._lock:
            net = self._net
            net.train()

            if base_samples:
                self._optimizer.zero_grad()
                total_loss = 0.0

                for sample in base_samples:
                    x = (
                        _fab.tensor(sample["features"])
                        if _fab
                        else torch.tensor(sample["features"], dtype=torch.float32)
                    )
                    probs = net(x)

                    advantage = sample["outcome"] - avg_outcome

                    step_mask = (
                        _fab.tensor([sample["steps_run"].get(name, 1.0) for name in GATABLE_STEPS])
                        if _fab
                        else torch.tensor(
                            [sample["steps_run"].get(name, 1.0) for name in GATABLE_STEPS],
                            dtype=torch.float32,
                        )
                    )

                    log_probs = step_mask * torch.log(probs + 1e-8) + (1.0 - step_mask) * torch.log(
                        1.0 - probs + 1e-8
                    )
                    loss = -advantage * log_probs.sum()
                    total_loss = total_loss + loss

                total_loss = total_loss / len(base_samples)
                if _fab:
                    _fab.scale_loss(total_loss).backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    _fab.scaler_step(self._optimizer)
                    _fab.scaler_update()
                else:
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    self._optimizer.step()

            if z_samples and self._net_z is not None:
                self._net_z.train()
                self._optimizer_z.zero_grad()
                total_loss_z = 0.0
                for sample in z_samples:
                    combined = sample["features"] + sample["z_t"]
                    x = (
                        _fab.tensor(combined)
                        if _fab
                        else torch.tensor(combined, dtype=torch.float32)
                    )
                    probs = self._net_z(x)
                    advantage = sample["outcome"] - avg_outcome
                    step_mask = (
                        _fab.tensor([sample["steps_run"].get(name, 1.0) for name in GATABLE_STEPS])
                        if _fab
                        else torch.tensor(
                            [sample["steps_run"].get(name, 1.0) for name in GATABLE_STEPS],
                            dtype=torch.float32,
                        )
                    )
                    log_probs = step_mask * torch.log(probs + 1e-8) + (1.0 - step_mask) * torch.log(
                        1.0 - probs + 1e-8
                    )
                    loss_z = -advantage * log_probs.sum()
                    total_loss_z = total_loss_z + loss_z
                total_loss_z = total_loss_z / len(z_samples)
                if _fab:
                    _fab.scale_loss(total_loss_z).backward()
                    nn.utils.clip_grad_norm_(self._net_z.parameters(), 1.0)
                    _fab.scaler_step(self._optimizer_z)
                    _fab.scaler_update()
                else:
                    total_loss_z.backward()
                    nn.utils.clip_grad_norm_(self._net_z.parameters(), 1.0)
                    self._optimizer_z.step()
                self._net_z.eval()

            net.eval()
            self._train_steps += 1

    def stats(self) -> Dict[str, Any]:
        return {
            "available": self._available,
            "train_steps": self._train_steps,
            "learned_weight": round(self.learned_weight, 3),
            "n_gatable_steps": N_GATABLE,
            "has_backbone": self._has_backbone,
            "active_biases": len(self._biases),
        }

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "train_steps": self._train_steps,
            "biases": dict(self._biases),
        }
        if self._available and self._net is not None:
            data["net_state"] = {k: v.tolist() for k, v in self._net.state_dict().items()}
        if self._available and self._net_z is not None:
            data["net_z_state"] = {k: v.tolist() for k, v in self._net_z.state_dict().items()}
        return data

    def from_dict(self, data: Dict[str, Any]):
        saved_steps = data.get("train_steps", 0)
        self._biases = data.get("biases", {})
        if not self._available:
            self._train_steps = saved_steps
            return
        loaded_all = True
        for key, net in [("net_state", self._net), ("net_z_state", self._net_z)]:
            net_state = data.get(key)
            if net_state and net is not None:
                try:
                    restored = {k: torch.tensor(v) for k, v in net_state.items()}
                    net.load_state_dict(restored, strict=True)
                    net.eval()
                except (RuntimeError, Exception):
                    print(f"[ProcessGate] Skipping {key} (dim mismatch or error)", flush=True)
                    loaded_all = False
            else:
                loaded_all = False
        self._train_steps = saved_steps if loaded_all else 0
