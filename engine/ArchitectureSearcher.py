"""
ArchitectureSearcher — Self-modifying architecture search for HaromaX6.

Tracks per-module effectiveness over time, detects performance gaps,
proposes modifications (ProcessGate biases, hyper-param nudges), and
applies them safely.  A meta-learning calibrator learns when proposals
actually help versus hurt, adjusting proposal aggressiveness accordingly.

Gracefully degrades to no-op when PyTorch is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import math
import time

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except (ImportError, OSError):
    _TORCH = False

from engine.ComputeFabric import get_fabric as _get_fabric


_PROPOSAL_TYPES = ["enable_step", "disable_step", "bias_up", "bias_down", "noop"]
_N_PROPOSALS = len(_PROPOSAL_TYPES)
_STATE_DIM = 48
_HIDDEN = 32


# ------------------------------------------------------------------
# Neural components
# ------------------------------------------------------------------


class _ProposalScorerNet(nn.Module if _TORCH else object):
    """Scores candidate architecture modifications given cognitive state."""

    def __init__(self):
        if not _TORCH:
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_STATE_DIM, _HIDDEN),
            nn.ReLU(),
            nn.Linear(_HIDDEN, _N_PROPOSALS),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class _CalibrationNet(nn.Module if _TORCH else object):
    """Predicts whether a proposed modification will improve outcomes."""

    def __init__(self):
        if not _TORCH:
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_STATE_DIM + _N_PROPOSALS, _HIDDEN),
            nn.ReLU(),
            nn.Linear(_HIDDEN, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------


@dataclass
class ModuleRecord:
    """Rolling effectiveness tracker for one gatable step."""

    name: str
    outcome_sum: float = 0.0
    outcome_count: int = 0
    enabled_count: int = 0
    disabled_count: int = 0
    enabled_outcome_sum: float = 0.0
    disabled_outcome_sum: float = 0.0

    @property
    def avg_outcome(self) -> float:
        return self.outcome_sum / max(self.outcome_count, 1)

    @property
    def marginal_value(self) -> float:
        """How much better outcomes are when this module is ON vs OFF."""
        on_avg = self.enabled_outcome_sum / max(self.enabled_count, 1)
        off_avg = self.disabled_outcome_sum / max(self.disabled_count, 1)
        if self.enabled_count < 3 or self.disabled_count < 3:
            return 0.0
        return on_avg - off_avg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "outcome_sum": self.outcome_sum,
            "outcome_count": self.outcome_count,
            "enabled_count": self.enabled_count,
            "disabled_count": self.disabled_count,
            "enabled_outcome_sum": self.enabled_outcome_sum,
            "disabled_outcome_sum": self.disabled_outcome_sum,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModuleRecord":
        return cls(
            name=d["name"],
            outcome_sum=d.get("outcome_sum", 0.0),
            outcome_count=d.get("outcome_count", 0),
            enabled_count=d.get("enabled_count", 0),
            disabled_count=d.get("disabled_count", 0),
            enabled_outcome_sum=d.get("enabled_outcome_sum", 0.0),
            disabled_outcome_sum=d.get("disabled_outcome_sum", 0.0),
        )


@dataclass
class Proposal:
    """One architecture modification proposal."""

    proposal_type: str
    target_step: str = ""
    bias_delta: float = 0.0
    confidence: float = 0.0
    rationale: str = ""
    applied: bool = False
    outcome_before: float = 0.0
    outcome_after: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_type": self.proposal_type,
            "target_step": self.target_step,
            "bias_delta": self.bias_delta,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "applied": self.applied,
            "outcome_before": self.outcome_before,
            "outcome_after": self.outcome_after,
        }


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------


class ArchitectureSearcher:
    """
    Monitors module effectiveness, proposes safe modifications via
    ProcessGate biases, and calibrates its own proposal quality.
    """

    _RAMP_STEPS = 150
    _MAX_WEIGHT = 0.6
    _DECAY = 0.995
    _MAX_BIAS_MAG = 0.25
    _MIN_OBSERVATIONS = 20

    def __init__(self, gatable_steps: Optional[List[str]] = None):
        from engine.ProcessGate import GATABLE_STEPS as _DEFAULT_STEPS

        self._step_names: List[str] = list(gatable_steps or _DEFAULT_STEPS)

        self._records: Dict[str, ModuleRecord] = {
            name: ModuleRecord(name=name) for name in self._step_names
        }
        self._proposals_history: List[Dict[str, Any]] = []
        self._active_proposals: List[Proposal] = []
        self._train_steps: int = 0
        self._train_buffer: List[Dict[str, Any]] = []
        self._buffer_cap = 256
        self._proposal_cooldown: int = 0
        self._total_proposals: int = 0
        self._successful_proposals: int = 0

        self.available = _TORCH
        if _TORCH:
            self._scorer = _ProposalScorerNet()
            self._calibrator = _CalibrationNet()
            _fab = _get_fabric()
            if _fab:
                self._scorer = _fab.register("arch_scorer", self._scorer)
                self._calibrator = _fab.register("arch_calibrator", self._calibrator)
            params = list(self._scorer.parameters()) + list(self._calibrator.parameters())
            self._optim = torch.optim.Adam(params, lr=3e-4)
        else:
            self._scorer = None
            self._calibrator = None
            self._optim = None

    @property
    def learned_weight(self) -> float:
        if self._train_steps < 1:
            return 0.0
        return min(self._MAX_WEIGHT, self._train_steps / self._RAMP_STEPS * self._MAX_WEIGHT)

    # ------------------------------------------------------------------
    # Effectiveness tracking
    # ------------------------------------------------------------------

    def record_cycle(self, gate_decisions: Dict[str, bool], outcome_score: float):
        """Record which modules ran and the resulting outcome."""
        for name in self._step_names:
            rec = self._records.get(name)
            if rec is None:
                continue
            rec.outcome_sum = rec.outcome_sum * self._DECAY + outcome_score
            rec.outcome_count += 1
            if gate_decisions.get(name, True):
                rec.enabled_count += 1
                rec.enabled_outcome_sum = rec.enabled_outcome_sum * self._DECAY + outcome_score
            else:
                rec.disabled_count += 1
                rec.disabled_outcome_sum = rec.disabled_outcome_sum * self._DECAY + outcome_score

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    def detect_gaps(self) -> List[Dict[str, Any]]:
        """Find modules that are consistently dragging down performance."""
        gaps: List[Dict[str, Any]] = []
        for name, rec in self._records.items():
            if rec.outcome_count < self._MIN_OBSERVATIONS:
                continue
            mv = rec.marginal_value
            if mv < -0.05:
                gaps.append(
                    {
                        "step": name,
                        "marginal_value": round(mv, 4),
                        "type": "negative_marginal",
                        "enabled_pct": round(rec.enabled_count / max(rec.outcome_count, 1), 2),
                    }
                )
            elif rec.enabled_count > self._MIN_OBSERVATIONS and rec.avg_outcome < 0.3:
                gaps.append(
                    {
                        "step": name,
                        "avg_outcome": round(rec.avg_outcome, 4),
                        "type": "low_avg",
                        "enabled_pct": round(rec.enabled_count / max(rec.outcome_count, 1), 2),
                    }
                )
        return gaps

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------

    def generate_proposals(
        self,
        cognitive_snapshot: Optional[List[float]] = None,
    ) -> List[Proposal]:
        """Create ranked modification proposals based on gap analysis and
        neural scoring."""
        if self._proposal_cooldown > 0:
            self._proposal_cooldown -= 1
            return []

        gaps = self.detect_gaps()
        proposals: List[Proposal] = []

        for gap in gaps:
            step = gap["step"]
            if gap["type"] == "negative_marginal":
                proposals.append(
                    Proposal(
                        proposal_type="bias_down",
                        target_step=step,
                        bias_delta=-0.15,
                        rationale=f"{step} marginal value {gap['marginal_value']:.3f}",
                    )
                )
            elif gap["type"] == "low_avg":
                proposals.append(
                    Proposal(
                        proposal_type="disable_step",
                        target_step=step,
                        bias_delta=-self._MAX_BIAS_MAG,
                        rationale=f"{step} avg outcome {gap['avg_outcome']:.3f}",
                    )
                )

        if _TORCH and self._scorer is not None and cognitive_snapshot:
            snap = cognitive_snapshot[:_STATE_DIM]
            while len(snap) < _STATE_DIM:
                snap.append(0.0)
            with torch.no_grad():
                _fab = _get_fabric()
                x = _fab.tensor([snap]) if _fab else torch.tensor([snap], dtype=torch.float32)
                scores = self._scorer(x).squeeze(0)
            for i, ptype in enumerate(_PROPOSAL_TYPES):
                if ptype == "noop":
                    continue
                score = scores[i].item()
                if score > 0.6:
                    best_step = self._pick_target_for_proposal(ptype)
                    if best_step:
                        proposals.append(
                            Proposal(
                                proposal_type=ptype,
                                target_step=best_step,
                                bias_delta=self._bias_for_type(ptype),
                                confidence=score,
                                rationale=f"neural scorer {score:.2f}",
                            )
                        )

        if proposals:
            proposals = self._calibrate_proposals(proposals, cognitive_snapshot)

        return proposals

    def _pick_target_for_proposal(self, ptype: str) -> str:
        """Heuristically pick the best gatable step to target."""
        if ptype in ("enable_step", "bias_up"):
            candidates = [
                (n, r)
                for n, r in self._records.items()
                if r.marginal_value > 0.02 and r.disabled_count > 5
            ]
            if candidates:
                candidates.sort(key=lambda t: t[1].marginal_value, reverse=True)
                return candidates[0][0]
        elif ptype in ("disable_step", "bias_down"):
            candidates = [(n, r) for n, r in self._records.items() if r.marginal_value < -0.02]
            if candidates:
                candidates.sort(key=lambda t: t[1].marginal_value)
                return candidates[0][0]
        return ""

    @staticmethod
    def _bias_for_type(ptype: str) -> float:
        return {
            "enable_step": 0.15,
            "disable_step": -0.20,
            "bias_up": 0.10,
            "bias_down": -0.10,
            "noop": 0.0,
        }.get(ptype, 0.0)

    # ------------------------------------------------------------------
    # Meta-learning calibration
    # ------------------------------------------------------------------

    def _calibrate_proposals(
        self,
        proposals: List[Proposal],
        snapshot: Optional[List[float]],
    ) -> List[Proposal]:
        """Use the calibration net to filter proposals unlikely to help."""
        if not _TORCH or self._calibrator is None or snapshot is None:
            return proposals

        snap = snapshot[:_STATE_DIM]
        while len(snap) < _STATE_DIM:
            snap.append(0.0)

        calibrated: List[Proposal] = []
        for p in proposals:
            prop_vec = [0.0] * _N_PROPOSALS
            idx = (
                _PROPOSAL_TYPES.index(p.proposal_type) if p.proposal_type in _PROPOSAL_TYPES else -1
            )
            if idx >= 0:
                prop_vec[idx] = 1.0
            combined = snap + prop_vec
            with torch.no_grad():
                _fab = _get_fabric()
                x = (
                    _fab.tensor([combined])
                    if _fab
                    else torch.tensor([combined], dtype=torch.float32)
                )
                benefit_prob = self._calibrator(x).item()
            p.confidence = max(p.confidence, benefit_prob)
            if benefit_prob > 0.4:
                calibrated.append(p)

        calibrated.sort(key=lambda pp: pp.confidence, reverse=True)
        return calibrated[:3]

    # ------------------------------------------------------------------
    # Safe execution via ProcessGate bias
    # ------------------------------------------------------------------

    def apply_proposals(self, proposals: List[Proposal], process_gate) -> List[Proposal]:
        """Apply accepted proposals by adjusting ProcessGate biases."""
        applied: List[Proposal] = []
        for p in proposals:
            if not p.target_step:
                continue
            clamped = max(-self._MAX_BIAS_MAG, min(self._MAX_BIAS_MAG, p.bias_delta))
            process_gate.set_bias(p.target_step, clamped)
            p.applied = True
            applied.append(p)
            self._total_proposals += 1

        self._active_proposals = applied
        self._proposal_cooldown = 10
        return applied

    def evaluate_proposals(self, outcome_score: float):
        """After a cooldown period, assess whether active proposals helped."""
        for p in self._active_proposals:
            p.outcome_after = outcome_score
            delta = p.outcome_after - p.outcome_before
            if delta > 0.01:
                self._successful_proposals += 1
            self._proposals_history.append(p.to_dict())

            self._record_calibration_sample(p, delta)

        if len(self._proposals_history) > 200:
            self._proposals_history = self._proposals_history[-200:]

        self._active_proposals = []

    def _record_calibration_sample(self, p: Proposal, delta: float):
        """Store training samples for both scorer and calibrator nets."""
        if not _TORCH:
            return
        prop_vec = [0.0] * _N_PROPOSALS
        idx = _PROPOSAL_TYPES.index(p.proposal_type) if p.proposal_type in _PROPOSAL_TYPES else -1
        if idx >= 0:
            prop_vec[idx] = 1.0
        self._train_buffer.append(
            {
                "proposal_vec": prop_vec,
                "delta": delta,
                "confidence": p.confidence,
            }
        )
        if len(self._train_buffer) > self._buffer_cap:
            self._train_buffer = self._train_buffer[-self._buffer_cap :]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> Optional[float]:
        if not _TORCH or not self._train_buffer or len(self._train_buffer) < 8:
            return None

        import random

        batch = random.sample(self._train_buffer, min(16, len(self._train_buffer)))

        _fab = _get_fabric()
        prop_vecs = (
            _fab.tensor([s["proposal_vec"] for s in batch])
            if _fab
            else torch.tensor([s["proposal_vec"] for s in batch], dtype=torch.float32)
        )
        deltas = (
            _fab.tensor([s["delta"] for s in batch]).unsqueeze(1)
            if _fab
            else torch.tensor([s["delta"] for s in batch], dtype=torch.float32).unsqueeze(1)
        )
        targets = (deltas > 0.0).float()

        dummy_state = (
            _fab.zeros(len(batch), _STATE_DIM) if _fab else torch.zeros(len(batch), _STATE_DIM)
        )
        combined = torch.cat([dummy_state, prop_vecs], dim=1)
        preds = self._calibrator(combined)
        loss = nn.functional.binary_cross_entropy(preds, targets)

        self._optim.zero_grad()
        if _fab:
            _fab.scale_loss(loss).backward()
            nn.utils.clip_grad_norm_(
                list(self._scorer.parameters()) + list(self._calibrator.parameters()), 1.0
            )
            _fab.scaler_step(self._optim)
            _fab.scaler_update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self._scorer.parameters()) + list(self._calibrator.parameters()), 1.0
            )
            self._optim.step()

        self._train_steps += 1
        return loss.item()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        success_rate = self._successful_proposals / max(self._total_proposals, 1)
        top_gaps = sorted(
            [
                (n, r.marginal_value)
                for n, r in self._records.items()
                if r.outcome_count >= self._MIN_OBSERVATIONS
            ],
            key=lambda t: t[1],
        )[:5]
        return {
            "train_steps": self._train_steps,
            "learned_weight": round(self.learned_weight, 3),
            "total_proposals": self._total_proposals,
            "successful_proposals": self._successful_proposals,
            "success_rate": round(success_rate, 3),
            "cooldown": self._proposal_cooldown,
            "active_proposals": len(self._active_proposals),
            "top_gaps": [{"step": n, "mv": round(mv, 4)} for n, mv in top_gaps],
            "available": self.available,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "train_steps": self._train_steps,
            "total_proposals": self._total_proposals,
            "successful_proposals": self._successful_proposals,
            "proposal_cooldown": self._proposal_cooldown,
            "proposals_history": self._proposals_history[-100:],
            "records": {n: r.to_dict() for n, r in self._records.items()},
        }
        if _TORCH and self._scorer is not None:
            data["scorer_state"] = {k: v.tolist() for k, v in self._scorer.state_dict().items()}
            data["calibrator_state"] = {
                k: v.tolist() for k, v in self._calibrator.state_dict().items()
            }
        return data

    def from_dict(self, data: Dict[str, Any]):
        self._train_steps = data.get("train_steps", 0)
        self._total_proposals = data.get("total_proposals", 0)
        self._successful_proposals = data.get("successful_proposals", 0)
        self._proposal_cooldown = data.get("proposal_cooldown", 0)
        self._proposals_history = data.get("proposals_history", [])

        for name, rd in data.get("records", {}).items():
            self._records[name] = ModuleRecord.from_dict(rd)

        if _TORCH:
            for key, net in [
                ("scorer_state", self._scorer),
                ("calibrator_state", self._calibrator),
            ]:
                state = data.get(key)
                if state and net is not None:
                    try:
                        converted = {k: torch.tensor(v) for k, v in state.items()}
                        net.load_state_dict(converted)
                    except Exception as _e:
                        print(f"[ArchitectureSearcher] from_dict load failed: {_e}", flush=True)
