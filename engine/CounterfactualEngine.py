"""
CounterfactualEngine — "What if?" reasoning over the KnowledgeGraph.

Generates hypothetical branches by temporarily modifying the knowledge
graph and re-running reasoning, then comparing outcomes against the
actual state.  Three branch types:

  1. Removal   — what if the newest relation hadn't been learned?
  2. Alternative — what if the opposite relation were true?
  3. Goal       — what if I had different goals?

Phase 13: A learned gate (`_CounterfactualGateNet`) decides whether to
explore and how deep, replacing the fixed max_branches=2 heuristic.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple, Set
from copy import deepcopy
import math

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from engine.ComputeFabric import get_fabric as _get_fabric

_GATE_INPUT_DIM = 12
_GATE_THRESHOLD_INIT = 0.3


class _CounterfactualGateNet(nn.Module if _TORCH_AVAILABLE else object):
    """MLP 12 -> 16 -> 8 -> 2 (sigmoid: explore_probability, depth_scale)."""

    def __init__(self):
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        self.fc1 = nn.Linear(12, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        return torch.sigmoid(self.fc3(h))


class CounterfactualEngine:
    """Performance-bounded counterfactual reasoning with a learned gate."""

    def __init__(self, max_branches: int = 3):
        self.max_branches = max_branches
        self._history: List[Dict[str, Any]] = []

        self._gate_available = _TORCH_AVAILABLE
        self._gate_train_steps: int = 0
        self._gate_threshold = _GATE_THRESHOLD_INIT
        self._buffer_cap = 512
        self._pending_outcomes: List[Dict[str, Any]] = []

        if self._gate_available:
            self._gate_net = _CounterfactualGateNet()
            _fab = _get_fabric()
            if _fab:
                self._gate_net = _fab.register("cf_gate", self._gate_net)
            self._gate_net.eval()
            self._gate_optimizer = torch.optim.Adam(self._gate_net.parameters(), lr=0.001)
        else:
            self._gate_net = None
            self._gate_optimizer = None

    @property
    def learned_weight(self) -> float:
        if not self._gate_available:
            return 0.0
        return min(0.8, self._gate_train_steps / 100)

    def gate(self, features: List[float]) -> Dict[str, Any]:
        """Run the gate net to decide explore probability and depth."""
        if not self._gate_available or self._gate_net is None:
            return {"explore": True, "effective_max_branches": 2, "gated": False}

        with torch.no_grad():
            _fab = _get_fabric()
            x = _fab.tensor(features) if _fab else torch.tensor(features, dtype=torch.float32)
            out = self._gate_net(x)
            explore_prob = out[0].item()
            depth_scale = out[1].item()

        lw = self.learned_weight
        blended_prob = (1.0 - lw) * 0.7 + lw * explore_prob
        blended_depth = (1.0 - lw) * 2.0 + lw * (depth_scale * 3.0)
        effective_branches = max(0, min(3, round(blended_depth)))
        should_explore = blended_prob >= self._gate_threshold

        return {
            "explore": should_explore,
            "effective_max_branches": effective_branches,
            "explore_prob": round(blended_prob, 3),
            "depth_scale": round(depth_scale, 3),
            "gated": True,
            "learned_weight": round(lw, 3),
        }

    def evaluate(
        self,
        knowledge_graph,
        reasoning_engine,
        reasoning_result: Dict[str, Any],
        knowledge_diff: Dict[str, Any],
        active_goals: List[Dict[str, Any]],
        nlu_result: Optional[Dict[str, Any]] = None,
        gate_decision: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        if gate_decision and not gate_decision.get("explore", True):
            result = {"branches": [], "counterfactual_depth": 0, "skipped_by_gate": True}
            self._history.append(result)
            if len(self._history) > 50:
                self._history = self._history[-50:]
            return result

        effective_max = self.max_branches
        if gate_decision and gate_decision.get("gated"):
            effective_max = gate_decision.get("effective_max_branches", 2)

        branches: List[Dict[str, Any]] = []

        if (
            effective_max > 0
            and knowledge_diff.get("changed")
            and knowledge_diff.get("new_relations", 0) > 0
        ):
            branch = self._removal_branch(
                knowledge_graph, reasoning_engine, reasoning_result, active_goals
            )
            if branch:
                branches.append(branch)

        if len(branches) < effective_max and nlu_result and nlu_result.get("relations"):
            branch = self._alternative_branch(
                knowledge_graph, reasoning_engine, nlu_result, reasoning_result, active_goals
            )
            if branch:
                branches.append(branch)

        if len(branches) < effective_max and len(active_goals) >= 2:
            branch = self._goal_branch(
                knowledge_graph, reasoning_engine, reasoning_result, active_goals
            )
            if branch:
                branches.append(branch)

        result = {
            "branches": branches,
            "counterfactual_depth": len(branches),
        }
        if gate_decision:
            result["gate_info"] = gate_decision
        self._history.append(result)
        if len(self._history) > 50:
            self._history = self._history[-50:]
        return result

    def record_outcome(self, features: List[float], cf_value: float):
        self._pending_outcomes.append({"features": features, "value": cf_value})
        if len(self._pending_outcomes) > self._buffer_cap:
            self._pending_outcomes = self._pending_outcomes[-self._buffer_cap :]

    def train_step(self):
        if not self._gate_available or not self._pending_outcomes:
            return
        batch = self._pending_outcomes[-32:]

        self._gate_net.train()
        _fab = _get_fabric()
        total_loss = 0.0
        for sample in batch:
            x = (
                _fab.tensor(sample["features"])
                if _fab
                else torch.tensor(sample["features"], dtype=torch.float32)
            )
            target_explore = 1.0 if sample["value"] > 0.3 else 0.0
            target_depth = min(1.0, sample["value"])
            target = (
                _fab.tensor([target_explore, target_depth])
                if _fab
                else torch.tensor([target_explore, target_depth], dtype=torch.float32)
            )

            pred = self._gate_net(x)
            loss = nn.functional.mse_loss(pred, target)
            self._gate_optimizer.zero_grad()
            if _fab:
                _fab.scale_loss(loss).backward()
                _fab.scaler_step(self._gate_optimizer)
                _fab.scaler_update()
            else:
                loss.backward()
                self._gate_optimizer.step()
            total_loss += loss.item()

        self._gate_net.eval()
        self._gate_train_steps += 1
        if self._gate_train_steps % 20 == 0:
            avg_value = sum(s["value"] for s in batch) / len(batch)
            self._gate_threshold = max(0.1, min(0.5, 0.3 + (avg_value - 0.5) * 0.2))

    def _removal_branch(
        self,
        kg,
        reasoning_engine,
        actual_reasoning: Dict[str, Any],
        active_goals: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        recent_rels = sorted(
            range(len(kg.relations)), key=lambda i: kg.relations[i].timestamp, reverse=True
        )
        if not recent_rels:
            return None

        remove_indices = set(recent_rels[: min(3, len(recent_rels))])
        removed_predicates = [
            kg.relations[i].predicate for i in remove_indices if i < len(kg.relations)
        ]

        shadow = _ShadowKG(kg, remove_indices)
        alt_result = reasoning_engine.reason(shadow, active_goals)

        actual_depth = actual_reasoning.get("reasoning_depth", 0)
        alt_depth = alt_result.get("reasoning_depth", 0)
        outcome_diff = (actual_depth - alt_depth) / max(actual_depth, 1)

        insight = (
            f"Removing recent relations ({', '.join(removed_predicates[:3])}) "
            f"{'reduces' if outcome_diff > 0 else 'does not reduce'} "
            f"reasoning depth by {abs(actual_depth - alt_depth)}"
        )

        return {
            "type": "removal",
            "insight": insight,
            "outcome_diff": round(outcome_diff, 3),
            "removed_predicates": removed_predicates[:3],
        }

    def _alternative_branch(
        self,
        kg,
        reasoning_engine,
        nlu_result: Dict[str, Any],
        actual_reasoning: Dict[str, Any],
        active_goals: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        relations = nlu_result.get("relations", [])
        if not relations:
            return None

        rel = relations[0]
        subj = rel.get("subject", "")
        pred = rel.get("predicate", "")
        obj = rel.get("object", "")
        if not (subj and pred and obj):
            return None

        neg_pred = f"not_{pred}" if not pred.startswith("not_") else pred[4:]

        shadow = _ShadowKGWithOverride(kg, subj, pred, obj, neg_pred)
        alt_result = reasoning_engine.reason(shadow, active_goals)

        actual_depth = actual_reasoning.get("reasoning_depth", 0)
        alt_depth = alt_result.get("reasoning_depth", 0)
        outcome_diff = (actual_depth - alt_depth) / max(actual_depth, 1)

        insight = (
            f"If '{subj} {neg_pred} {obj}' instead of "
            f"'{subj} {pred} {obj}', reasoning depth "
            f"changes by {alt_depth - actual_depth}"
        )

        return {
            "type": "alternative",
            "insight": insight,
            "outcome_diff": round(outcome_diff, 3),
            "original": f"{subj} {pred} {obj}",
            "counterfactual": f"{subj} {neg_pred} {obj}",
        }

    def _goal_branch(
        self,
        kg,
        reasoning_engine,
        actual_reasoning: Dict[str, Any],
        active_goals: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if len(active_goals) < 2:
            return None

        swapped_goals = list(active_goals)
        swapped_goals[0], swapped_goals[1] = swapped_goals[1], swapped_goals[0]
        alt_result = reasoning_engine.reason(kg, swapped_goals)

        actual_steps = len(actual_reasoning.get("plan_steps", []))
        alt_steps = len(alt_result.get("plan_steps", []))

        insight = (
            f"Swapping top goal '{active_goals[0].get('goal_id', '?')}' with "
            f"'{active_goals[1].get('goal_id', '?')}' "
            f"produces {alt_steps} plan steps (vs {actual_steps})"
        )

        return {
            "type": "goal",
            "insight": insight,
            "outcome_diff": round((actual_steps - alt_steps) / max(actual_steps, 1), 3),
            "swapped_goals": [
                active_goals[0].get("goal_id", ""),
                active_goals[1].get("goal_id", ""),
            ],
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "total_evaluations": len(self._history),
            "avg_branches": round(
                sum(h["counterfactual_depth"] for h in self._history) / max(len(self._history), 1),
                2,
            ),
            "gate_train_steps": self._gate_train_steps,
            "gate_learned_weight": round(self.learned_weight, 3),
            "gate_threshold": round(self._gate_threshold, 3),
        }

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "gate_train_steps": self._gate_train_steps,
            "gate_threshold": self._gate_threshold,
        }
        if self._gate_available and self._gate_net is not None:
            data["gate_state"] = {k: v.tolist() for k, v in self._gate_net.state_dict().items()}
        return data

    def from_dict(self, data: Dict[str, Any]):
        self._gate_train_steps = data.get("gate_train_steps", 0)
        self._gate_threshold = data.get("gate_threshold", _GATE_THRESHOLD_INIT)
        if not self._gate_available:
            return
        gate_state = data.get("gate_state")
        if gate_state and self._gate_net is not None:
            try:
                restored = {k: torch.tensor(v) for k, v in gate_state.items()}
                self._gate_net.load_state_dict(restored)
                self._gate_net.eval()
            except (RuntimeError, Exception):
                pass


class _ShadowKG:
    """Read-only KG view that hides a set of relation indices."""

    def __init__(self, real_kg, remove_indices: Set[int]):
        self._real = real_kg
        self._remove = remove_indices
        self.entities = real_kg.entities

        self.relations = [r for i, r in enumerate(real_kg.relations) if i not in remove_indices]

        self._adjacency = {}
        self._relation_index = {}
        for idx, rel in enumerate(self.relations):
            self._adjacency.setdefault(rel.subject_id, []).append(idx)
            self._adjacency.setdefault(rel.object_id, []).append(idx)
            self._relation_index[rel.triple_key()] = idx

    def _entity_name(self, entity_id: str) -> str:
        return self._real._entity_name(entity_id)


class _ShadowKGWithOverride:
    """Read-only KG view that replaces one predicate with its negation."""

    def __init__(self, real_kg, subj_name: str, orig_pred: str, obj_name: str, new_pred: str):
        self._real = real_kg
        self.entities = real_kg.entities

        from core.KnowledgeGraph import Relation as KGRelation

        self.relations = []
        for rel in real_kg.relations:
            s_name = real_kg._entity_name(rel.subject_id).lower()
            o_name = real_kg._entity_name(rel.object_id).lower()
            if (
                s_name == subj_name.lower()
                and rel.predicate == orig_pred
                and o_name == obj_name.lower()
            ):
                overridden = KGRelation(
                    subject_id=rel.subject_id,
                    predicate=new_pred,
                    object_id=rel.object_id,
                    confidence=rel.confidence,
                    timestamp=rel.timestamp,
                    source_cycle=rel.source_cycle,
                    source="counterfactual",
                )
                self.relations.append(overridden)
            else:
                self.relations.append(rel)

        self._adjacency = {}
        self._relation_index = {}
        for idx, rel in enumerate(self.relations):
            self._adjacency.setdefault(rel.subject_id, []).append(idx)
            self._adjacency.setdefault(rel.object_id, []).append(idx)
            self._relation_index[rel.triple_key()] = idx

    def _entity_name(self, entity_id: str) -> str:
        return self._real._entity_name(entity_id)
