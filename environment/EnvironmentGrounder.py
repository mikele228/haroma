"""
EnvironmentGrounder — World-state-to-KG bridge for HaromaX6 (Phase 15).

Extracts structured triples from the TextEnvironment's internal state,
tracks action-effect transitions, learns causal rules from repeated
observations, and trains a small predictive model for anticipating
environment outcomes.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import hashlib

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except ImportError:
    _TORCH = False

from engine.ComputeFabric import get_fabric as _get_fabric

_STATE_DIM = 32
_ACTION_DIM = 8
_OUTCOME_DIM = 8


if _TORCH:

    class _CausalPredictorNet(nn.Module):
        """Predicts environment outcome features from state + action features."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(_STATE_DIM + _ACTION_DIM, 24),
                nn.ReLU(),
                nn.Linear(24, 16),
                nn.ReLU(),
                nn.Linear(16, _OUTCOME_DIM),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


class EnvironmentGrounder:
    """Bridges environment state into KG and learns causal rules."""

    _ACTION_TYPES = ["move", "look", "talk", "take", "use", "wait", "other"]

    def __init__(self):
        self._transition_history: List[Dict[str, Any]] = []
        self._history_cap = 500
        self._causal_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self._causal_effects: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self._train_steps = 0
        self._train_buffer: List[Dict[str, Any]] = []
        self._buffer_cap = 256

        self.available = _TORCH
        if _TORCH:
            self._predictor = _CausalPredictorNet()
            _fab = _get_fabric()
            if _fab:
                self._predictor = _fab.register("causal_predictor", self._predictor)
            self._optim = torch.optim.Adam(self._predictor.parameters(), lr=1e-3)
        else:
            self._predictor = None
            self._optim = None

    def extract_world_triples(self, env) -> List[Dict[str, Any]]:
        """Read env internal state and produce NLU-shaped entity/relation dicts."""
        triples: List[Dict[str, Any]] = []

        player_loc = getattr(env, "_player_location", None)

        rooms = getattr(env, "_rooms", {})
        for room_id, room in rooms.items():
            room_name = getattr(room, "name", room_id) if hasattr(room, "name") else room_id
            triples.append(
                {
                    "subject": room_name,
                    "predicate": "is_a",
                    "object": "room",
                    "source": "environment",
                    "confidence": 0.95,
                }
            )
            exits = getattr(room, "exits", {}) if hasattr(room, "exits") else {}
            for direction, target_id in exits.items():
                target_room = rooms.get(target_id)
                target_name = (
                    getattr(target_room, "name", target_id)
                    if target_room and hasattr(target_room, "name")
                    else target_id
                )
                triples.append(
                    {
                        "subject": room_name,
                        "predicate": f"connects_{direction}",
                        "object": target_name,
                        "source": "environment",
                        "confidence": 0.95,
                    }
                )

        objects = getattr(env, "_objects", {})
        for obj_id, obj in objects.items():
            obj_name = getattr(obj, "name", obj_id) if hasattr(obj, "name") else obj_id
            obj_loc = getattr(obj, "location", "") if hasattr(obj, "location") else ""
            if obj_loc:
                loc_room = rooms.get(obj_loc)
                loc_name = (
                    getattr(loc_room, "name", obj_loc)
                    if loc_room and hasattr(loc_room, "name")
                    else obj_loc
                )
                triples.append(
                    {
                        "subject": obj_name,
                        "predicate": "located_in",
                        "object": loc_name,
                        "source": "environment",
                        "confidence": 0.9,
                    }
                )
            props = getattr(obj, "properties", {}) if hasattr(obj, "properties") else {}
            for prop_key, prop_val in props.items():
                triples.append(
                    {
                        "subject": obj_name,
                        "predicate": f"has_{prop_key}",
                        "object": str(prop_val),
                        "source": "environment",
                        "confidence": 0.85,
                    }
                )

        agents = getattr(env, "_agents", {})
        for agent_id, agent in agents.items():
            agent_name = getattr(agent, "name", agent_id) if hasattr(agent, "name") else agent_id
            agent_loc = getattr(agent, "location", "") if hasattr(agent, "location") else ""
            if agent_loc:
                loc_room = rooms.get(agent_loc)
                loc_name = (
                    getattr(loc_room, "name", agent_loc)
                    if loc_room and hasattr(loc_room, "name")
                    else agent_loc
                )
                triples.append(
                    {
                        "subject": agent_name,
                        "predicate": "is_in",
                        "object": loc_name,
                        "source": "environment",
                        "confidence": 0.9,
                    }
                )

        if player_loc:
            loc_room = rooms.get(player_loc)
            loc_name = (
                getattr(loc_room, "name", player_loc)
                if loc_room and hasattr(loc_room, "name")
                else player_loc
            )
            triples.append(
                {
                    "subject": "player",
                    "predicate": "is_in",
                    "object": loc_name,
                    "source": "environment",
                    "confidence": 0.95,
                }
            )

        inventory = getattr(env, "_inventory", [])
        for item_id in inventory:
            obj = objects.get(item_id)
            item_name = getattr(obj, "name", item_id) if obj and hasattr(obj, "name") else item_id
            triples.append(
                {
                    "subject": "player",
                    "predicate": "carries",
                    "object": item_name,
                    "source": "environment",
                    "confidence": 0.95,
                }
            )

        return triples

    def record_transition(
        self, pre_state: Dict[str, Any], action: Dict[str, Any], post_state: Dict[str, Any]
    ):
        """Store (room, verb, target, changes) for causal rule extraction."""
        room = pre_state.get("location", "")
        verb = action.get("strategy", action.get("action_type", "unknown"))
        target = action.get("text", "")[:30]
        post_room = post_state.get("location", "")

        changes: List[str] = []
        if room != post_room:
            changes.append(f"moved_to:{post_room}")

        pre_items = set(pre_state.get("inventory", []))
        post_items = set(post_state.get("inventory", []))
        for gained in post_items - pre_items:
            changes.append(f"gained:{gained}")
        for lost in pre_items - post_items:
            changes.append(f"lost:{lost}")

        transition = {
            "room": room,
            "verb": verb,
            "target": target,
            "changes": changes,
            "pre_location": room,
            "post_location": post_room,
        }
        self._transition_history.append(transition)
        if len(self._transition_history) > self._history_cap:
            self._transition_history = self._transition_history[-self._history_cap :]

        key = (room, verb, target)
        self._causal_counts[key] += 1
        for change in changes:
            self._causal_effects[key].append(change)
        if len(self._causal_effects[key]) > self._history_cap:
            self._causal_effects[key] = self._causal_effects[key][-self._history_cap :]

    def extract_causal_rules(self, min_support: int = 3) -> List[Dict[str, Any]]:
        """Mine transition history for repeated cause-effect patterns."""
        rules: List[Dict[str, Any]] = []
        for key, count in self._causal_counts.items():
            if count < min_support:
                continue
            room, verb, target = key
            effects = self._causal_effects.get(key, [])
            if not effects:
                continue

            effect_counts: Dict[str, int] = defaultdict(int)
            for e in effects:
                effect_counts[e] += 1

            dominant = max(effect_counts.items(), key=lambda x: x[1])
            confidence = dominant[1] / count

            rules.append(
                {
                    "name": f"env_{verb}_{target[:10]}",
                    "pattern_a": (room, verb, target),
                    "pattern_b": ("", "", ""),
                    "conclusion": ("", dominant[0], ""),
                    "source": "environment",
                    "support": count,
                    "confidence": round(confidence, 3),
                }
            )

        return rules

    def _encode_state(self, state_dict: Dict[str, Any]) -> List[float]:
        """Encode environment state into a fixed-size feature vector."""
        loc = state_dict.get("location", "")
        loc_hash = int(hashlib.md5(loc.encode()).hexdigest(), 16) % 1000 / 1000.0
        n_objects = min(1.0, len(state_dict.get("visible_objects", [])) / 10.0)
        n_npcs = min(1.0, len(state_dict.get("visible_agents", [])) / 5.0)
        n_exits = min(1.0, len(state_dict.get("exits", [])) / 4.0)
        tick = min(1.0, state_dict.get("world_tick", 0) / 200.0)
        n_inventory = min(1.0, len(state_dict.get("inventory", [])) / 10.0)

        vec = [loc_hash, n_objects, n_npcs, n_exits, tick, n_inventory]
        vec.extend([0.0] * (_STATE_DIM - len(vec)))
        return vec[:_STATE_DIM]

    def _encode_action(self, action: Dict[str, Any]) -> List[float]:
        """Encode action into a fixed-size feature vector."""
        strategy = action.get("strategy", action.get("action_type", "other"))
        idx = self._ACTION_TYPES.index(strategy) if strategy in self._ACTION_TYPES else 6
        vec = [0.0] * _ACTION_DIM
        if idx < _ACTION_DIM:
            vec[idx] = 1.0
        return vec

    def predict_outcome(
        self, state: Dict[str, Any], action: Dict[str, Any]
    ) -> Optional[List[float]]:
        if not self.available or self._predictor is None:
            return None
        if self._train_steps < 5:
            return None
        state_feats = self._encode_state(state)
        action_feats = self._encode_action(action)
        with torch.no_grad():
            _fab = _get_fabric()
            x = (
                _fab.tensor([state_feats + action_feats], dtype=torch.float32)
                if _fab
                else torch.tensor([state_feats + action_feats], dtype=torch.float32)
            )
            pred = self._predictor(x).squeeze(0)
        return pred.tolist()

    def train_step(self) -> Optional[float]:
        if not self.available or len(self._train_buffer) < 4:
            return None
        import random

        batch_size = min(16, len(self._train_buffer))
        batch = random.sample(self._train_buffer, batch_size)

        _fab = _get_fabric()
        inputs = (
            _fab.tensor([b["input"] for b in batch], dtype=torch.float32)
            if _fab
            else torch.tensor([b["input"] for b in batch], dtype=torch.float32)
        )
        targets = (
            _fab.tensor([b["target"] for b in batch], dtype=torch.float32)
            if _fab
            else torch.tensor([b["target"] for b in batch], dtype=torch.float32)
        )

        pred = self._predictor(inputs)
        loss = nn.functional.mse_loss(pred, targets)

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

    def record_training_sample(
        self, pre_state: Dict[str, Any], action: Dict[str, Any], post_state: Dict[str, Any]
    ):
        """Record a training sample for the causal predictor."""
        if not self.available:
            return
        state_feats = self._encode_state(pre_state)
        action_feats = self._encode_action(action)
        target_feats = self._encode_state(post_state)[:_OUTCOME_DIM]

        self._train_buffer.append(
            {
                "input": state_feats + action_feats,
                "target": target_feats,
            }
        )
        if len(self._train_buffer) > self._buffer_cap:
            self._train_buffer = self._train_buffer[-self._buffer_cap :]

    def stats(self) -> Dict[str, Any]:
        return {
            "transition_count": len(self._transition_history),
            "unique_causes": len(self._causal_counts),
            "train_steps": self._train_steps,
            "buffer_size": len(self._train_buffer),
            "available": self.available,
        }

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "transition_history": self._transition_history[-200:],
            "causal_counts": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in self._causal_counts.items()},
            "causal_effects": {
                f"{k[0]}|{k[1]}|{k[2]}": v[-200:] for k, v in self._causal_effects.items()
            },
            "train_steps": self._train_steps,
        }
        if self.available and self._predictor is not None:
            data["predictor_state"] = {
                k: v.tolist() for k, v in self._predictor.state_dict().items()
            }
        return data

    def from_dict(self, data: Dict[str, Any]):
        self._transition_history = data.get("transition_history", [])
        self._train_steps = data.get("train_steps", 0)

        raw_counts = data.get("causal_counts", {})
        self._causal_counts = defaultdict(int)
        for key_str, count in raw_counts.items():
            parts = key_str.split("|", 2)
            if len(parts) == 3:
                self._causal_counts[tuple(parts)] = count

        raw_effects = data.get("causal_effects", {})
        self._causal_effects = defaultdict(list)
        for key_str, effects in raw_effects.items():
            parts = key_str.split("|", 2)
            if len(parts) == 3:
                self._causal_effects[tuple(parts)] = effects

        if self.available and self._predictor is not None:
            ps = data.get("predictor_state")
            if ps:
                try:
                    state = {k: torch.tensor(v) for k, v in ps.items()}
                    self._predictor.load_state_dict(state)
                except Exception as _e:
                    print(f"[EnvironmentGrounder] from_dict load failed: {_e}", flush=True)
