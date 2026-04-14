"""
SelfModel — Predictive Self-Awareness for HaromaX6 (Phase 9).

Predicts Elarion's own cognitive outcomes (emotion, curiosity, action
strategy, attention pattern) *before* each cycle processes, then
compares predictions to actuals.  The prediction error about self --
"self-surprise" -- is a computational form of self-awareness grounded
in predictive processing theory.

If PyTorch is unavailable the class degrades to a no-op stub.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from collections import deque
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    np = None  # type: ignore

from engine.ComputeFabric import get_fabric as _get_fabric

STRATEGIES = ["inform", "inquire", "empathize", "advance_goal", "reflect"]
ATTENTION_SOURCES = [
    "perception",
    "memory",
    "emotion",
    "knowledge",
    "goals",
    "curiosity",
    "reasoning",
    "counterfactual",
    "metacognition",
]

_STRATEGY_TO_IDX = {s: i for i, s in enumerate(STRATEGIES)}
_ATTENTION_TO_IDX = {s: i for i, s in enumerate(ATTENTION_SOURCES)}

_SCALAR_DIM = 14  # 13 original + 1 cycle_delta
_GRU_HIDDEN = 512
_GRU_LAYERS = 2
_HISTORY_FEATURES = 32
N_CONTINUOUS = 3  # valence, arousal, curiosity
N_STRATEGIES = len(STRATEGIES)
N_ATTENTION = len(ATTENTION_SOURCES)
if _TORCH_AVAILABLE:

    class _TemporalSelfNet(nn.Module):
        """GRU-based self-model with temporal context and meta-prediction.

        Maintains a hidden state across cognitive cycles, enabling the
        self-model to learn temporal patterns in its own behaviour.
        """

        def __init__(self, input_dim: int = 270):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_dim + _HISTORY_FEATURES,
                hidden_size=_GRU_HIDDEN,
                num_layers=_GRU_LAYERS,
                batch_first=True,
                dropout=0.1 if _GRU_LAYERS > 1 else 0.0,
            )
            self.continuous_head = nn.Linear(_GRU_HIDDEN, N_CONTINUOUS)
            self.strategy_head = nn.Linear(_GRU_HIDDEN, N_STRATEGIES)
            self.attention_head = nn.Linear(_GRU_HIDDEN, N_ATTENTION)
            self.meta_head = nn.Sequential(
                nn.Linear(_GRU_HIDDEN, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            nn.init.xavier_uniform_(self.continuous_head.weight)
            nn.init.xavier_uniform_(self.strategy_head.weight)
            nn.init.xavier_uniform_(self.attention_head.weight)

        def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None):
            out, h = self.gru(
                x.unsqueeze(0).unsqueeze(0) if x.dim() == 1 else x.unsqueeze(1), hidden
            )
            out = out.squeeze(0).squeeze(0)
            continuous = self.continuous_head(out)
            strategy_logits = self.strategy_head(out)
            attention_logits = self.attention_head(out)
            meta_confidence = self.meta_head(out)
            return continuous, strategy_logits, attention_logits, meta_confidence, h


class SelfModel:
    """Public interface consumed by control.py and MetaCognitionEngine."""

    _HISTORY_SIZE = 32

    def __init__(self, encoder=None, embed_dim: int = 256):
        self._encoder = encoder
        self._embed_dim = embed_dim
        self._input_dim = embed_dim + _SCALAR_DIM
        self._train_steps: int = 0
        self._buffer_cap = 1024
        self._available = _TORCH_AVAILABLE

        self._gru_hidden: Optional[Any] = None
        self._history_errors: deque = deque(maxlen=self._HISTORY_SIZE)
        self._history_vals: deque = deque(maxlen=self._HISTORY_SIZE)
        self._history_aros: deque = deque(maxlen=self._HISTORY_SIZE)
        self._history_curs: deque = deque(maxlen=self._HISTORY_SIZE)
        self._history_strat_hits: deque = deque(maxlen=self._HISTORY_SIZE)
        self._last_meta_target: float = 0.5

        if self._available:
            self._temporal_model = _TemporalSelfNet(input_dim=self._input_dim)
            _fab = _get_fabric()
            if _fab:
                self._temporal_model = _fab.register(
                    "self_predictor_temporal", self._temporal_model
                )
            self._temporal_model.eval()
            self._temporal_optimizer = torch.optim.Adam(self._temporal_model.parameters(), lr=5e-4)
        else:
            self._temporal_model = None
            self._temporal_optimizer = None

        self._accuracy_window = 50
        self._emotion_surprises: deque = deque(maxlen=self._accuracy_window)
        self._strategy_hits: deque = deque(maxlen=self._accuracy_window)
        self._attention_hits: deque = deque(maxlen=self._accuracy_window)

        self._last_raw_outputs: Optional[tuple] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._available

    @property
    def learned_weight(self) -> float:
        if not self._available:
            return 0.0
        return min(0.9, self._train_steps / 100.0)

    @property
    def overall_accuracy(self) -> float:
        if not self._emotion_surprises:
            return 0.0
        emo_acc = 1.0 - (sum(self._emotion_surprises) / max(len(self._emotion_surprises), 1))
        strat_acc = sum(self._strategy_hits) / max(len(self._strategy_hits), 1)
        att_acc = sum(self._attention_hits) / max(len(self._attention_hits), 1)
        return 0.4 * emo_acc + 0.3 * strat_acc + 0.3 * att_acc

    # ------------------------------------------------------------------
    # Build input feature vector
    # ------------------------------------------------------------------

    def _history_features(self) -> List[float]:
        """Rolling statistics from the last N predictions and errors."""
        if not self._history_errors:
            return [0.0] * _HISTORY_FEATURES

        errs = list(self._history_errors)
        vals = list(self._history_vals) or [0.0]
        aros = list(self._history_aros) or [0.0]
        curs = list(self._history_curs) or [0.0]
        hits = list(self._history_strat_hits) or [0.0]

        mean_err = sum(errs) / len(errs)
        std_err = (sum((e - mean_err) ** 2 for e in errs) / len(errs)) ** 0.5
        mean_val = sum(vals) / len(vals)
        mean_aro = sum(aros) / len(aros)
        mean_cur = sum(curs) / len(curs)
        hit_rate = sum(hits) / len(hits)

        val_std = (sum((v - mean_val) ** 2 for v in vals) / len(vals)) ** 0.5
        aro_std = (sum((a - mean_aro) ** 2 for a in aros) / len(aros)) ** 0.5
        cur_std = (sum((c - mean_cur) ** 2 for c in curs) / len(curs)) ** 0.5

        err_trend = errs[-1] - errs[0] if len(errs) > 1 else 0.0
        val_trend = vals[-1] - vals[0] if len(vals) > 1 else 0.0
        aro_trend = aros[-1] - aros[0] if len(aros) > 1 else 0.0
        cur_trend = curs[-1] - curs[0] if len(curs) > 1 else 0.0

        max_err = max(errs) if errs else 0.0
        min_err = min(errs) if errs else 0.0

        n = len(errs)
        err_q1 = sorted(errs)[n // 4] if n >= 4 else mean_err
        err_q3 = sorted(errs)[(3 * n) // 4] if n >= 4 else mean_err

        features = [
            mean_err,
            std_err,
            mean_val,
            mean_aro,
            mean_cur,
            hit_rate,
            val_std,
            aro_std,
            cur_std,
            err_trend,
            val_trend,
            aro_trend,
            cur_trend,
            max_err,
            min_err,
            err_q1,
            err_q3,
            float(n) / 50.0,
        ]
        while len(features) < _HISTORY_FEATURES:
            features.append(0.0)
        return features[:_HISTORY_FEATURES]

    @staticmethod
    def _coerce_embedding_list(content_embedding) -> List[float]:
        """Flatten torch/numpy/nested vectors to a 1-D float list."""
        if content_embedding is None:
            return []
        if hasattr(content_embedding, "detach"):
            return [float(x) for x in content_embedding.detach().cpu().flatten().tolist()]
        try:
            import numpy as _np  # lazy: torch path already returned

            if isinstance(content_embedding, _np.ndarray):
                return [float(x) for x in content_embedding.flatten().tolist()]
        except Exception as _e:
            print(f"[SelfModel] embedding coerce error: {_e}", flush=True)
        if isinstance(content_embedding, (list, tuple)):
            flat: List[float] = []
            for x in content_embedding:
                if hasattr(x, "detach"):
                    flat.extend(float(t) for t in x.detach().cpu().flatten().tolist())
                else:
                    flat.append(float(x))
            return flat
        return []

    def _expected_body_embed_len(self) -> int:
        """Body = embed + _SCALAR_DIM; full GRU input = body + _HISTORY_FEATURES."""
        if self._temporal_model is None or not hasattr(self._temporal_model, "gru"):
            return self._embed_dim
        gru_in = int(self._temporal_model.gru.input_size)
        body = gru_in - _HISTORY_FEATURES
        return max(1, body - _SCALAR_DIM)

    def _build_input(self, content_embedding, prev_state: Dict[str, Any]) -> Optional[torch.Tensor]:
        if not self._available:
            return None

        embed_raw = self._coerce_embedding_list(content_embedding)
        tgt_len = self._expected_body_embed_len()
        if not embed_raw:
            embed = [0.0] * tgt_len
        elif len(embed_raw) > tgt_len:
            embed = embed_raw[:tgt_len]
        elif len(embed_raw) < tgt_len:
            embed = embed_raw + [0.0] * (tgt_len - len(embed_raw))
        else:
            embed = embed_raw

        prev_valence = prev_state.get("valence", 0.0)
        prev_arousal = prev_state.get("arousal", 0.0)
        prev_curiosity = prev_state.get("curiosity", 0.0)
        prev_outcome = prev_state.get("outcome_score", 0.0)

        prev_strategy = prev_state.get("strategy", "reflect")
        strategy_oh = [0.0] * N_STRATEGIES
        idx = _STRATEGY_TO_IDX.get(prev_strategy, 4)
        strategy_oh[idx] = 1.0

        has_external = float(prev_state.get("has_external", 1.0))
        wm_occupancy = prev_state.get("wm_occupancy", 0.0)
        recall_count = min(1.0, prev_state.get("recall_count", 0) / 10.0)
        cycle_norm = min(1.0, math.log1p(prev_state.get("cycle_count", 1)) / 10.0)
        cycle_delta = prev_state.get("cycle_delta", 0.0)

        features = (
            embed
            + [prev_valence, prev_arousal, prev_curiosity, prev_outcome]
            + strategy_oh
            + [has_external, wm_occupancy, recall_count, cycle_norm, cycle_delta]
        )

        _fab = _get_fabric()
        return _fab.tensor(features) if _fab else torch.tensor(features, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, content_embedding, prev_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._available:
            return None

        x = self._build_input(content_embedding, prev_state)
        if x is None:
            return None

        meta_confidence = 0.5

        hist = self._history_features()
        _fab = _get_fabric()
        x_hist = _fab.tensor(hist) if _fab else torch.tensor(hist, dtype=torch.float32)
        x_full = torch.cat([x, x_hist])
        with torch.no_grad():
            (continuous, strat_logits, att_logits, meta_conf, new_hidden) = self._temporal_model(
                x_full, self._gru_hidden
            )
        self._gru_hidden = new_hidden.detach()
        meta_confidence = float(meta_conf.item())

        self._last_raw_outputs = (continuous, strat_logits, att_logits)

        valence = float(continuous[0].clamp(-1, 1))
        arousal = float(continuous[1].clamp(0, 1))
        curiosity = float(continuous[2].clamp(0, 1))

        strat_probs = F.softmax(strat_logits, dim=0)
        att_probs = F.softmax(att_logits, dim=0)

        strategy_probs = {s: round(float(strat_probs[i]), 3) for i, s in enumerate(STRATEGIES)}
        attention_probs = {
            s: round(float(att_probs[i]), 3) for i, s in enumerate(ATTENTION_SOURCES)
        }

        return {
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
            "curiosity": round(curiosity, 3),
            "strategy": STRATEGIES[int(strat_probs.argmax())],
            "strategy_probs": strategy_probs,
            "attention": ATTENTION_SOURCES[int(att_probs.argmax())],
            "attention_probs": attention_probs,
            "meta_confidence": round(meta_confidence, 3),
        }

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------

    def compare(self, predicted: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
        if not predicted:
            return {}

        emo_err = abs(predicted["valence"] - actual.get("valence", 0.0)) + abs(
            predicted["arousal"] - actual.get("arousal", 0.0)
        )
        emotion_surprise = min(1.0, emo_err / 2.0)

        curiosity_surprise = min(1.0, abs(predicted["curiosity"] - actual.get("curiosity", 0.0)))

        actual_strategy = actual.get("strategy", "reflect")
        strategy_prob = predicted["strategy_probs"].get(actual_strategy, 0.2)
        strategy_surprise = 1.0 - strategy_prob

        actual_attention = actual.get("attention_winner", "perception")
        attention_prob = predicted["attention_probs"].get(actual_attention, 1.0 / N_ATTENTION)
        attention_surprise = 1.0 - attention_prob

        overall = (
            0.3 * emotion_surprise
            + 0.2 * curiosity_surprise
            + 0.3 * strategy_surprise
            + 0.2 * attention_surprise
        )

        self._emotion_surprises.append(emotion_surprise)
        self._strategy_hits.append(1.0 if predicted["strategy"] == actual_strategy else 0.0)
        self._attention_hits.append(1.0 if predicted["attention"] == actual_attention else 0.0)

        self._history_errors.append(overall)
        self._history_vals.append(actual.get("valence", 0.0))
        self._history_aros.append(actual.get("arousal", 0.0))
        self._history_curs.append(actual.get("curiosity", 0.0))
        self._history_strat_hits.append(1.0 if predicted["strategy"] == actual_strategy else 0.0)
        self._last_meta_target = 1.0 - min(1.0, overall)

        return {
            "emotion_surprise": round(emotion_surprise, 3),
            "curiosity_surprise": round(curiosity_surprise, 3),
            "strategy_surprise": round(strategy_surprise, 3),
            "attention_surprise": round(attention_surprise, 3),
            "overall_surprise": round(overall, 3),
            "emotion_surprised": emotion_surprise > 0.4,
            "strategy_surprised": strategy_surprise > 0.5,
            "attention_surprised": attention_surprise > 0.5,
            "accuracy": round(self.overall_accuracy, 3),
            "predicted_strategy": predicted["strategy"],
            "actual_strategy": actual_strategy,
            "predicted_attention": predicted["attention"],
            "actual_attention": actual_attention,
        }

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train_step(
        self, content_embedding, prev_state: Dict[str, Any], actual: Dict[str, Any]
    ) -> float:
        if not self._available:
            return 0.0

        x = self._build_input(content_embedding, prev_state)
        if x is None:
            return 0.0

        _fab = _get_fabric()
        target_continuous = (
            _fab.tensor(
                [
                    actual.get("valence", 0.0),
                    actual.get("arousal", 0.0),
                    actual.get("curiosity", 0.0),
                ]
            )
            if _fab
            else torch.tensor(
                [
                    actual.get("valence", 0.0),
                    actual.get("arousal", 0.0),
                    actual.get("curiosity", 0.0),
                ],
                dtype=torch.float32,
            )
        )

        actual_strategy = actual.get("strategy", "reflect")
        target_strategy = (
            _fab.tensor(_STRATEGY_TO_IDX.get(actual_strategy, 4), dtype=torch.long)
            if _fab
            else torch.tensor(_STRATEGY_TO_IDX.get(actual_strategy, 4), dtype=torch.long)
        )

        actual_attention = actual.get("attention_winner", "perception")
        target_attention = (
            _fab.tensor(_ATTENTION_TO_IDX.get(actual_attention, 0), dtype=torch.long)
            if _fab
            else torch.tensor(_ATTENTION_TO_IDX.get(actual_attention, 0), dtype=torch.long)
        )

        meta_target = (
            _fab.tensor([self._last_meta_target], dtype=torch.float32)
            if _fab
            else torch.tensor([self._last_meta_target], dtype=torch.float32)
        )

        self._temporal_model.train()
        hist = self._history_features()
        x_hist = _fab.tensor(hist) if _fab else torch.tensor(hist, dtype=torch.float32)
        x_full = torch.cat([x, x_hist])

        (continuous, strat_logits, att_logits, meta_conf, new_hidden) = self._temporal_model(
            x_full, self._gru_hidden
        )

        mse_loss = F.mse_loss(continuous, target_continuous)
        strat_loss = F.cross_entropy(strat_logits.unsqueeze(0), target_strategy.unsqueeze(0))
        att_loss = F.cross_entropy(att_logits.unsqueeze(0), target_attention.unsqueeze(0))
        meta_loss = F.mse_loss(meta_conf.squeeze(), meta_target.squeeze())

        total_loss = 0.35 * mse_loss + 0.25 * strat_loss + 0.25 * att_loss + 0.15 * meta_loss

        self._temporal_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self._temporal_model.parameters(), 1.0)
        self._temporal_optimizer.step()
        self._temporal_model.eval()
        self._gru_hidden = new_hidden.detach()

        self._train_steps += 1
        return float(total_loss.item())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "train_steps": self._train_steps,
            "embed_dim": self._embed_dim,
            "emotion_surprises": list(self._emotion_surprises),
            "strategy_hits": list(self._strategy_hits),
            "attention_hits": list(self._attention_hits),
        }
        if self._available and self._temporal_model is not None:
            data["temporal_state"] = {
                k: v.tolist() for k, v in self._temporal_model.state_dict().items()
            }
        return data

    def from_dict(self, data: Dict[str, Any]):
        self._train_steps = data.get("train_steps", 0)
        saved_dim = data.get("embed_dim", self._embed_dim)

        self._emotion_surprises.clear()
        for v in data.get("emotion_surprises", []):
            self._emotion_surprises.append(v)
        self._strategy_hits.clear()
        for v in data.get("strategy_hits", []):
            self._strategy_hits.append(v)
        self._attention_hits.clear()
        for v in data.get("attention_hits", []):
            self._attention_hits.append(v)

        if not self._available:
            return

        if saved_dim != self._embed_dim:
            _fab = _get_fabric()
            self._temporal_model = _TemporalSelfNet(input_dim=self._input_dim)
            if _fab:
                self._temporal_model = _fab.register(
                    "self_predictor_temporal", self._temporal_model
                )
            self._temporal_model.eval()
            self._temporal_optimizer = torch.optim.Adam(self._temporal_model.parameters(), lr=5e-4)
            self._gru_hidden = None
            self._train_steps = 0
            return

        temporal_state = data.get("temporal_state")
        if temporal_state and self._temporal_model is not None:
            try:
                restored = {k: torch.tensor(v) for k, v in temporal_state.items()}
                self._temporal_model.load_state_dict(restored)
                self._temporal_model.eval()
            except Exception as _e:
                print(f"[SelfModel] from_dict temporal_model load failed: {_e}", flush=True)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "available": self._available,
            "train_steps": self._train_steps,
            "learned_weight": round(self.learned_weight, 3),
            "overall_accuracy": round(self.overall_accuracy, 3),
            "emotion_accuracy": round(
                1.0 - (sum(self._emotion_surprises) / max(len(self._emotion_surprises), 1))
                if self._emotion_surprises
                else 0.0,
                3,
            ),
            "strategy_accuracy": round(
                sum(self._strategy_hits) / max(len(self._strategy_hits), 1)
                if self._strategy_hits
                else 0.0,
                3,
            ),
            "attention_accuracy": round(
                sum(self._attention_hits) / max(len(self._attention_hits), 1)
                if self._attention_hits
                else 0.0,
                3,
            ),
            "sample_count": len(self._emotion_surprises),
            "temporal_model": self._temporal_model is not None,
            "gru_hidden_active": self._gru_hidden is not None,
            "history_size": len(self._history_errors),
        }
