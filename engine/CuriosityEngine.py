"""
CuriosityEngine — Intrinsic Motivation for Sentient Cognition.

Generates internal drives from:
  - Prediction error (KG-based structural prediction + tag-based fallback)
  - Novelty detection (how different is this input from memory)
  - Information gap (what questions remain unanswered)
  - Emotional surprise (unexpected affect shifts)

Outputs goals that the system creates for itself, not from external input.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
import hashlib
import random
from utils.module_base import ModuleBase

try:
    import torch
    import torch.nn as nn
    import numpy as np

    _TORCH = True
except (ImportError, OSError):
    _TORCH = False


class TagSetWorldModel:
    """Tag-set transition model: learns (features_t) -> (features_t+1) over
    string feature sets and scores prediction error with Jaccard distance.
    Complements KG, embedding, and learned world models when tag features
    are the primary structural signal."""

    def __init__(self, max_history: int = 200):
        self.transitions: List[Tuple[frozenset, frozenset]] = []
        self.max_history = max_history
        self.total_error: float = 0.0
        self.total_predictions: int = 0

    def _similarity(self, a: frozenset, b: frozenset) -> float:
        if not a and not b:
            return 1.0
        union = len(a | b)
        if union == 0:
            return 1.0
        return len(a & b) / union

    def predict(self, current_features: Set[str]) -> Set[str]:
        if not self.transitions:
            return set()
        current = frozenset(current_features)
        best_sim = -1.0
        best_next: frozenset = frozenset()
        for prev, nxt in self.transitions:
            sim = self._similarity(current, prev)
            if sim > best_sim:
                best_sim = sim
                best_next = nxt
        return set(best_next)

    def learn(self, previous_features: Set[str], current_features: Set[str]):
        self.transitions.append((frozenset(previous_features), frozenset(current_features)))
        if len(self.transitions) > self.max_history:
            self.transitions = self.transitions[-self.max_history :]

    def compute_error(self, predicted: Set[str], actual: Set[str]) -> float:
        if not predicted and not actual:
            return 0.0
        union = len(predicted | actual)
        if union == 0:
            return 0.0
        error = 1.0 - len(predicted & actual) / union
        self.total_error += error
        self.total_predictions += 1
        return error

    @property
    def mean_error(self) -> float:
        if self.total_predictions == 0:
            return 0.5
        return self.total_error / self.total_predictions

    def stats(self) -> Dict[str, Any]:
        return {
            "transitions": len(self.transitions),
            "mean_error": round(self.mean_error, 4),
            "total_predictions": self.total_predictions,
        }


class KGWorldModel:
    """KnowledgeGraph-based world model that predicts structural state
    signatures (entity_count, relation_count, predicate_hash) and
    measures prediction error as normalized structural diff."""

    def __init__(self, max_history: int = 200):
        self.transitions: List[Tuple[Tuple[int, int, str], Tuple[int, int, str]]] = []
        self.max_history = max_history
        self.total_error: float = 0.0
        self.total_predictions: int = 0

    @staticmethod
    def _signature_from_summary(summary: Dict[str, Any]) -> Tuple[int, int, str]:
        ec = summary.get("entity_count", 0)
        rc = summary.get("relation_count", 0)
        preds = summary.get("predicate_types", [])
        ph = hashlib.md5("|".join(sorted(preds)).encode()).hexdigest()[:8]
        return (ec, rc, ph)

    def predict(self, current_sig: Tuple[int, int, str]) -> Tuple[int, int, str]:
        if not self.transitions:
            return current_sig
        best_dist = float("inf")
        best_next = current_sig
        for prev, nxt in self.transitions:
            d = abs(prev[0] - current_sig[0]) + abs(prev[1] - current_sig[1])
            if prev[2] != current_sig[2]:
                d += 2
            if d < best_dist:
                best_dist = d
                best_next = nxt
        return best_next

    def learn(self, prev_sig: Tuple[int, int, str], curr_sig: Tuple[int, int, str]):
        self.transitions.append((prev_sig, curr_sig))
        if len(self.transitions) > self.max_history:
            self.transitions = self.transitions[-self.max_history :]

    def compute_error(self, predicted: Tuple[int, int, str], actual: Tuple[int, int, str]) -> float:
        ec_diff = abs(predicted[0] - actual[0])
        rc_diff = abs(predicted[1] - actual[1])
        hash_diff = 0.0 if predicted[2] == actual[2] else 0.3
        max_scale = max(actual[0], actual[1], 1)
        error = min(1.0, (ec_diff + rc_diff) / max_scale + hash_diff)
        self.total_error += error
        self.total_predictions += 1
        return error

    @property
    def mean_error(self) -> float:
        if self.total_predictions == 0:
            return 0.5
        return self.total_error / self.total_predictions

    def stats(self) -> Dict[str, Any]:
        return {
            "transitions": len(self.transitions),
            "mean_error": round(self.mean_error, 4),
            "total_predictions": self.total_predictions,
        }


class EmbeddingWorldModel:
    """Dense-embedding state-transition model.

    Stores (embed_t, embed_t+1) pairs and predicts the next embedding
    by finding the most cosine-similar stored embed_t.
    """

    def __init__(self, max_history: int = 200):
        self.transitions: List[Tuple[Any, Any]] = []  # (np.ndarray, np.ndarray)
        self.max_history = max_history
        self.total_error: float = 0.0
        self.total_predictions: int = 0

    @staticmethod
    def _cosine(a, b) -> float:
        try:
            import numpy as np
        except ImportError:
            return 0.0
        dot = float(np.dot(a, b))
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def predict(self, current_embed):
        if not self.transitions:
            return current_embed
        best_sim = -2.0
        best_next = current_embed
        for prev, nxt in self.transitions:
            sim = self._cosine(current_embed, prev)
            if sim > best_sim:
                best_sim = sim
                best_next = nxt
        return best_next

    def learn(self, prev_embed, curr_embed):
        self.transitions.append((prev_embed, curr_embed))
        if len(self.transitions) > self.max_history:
            self.transitions = self.transitions[-self.max_history :]

    def compute_error(self, predicted, actual) -> float:
        error = 1.0 - max(0.0, self._cosine(predicted, actual))
        self.total_error += error
        self.total_predictions += 1
        return error

    @property
    def mean_error(self) -> float:
        if self.total_predictions == 0:
            return 0.5
        return self.total_error / self.total_predictions

    def stats(self) -> Dict[str, Any]:
        return {
            "transitions": len(self.transitions),
            "mean_error": round(self.mean_error, 4),
            "total_predictions": self.total_predictions,
        }


_EMBED_DIM = 256
_ACTION_DIM = 16
_WM_INPUT = _EMBED_DIM + _ACTION_DIM
_WM_HIDDEN = 128
_WM_MID = 64
_N_ENSEMBLE = 5
_REPLAY_CAP = 2048

STRATEGIES = ["inform", "inquire", "empathize", "advance_goal", "reflect"]
_STRATEGY_TO_IDX = {s: i for i, s in enumerate(STRATEGIES)}


if _TORCH:

    class _WorldModelMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(_WM_INPUT, _WM_HIDDEN),
                nn.ReLU(),
                nn.Linear(_WM_HIDDEN, _WM_MID),
                nn.ReLU(),
                nn.Linear(_WM_MID, _EMBED_DIM),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


class LearnedWorldModel:
    """Ensemble of 5 MLPs predicting next-state embeddings.

    Curiosity = ensemble disagreement (mean per-dimension variance of
    predictions across the ensemble members).
    """

    @staticmethod
    def _flat_state_embedding(state_emb: Any) -> List[float]:
        """Flatten torch/numpy/nested values to exactly ``_EMBED_DIM`` floats."""
        if state_emb is None:
            return [0.0] * _EMBED_DIM
        if hasattr(state_emb, "detach"):
            vals = [float(x) for x in state_emb.detach().cpu().flatten().tolist()]
        else:
            try:
                arr = np.asarray(state_emb, dtype=np.float32).reshape(-1)
                vals = [float(x) for x in arr.tolist()]
            except Exception:
                vals = []
                seq = state_emb if isinstance(state_emb, (list, tuple)) else [state_emb]
                for x in seq:
                    if hasattr(x, "detach"):
                        vals.extend(float(t) for t in x.detach().cpu().flatten().tolist())
                    else:
                        try:
                            vals.extend(
                                float(t)
                                for t in np.asarray(x, dtype=np.float32).reshape(-1).tolist()
                            )
                        except Exception:
                            vals.append(float(x))
        if len(vals) > _EMBED_DIM:
            vals = vals[:_EMBED_DIM]
        elif len(vals) < _EMBED_DIM:
            vals = vals + [0.0] * (_EMBED_DIM - len(vals))
        return vals

    def __init__(self):
        self._available = _TORCH
        self._models: List[Any] = []
        self._optimizers: List[Any] = []
        self._replay: List[Tuple[Any, Any, Any]] = []
        self._train_steps = 0

        if not _TORCH:
            return

        for _ in range(_N_ENSEMBLE):
            m = _WorldModelMLP()
            self._models.append(m)
            self._optimizers.append(torch.optim.Adam(m.parameters(), lr=1e-3))

    @property
    def available(self) -> bool:
        return self._available and len(self._models) > 0

    @staticmethod
    def _action_features(strategy: str, scalars: Optional[List[float]] = None) -> List[float]:
        """Build a 16-d action feature vector: 5 strategy one-hot + 11 scalars."""
        vec = [0.0] * _ACTION_DIM
        idx = _STRATEGY_TO_IDX.get(strategy, 0)
        if idx < 5:
            vec[idx] = 1.0
        if scalars:
            for i, v in enumerate(scalars[:11]):
                vec[5 + i] = float(v)
        return vec

    def predict(
        self,
        state_emb: Any,
        action_features: List[float],
    ) -> Tuple[Any, float]:
        """Return (mean_predicted_next_state, mean_variance)."""
        if not self.available:
            return state_emb, 0.0
        emb_list = self._flat_state_embedding(state_emb)
        inp = torch.tensor(
            emb_list + action_features,
            dtype=torch.float32,
        ).unsqueeze(0)
        preds = []
        for m in self._models:
            m.eval()
            with torch.no_grad():
                preds.append(m(inp).squeeze(0))
        stacked = torch.stack(preds)  # (N, embed_dim)
        mean_pred = stacked.mean(dim=0)
        variance = stacked.var(dim=0).mean().item()
        return mean_pred.detach().cpu().numpy(), variance

    def curiosity_score(
        self,
        state_emb: Any,
        action_features: List[float],
    ) -> float:
        """Ensemble disagreement as a curiosity signal in [0, 1]."""
        _, var = self.predict(state_emb, action_features)
        return min(1.0, var * 10.0)

    def train_step(
        self,
        state_emb: Any,
        action_features: List[float],
        actual_next_state: Any,
    ) -> float:
        """Train all ensemble members on a transition. Returns mean loss."""
        if not self.available:
            return 0.0

        self._replay.append(
            (
                self._flat_state_embedding(state_emb) + action_features,
                self._flat_state_embedding(actual_next_state),
                None,
            )
        )
        if len(self._replay) > _REPLAY_CAP:
            self._replay = self._replay[-_REPLAY_CAP:]

        batch_size = min(32, len(self._replay))
        batch = random.sample(self._replay, batch_size)
        inputs = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        targets = torch.tensor([b[1] for b in batch], dtype=torch.float32)

        total_loss = 0.0
        for m, opt in zip(self._models, self._optimizers):
            m.train()
            pred = m(inputs)
            loss = nn.functional.mse_loss(pred, targets)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        self._train_steps += 1
        return total_loss / _N_ENSEMBLE

    def stats(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "ensemble_size": _N_ENSEMBLE,
            "replay_size": len(self._replay),
            "train_steps": self._train_steps,
        }


class CuriosityEngine(ModuleBase):
    _MAX_NOVELTY_BASELINE = 5000

    def __init__(self, encoder=None):
        super().__init__("CuriosityEngine")
        self.world_model = TagSetWorldModel()
        self.kg_world_model = KGWorldModel()
        self.embed_world_model = EmbeddingWorldModel()
        self.learned_world_model = LearnedWorldModel()
        self._encoder = encoder
        self.novelty_baseline: Dict[str, int] = {}
        self.open_questions: List[str] = []
        self.curiosity_score: float = 0.0
        self._previous_features: Set[str] = set()
        self._last_prediction: Set[str] = set()
        self._prev_kg_sig: Optional[Tuple[int, int, str]] = None
        self._prev_embedding = None
        self._prev_strategy: str = "reflect"
        self._recent_embeddings: List[Any] = []

    def _extract_features(self, episode: Dict[str, Any]) -> Set[str]:
        features: Set[str] = set()
        if isinstance(episode.get("perception"), dict):
            features.update(episode["perception"].get("tags", []))
            content = episode["perception"].get("content", "")
            if isinstance(content, str):
                features.update(w.lower() for w in content.split() if len(w) > 2)
        features.update(episode.get("tags", []))
        emotion = episode.get("emotion", "")
        if emotion and emotion != "neutral":
            features.add(f"emo:{emotion}")
        return features

    def evaluate(
        self,
        episode_summary: Dict[str, Any],
        recalled_memories: list,
        forecast_result: Dict[str, Any],
        emotion_summary: Dict[str, Any],
        knowledge_summary: Optional[Dict[str, Any]] = None,
        current_embedding=None,
        last_strategy: str = "reflect",
    ) -> Dict[str, Any]:
        current_features = self._extract_features(episode_summary)

        novelty = self._compute_novelty(episode_summary, recalled_memories, current_embedding)
        tag_error = self._compute_prediction_error(current_features)
        kg_error = self._compute_kg_prediction_error(knowledge_summary)

        # Ensemble disagreement from learned world model (Upgrade 3)
        # Must run BEFORE _compute_embedding_prediction_error which
        # overwrites _prev_embedding
        ensemble_disagreement = 0.0
        if self.learned_world_model.available and current_embedding is not None:
            af = LearnedWorldModel._action_features(self._prev_strategy)
            ensemble_disagreement = self.learned_world_model.curiosity_score(current_embedding, af)

            if self._prev_embedding is not None:
                prev_af = LearnedWorldModel._action_features(self._prev_strategy)
                self.learned_world_model.train_step(
                    self._prev_embedding, prev_af, current_embedding
                )

        embed_error = self._compute_embedding_prediction_error(current_embedding)

        encoder_lw = self._encoder.learned_weight if self._encoder is not None else 0.0
        if encoder_lw > 0.1 and embed_error is not None:
            prediction_error = 0.4 * kg_error + 0.3 * tag_error + 0.3 * embed_error
        else:
            prediction_error = 0.6 * kg_error + 0.4 * tag_error
            embed_error = embed_error or 0.0

        emotional_surprise = self._compute_emotional_surprise(emotion_summary)

        gap_signal = 0.0
        if knowledge_summary:
            gap_count = knowledge_summary.get("gap_count", 0)
            gap_signal = min(1.0, gap_count / 5.0) * 0.1

        questions = self._generate_questions(episode_summary, novelty, prediction_error)

        if self.learned_world_model.available and ensemble_disagreement > 0:
            self.curiosity_score = (
                ensemble_disagreement * 0.50
                + emotional_surprise * 0.20
                + (len(questions) / 5.0) * 0.15
                + gap_signal * 0.15
            )
        else:
            self.curiosity_score = (
                novelty * 0.30
                + prediction_error * 0.30
                + emotional_surprise * 0.15
                + (len(questions) / 5.0) * 0.10
                + gap_signal * 0.15
            )
        self.curiosity_score = min(1.0, self.curiosity_score)

        generated_goals = self._spawn_goals(novelty, prediction_error, questions)

        self._previous_features = current_features
        self._prev_strategy = last_strategy

        return {
            "curiosity_score": round(self.curiosity_score, 3),
            "novelty": round(novelty, 3),
            "prediction_error": round(prediction_error, 3),
            "tag_prediction_error": round(tag_error, 3),
            "kg_prediction_error": round(kg_error, 3),
            "embed_prediction_error": round(embed_error, 3),
            "ensemble_disagreement": round(ensemble_disagreement, 3),
            "emotional_surprise": round(emotional_surprise, 3),
            "knowledge_gap_signal": round(gap_signal, 3),
            "questions": questions,
            "generated_goals": generated_goals,
            "drive_state": self._classify_drive(),
            "predicted_features": sorted(self._last_prediction)[:10],
            "actual_features": sorted(current_features)[:10],
            "world_model": self.world_model.stats(),
            "kg_world_model": self.kg_world_model.stats(),
            "embed_world_model": self.embed_world_model.stats(),
            "learned_world_model": self.learned_world_model.stats(),
        }

    def _compute_novelty(
        self, episode: Dict[str, Any], memories: list, current_embedding=None
    ) -> float:
        tags = set()
        if isinstance(episode.get("perception"), dict):
            tags.update(episode["perception"].get("tags", []))
        tags.update(episode.get("tags", []))

        if not tags:
            return 0.1

        for tag in tags:
            self.novelty_baseline[tag] = self.novelty_baseline.get(tag, 0) + 1

        if len(self.novelty_baseline) > self._MAX_NOVELTY_BASELINE:
            pruned = {k: v for k, v in self.novelty_baseline.items() if v >= 2}
            if len(pruned) > self._MAX_NOVELTY_BASELINE:
                sorted_items = sorted(pruned.items(), key=lambda x: x[1], reverse=True)
                pruned = dict(sorted_items[: self._MAX_NOVELTY_BASELINE])
            self.novelty_baseline = pruned

        avg_seen = sum(self.novelty_baseline.get(t, 0) for t in tags) / max(len(tags), 1)
        memory_overlap = 0
        for mem in memories:
            if isinstance(mem, dict):
                mem_tags = set(mem.get("tags", []))
            else:
                mem_tags = set(getattr(mem, "tags", None) or [])
            memory_overlap += len(tags.intersection(mem_tags))

        tag_novelty = 1.0 / (1.0 + avg_seen * 0.1 + memory_overlap * 0.2)
        tag_novelty = min(1.0, tag_novelty)

        encoder_lw = self._encoder.learned_weight if self._encoder is not None else 0.0
        if encoder_lw > 0.1 and current_embedding is not None and self._recent_embeddings:
            avg_sim = sum(
                EmbeddingWorldModel._cosine(current_embedding, e) for e in self._recent_embeddings
            ) / len(self._recent_embeddings)
            embed_novelty = max(0.0, 1.0 - avg_sim)
            novelty = tag_novelty * (1.0 - encoder_lw) + embed_novelty * encoder_lw
        else:
            novelty = tag_novelty

        if current_embedding is not None:
            self._recent_embeddings.append(current_embedding)
            if len(self._recent_embeddings) > 20:
                self._recent_embeddings = self._recent_embeddings[-20:]

        return min(1.0, novelty)

    def _compute_embedding_prediction_error(self, current_embedding) -> Optional[float]:
        if current_embedding is None or self._encoder is None:
            return None
        if self._prev_embedding is None:
            self._prev_embedding = current_embedding
            return 0.5
        predicted = self.embed_world_model.predict(self._prev_embedding)
        error = self.embed_world_model.compute_error(predicted, current_embedding)
        self.embed_world_model.learn(self._prev_embedding, current_embedding)
        self._prev_embedding = current_embedding
        return min(1.0, error)

    def _compute_kg_prediction_error(self, knowledge_summary: Optional[Dict[str, Any]]) -> float:
        if not knowledge_summary:
            return 0.5
        current_sig = KGWorldModel._signature_from_summary(knowledge_summary)
        if self._prev_kg_sig is None:
            self._prev_kg_sig = current_sig
            return 0.5
        predicted = self.kg_world_model.predict(self._prev_kg_sig)
        error = self.kg_world_model.compute_error(predicted, current_sig)
        self.kg_world_model.learn(self._prev_kg_sig, current_sig)
        self._prev_kg_sig = current_sig
        return min(1.0, error)

    def _compute_prediction_error(self, current_features: Set[str]) -> float:
        if not self._previous_features:
            self._last_prediction = set()
            return 0.5

        predicted = self.world_model.predict(self._previous_features)
        self._last_prediction = predicted
        error = self.world_model.compute_error(predicted, current_features)
        self.world_model.learn(self._previous_features, current_features)
        return min(1.0, error)

    def _compute_emotional_surprise(self, emotion: Dict[str, Any]) -> float:
        current = emotion.get("dominant", "neutral")
        intensity = emotion.get("intensity", 0.0)
        if current == "neutral":
            return 0.0
        return min(1.0, intensity * 0.8)

    def _generate_questions(
        self, episode: Dict[str, Any], novelty: float, error: float
    ) -> List[str]:
        questions: List[str] = []
        if novelty > 0.6:
            questions.append("What is this new pattern I am encountering?")
        if error > 0.5:
            questions.append("Why did my prediction fail?")
        if episode.get("drift_score", 0) > 0.3:
            questions.append("Why am I drifting from my identity?")
        if episode.get("collapse_risk", 0) > 0.3:
            questions.append("Which goals are at risk of collapse?")
        if self.world_model.mean_error > 0.6:
            questions.append("My world model is consistently wrong -- what am I missing?")

        self.open_questions = questions
        return questions

    def _spawn_goals(
        self, novelty: float, error: float, questions: List[str]
    ) -> List[Dict[str, Any]]:
        goals: List[Dict[str, Any]] = []
        if novelty > 0.7:
            goals.append(
                {
                    "goal_id": "curiosity_explore",
                    "description": "Explore novel pattern in recent input",
                    "priority": novelty,
                    "source": "curiosity",
                }
            )
        if error > 0.6:
            goals.append(
                {
                    "goal_id": "curiosity_correct_model",
                    "description": "Update internal model to reduce prediction error",
                    "priority": error,
                    "source": "curiosity",
                }
            )
        if len(questions) >= 3:
            goals.append(
                {
                    "goal_id": "curiosity_deep_reflect",
                    "description": "Initiate deep reflection cycle to resolve open questions",
                    "priority": 0.8,
                    "source": "curiosity",
                }
            )
        return goals

    def _classify_drive(self) -> str:
        if self.curiosity_score > 0.7:
            return "driven"
        elif self.curiosity_score > 0.4:
            return "curious"
        elif self.curiosity_score > 0.1:
            return "attentive"
        return "calm"

    def summarize(self) -> Dict[str, Any]:
        return {
            "curiosity_score": round(self.curiosity_score, 3),
            "drive_state": self._classify_drive(),
            "open_questions": self.open_questions,
            "novelty_vocabulary": len(self.novelty_baseline),
            "world_model": self.world_model.stats(),
            "kg_world_model": self.kg_world_model.stats(),
            "embed_world_model": self.embed_world_model.stats(),
            "learned_world_model": self.learned_world_model.stats(),
        }

    def reset(self):
        self.world_model = TagSetWorldModel()
        self.kg_world_model = KGWorldModel()
        self.embed_world_model = EmbeddingWorldModel()
        self.learned_world_model = LearnedWorldModel()
        self.novelty_baseline.clear()
        self.open_questions.clear()
        self.curiosity_score = 0.0
        self._previous_features.clear()
        self._last_prediction.clear()
        self._prev_kg_sig = None
        self._prev_embedding = None
        self._prev_strategy = "reflect"
        self._recent_embeddings.clear()
