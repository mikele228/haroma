from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
import os
from utils.module_base import ModuleBase
import hashlib
import time

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except (ImportError, OSError):
    _TORCH = False

try:
    from transformers import AutoTokenizer, AutoModel

    _TRANSFORMERS = True
except (ImportError, OSError):
    _TRANSFORMERS = False


class LearnedEmotionModel:
    """Associative model that learns context -> emotion mappings from experience."""

    _MAX_ASSOCIATIONS = 5000

    def __init__(self, min_samples: int = 5):
        self.associations: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.experience_count: int = 0
        self.min_samples = min_samples

    def _feature_hash(self, feature: str) -> str:
        return hashlib.md5(feature.encode()).hexdigest()[:8]

    def _extract_features(self, context: Dict[str, Any]) -> List[str]:
        features: List[str] = []
        for key in ("content", "text", "thought"):
            val = context.get(key)
            if isinstance(val, str):
                for word in val.lower().split():
                    w = word.strip(".,!?;:'\"()[]")
                    if len(w) > 2:
                        features.append(w)
        tags = context.get("tags", [])
        if isinstance(tags, list):
            features.extend(str(t).lower() for t in tags)
        return features

    def predict(self, context: Dict[str, Any]) -> Tuple[str, float]:
        if self.experience_count < self.min_samples:
            return "neutral", 0.0
        features = self._extract_features(context)
        if not features:
            return "neutral", 0.0
        emotion_totals: Dict[str, float] = defaultdict(float)
        matches = 0
        for f in features:
            fh = self._feature_hash(f)
            if fh in self.associations:
                matches += 1
                for emo, weight in self.associations[fh].items():
                    emotion_totals[emo] += weight
        if not emotion_totals or matches == 0:
            return "neutral", 0.0
        best = max(emotion_totals, key=emotion_totals.get)
        total = sum(emotion_totals.values())
        confidence = emotion_totals[best] / total if total > 0 else 0.0
        coverage = min(1.0, matches / max(len(features), 1))
        return best, confidence * coverage

    def learn(self, context: Dict[str, Any], emotion: str, intensity: float = 1.0):
        if emotion == "neutral":
            return
        features = self._extract_features(context)
        if not features:
            return
        for f in features:
            fh = self._feature_hash(f)
            self.associations[fh][emotion] += intensity
            for emo in list(self.associations[fh]):
                if emo != emotion:
                    self.associations[fh][emo] *= 0.95
        self.experience_count += 1
        if len(self.associations) > self._MAX_ASSOCIATIONS:
            scored = sorted(self.associations.items(), key=lambda kv: sum(kv[1].values()))
            keep = dict(scored[len(scored) - self._MAX_ASSOCIATIONS :])
            self.associations = defaultdict(lambda: defaultdict(float), keep)

    @property
    def learned_weight(self) -> float:
        return min(0.7, self.experience_count / 100.0)

    def stats(self) -> Dict[str, Any]:
        return {
            "experience_count": self.experience_count,
            "feature_count": len(self.associations),
            "learned_weight": round(self.learned_weight, 3),
        }


EMOTION_LABELS = [
    "joy",
    "wonder",
    "curiosity",
    "fear",
    "sadness",
    "anger",
    "resolve",
    "peace",
    "surprise",
]
_EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_LABELS)}
_N_EMOTIONS = len(EMOTION_LABELS)


class _TransformerEmotionModel:
    """Transformer-based emotion classifier on a frozen encoder backbone ([CLS]).

    Backbone defaults to DistilBERT; set env ``HAROMA_EMOTION_ENCODER`` to a Hugging Face
    model id (e.g. ``google/mobilebert-uncased``). Hidden size comes from ``config.hidden_size``
    (768 for DistilBERT, 512 for MobileBERT).

    This is **not** the chat LLM (GGUF / API) — only the emotion pathway.

    Three heads on top of [CLS]:
      - emotion classifier  (hidden -> 9 emotions)
      - valence regressor   (hidden -> 1)
      - arousal regressor   (hidden -> 1)

    The backbone is frozen; only the heads are trained online.
    """

    def __init__(self, model_name: Optional[str] = None):
        self._available = False
        self._tokenizer = None
        self._backbone = None
        self._emo_head = None
        self._val_head = None
        self._aro_head = None
        self._optimizer = None
        self._device = "cpu"
        self._train_steps = 0
        self._encoder_id = (
            (model_name or os.environ.get("HAROMA_EMOTION_ENCODER", "") or "").strip()
            or "distilbert-base-uncased"
        )

        if not (_TORCH and _TRANSFORMERS):
            return

        try:
            from engine.ModelCache import get_hf_encoder

            self._tokenizer, self._backbone = get_hf_encoder(self._encoder_id)
            self._backbone.eval()
            for p in self._backbone.parameters():
                if p.requires_grad:
                    p.requires_grad = False

            hidden = self._backbone.config.hidden_size
            self._emo_head = nn.Linear(hidden, _N_EMOTIONS)
            self._val_head = nn.Linear(hidden, 1)
            self._aro_head = nn.Linear(hidden, 1)

            nn.init.xavier_uniform_(self._emo_head.weight)
            nn.init.xavier_uniform_(self._val_head.weight)
            nn.init.xavier_uniform_(self._aro_head.weight)

            head_params = (
                list(self._emo_head.parameters())
                + list(self._val_head.parameters())
                + list(self._aro_head.parameters())
            )
            self._optimizer = torch.optim.Adam(head_params, lr=1e-3)
            self._available = True
        except Exception as exc:
            print(f"[EmotionEngine] Transformer init failed: {exc}")

    @property
    def available(self) -> bool:
        return self._available

    def predict(self, text: str) -> Tuple[str, float, float, float]:
        """Return (emotion_label, valence, arousal, confidence)."""
        if not self._available or not text.strip():
            return "neutral", 0.0, 0.1, 0.0

        try:
            tokens = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            with torch.no_grad():
                out = self._backbone(**tokens)
                cls = out.last_hidden_state[:, 0, :]  # [CLS]

                emo_logits = self._emo_head(cls)  # (1, 9)
                valence = torch.tanh(self._val_head(cls))  # (1, 1)
                arousal = torch.sigmoid(self._aro_head(cls))  # (1, 1)

            probs = torch.softmax(emo_logits, dim=-1).squeeze(0)
            top_idx = int(probs.argmax())
            confidence = float(probs[top_idx])
            label = EMOTION_LABELS[top_idx]
            return label, float(valence.item()), float(arousal.item()), confidence
        except Exception:
            return "neutral", 0.0, 0.1, 0.0

    def train_step(
        self,
        text: str,
        target_emotion: str,
        target_valence: float,
        target_arousal: float,
    ) -> float:
        """Online fine-tuning of heads only. Returns loss."""
        if not self._available:
            return 0.0
        emo_idx = _EMOTION_TO_IDX.get(target_emotion)
        if emo_idx is None:
            return 0.0

        try:
            tokens = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            with torch.no_grad():
                out = self._backbone(**tokens)
                cls = out.last_hidden_state[:, 0, :].detach()

            emo_logits = self._emo_head(cls)
            valence_pred = torch.tanh(self._val_head(cls))
            arousal_pred = torch.sigmoid(self._aro_head(cls))

            emo_target = torch.tensor([emo_idx], dtype=torch.long)
            val_target = torch.tensor([[target_valence]], dtype=torch.float32)
            aro_target = torch.tensor([[target_arousal]], dtype=torch.float32)

            loss_emo = nn.functional.cross_entropy(emo_logits, emo_target)
            loss_val = nn.functional.mse_loss(valence_pred, val_target)
            loss_aro = nn.functional.mse_loss(arousal_pred, aro_target)
            loss = loss_emo + loss_val + loss_aro

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            self._train_steps += 1
            return float(loss.item())
        except Exception:
            return 0.0

    @property
    def learned_weight(self) -> float:
        if not self._available:
            return 0.0
        return min(0.85, self._train_steps / 200.0)

    def stats(self) -> Dict[str, Any]:
        return {
            "available": self._available,
            "encoder": self._encoder_id,
            "train_steps": self._train_steps,
            "learned_weight": round(self.learned_weight, 3),
        }


EMOTION_KEYWORDS = {
    "joy": [
        "happy",
        "joy",
        "delight",
        "wonderful",
        "beautiful",
        "love",
        "warm",
        "bright",
        "light",
        "success",
    ],
    "wonder": [
        "wonder",
        "awe",
        "amazing",
        "incredible",
        "mystery",
        "discover",
        "revelation",
        "realize",
    ],
    "curiosity": [
        "curious",
        "strange",
        "novel",
        "unknown",
        "question",
        "why",
        "how",
        "what",
        "explore",
    ],
    "fear": ["fear", "afraid", "dark", "danger", "threat", "terror", "dread", "anxious"],
    "sadness": ["sad", "loss", "grief", "alone", "empty", "gone", "miss", "tears"],
    "anger": ["anger", "angry", "furious", "injustice", "wrong", "rage", "betray"],
    "resolve": ["resolve", "determined", "commit", "oath", "protect", "defend", "fight", "persist"],
    "peace": ["peace", "calm", "serene", "quiet", "still", "rest", "harmony"],
    "surprise": ["surprise", "unexpected", "sudden", "shock", "emerge", "appear"],
}

VALENCE_MAP = {
    "joy": 0.9,
    "wonder": 0.8,
    "curiosity": 0.6,
    "peace": 0.7,
    "surprise": 0.3,
    "resolve": 0.5,
    "sadness": -0.7,
    "fear": -0.6,
    "anger": -0.5,
    "neutral": 0.0,
}

AROUSAL_MAP = {
    "joy": 0.7,
    "wonder": 0.6,
    "curiosity": 0.5,
    "peace": 0.1,
    "surprise": 0.8,
    "resolve": 0.6,
    "sadness": 0.3,
    "fear": 0.8,
    "anger": 0.9,
    "neutral": 0.1,
}


class EmotionEngine(ModuleBase):
    """Derives emotional state from input content using a dual system:
    learned associations (improves with experience) + keyword fallback."""

    _MAX_HISTORY = 2000

    def __init__(self, memory_tree=None, agent_id: str = "root"):
        super().__init__("EmotionEngine")
        self.agent_id = agent_id
        self.memory_tree = memory_tree
        self.current_emotion: str = "neutral"
        self.intensity: float = 0.0
        self.valence: float = 0.0
        self.arousal: float = 0.0
        self.history: List[Dict[str, Any]] = []
        self._staleness_counter: int = 0
        self.learned_model = LearnedEmotionModel()
        self.transformer_model = _TransformerEmotionModel()
        self._last_context: Optional[Dict[str, Any]] = None

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.ingest(data)
        tags = data.get("tags", [])
        reflected_tags = [t for t in tags if t.startswith("emotion:")]
        return {
            "emotion": self.current_emotion,
            "intensity": self.intensity,
            "valence": self.valence,
            "arousal": self.arousal,
            "reflected_tags": reflected_tags or tags,
            "agent_id": self.agent_id,
        }

    def ingest(self, data: Dict[str, Any]):
        self._last_context = data

        if "label" in data and data["label"] != "neutral":
            self.update_emotion(data["label"], data.get("intensity", 0.7), data)
            return

        text = self._extract_text(data)
        kw_emotion, kw_confidence = self._detect_emotion(text)
        learned_emotion, learned_confidence = self.learned_model.predict(data)

        # Transformer-based prediction (Upgrade 2)
        tf_emotion, tf_valence, tf_arousal, tf_confidence = ("neutral", 0.0, 0.1, 0.0)
        if self.transformer_model.available and text.strip():
            tf_emotion, tf_valence, tf_arousal, tf_confidence = self.transformer_model.predict(text)

        tw = self.transformer_model.learned_weight
        lw = self.learned_model.learned_weight

        if tw > 0 and tf_emotion != "neutral" and tf_confidence > 0.15:
            kw_weight = max(0.0, 1.0 - tw - lw * 0.3)
            assoc_weight = lw * (1.0 - tw) * 0.5
            final_emotion = tf_emotion
            final_intensity = (
                tf_confidence * tw + learned_confidence * assoc_weight + kw_confidence * kw_weight
            )
        elif lw > 0 and learned_emotion != "neutral" and learned_confidence > 0.2:
            if kw_emotion != "neutral":
                if learned_confidence * lw > kw_confidence * (1 - lw):
                    final_emotion = learned_emotion
                    final_intensity = learned_confidence * lw + kw_confidence * (1 - lw)
                else:
                    final_emotion = kw_emotion
                    final_intensity = kw_confidence * (1 - lw) + learned_confidence * lw
            else:
                final_emotion = learned_emotion
                final_intensity = learned_confidence * lw
        elif kw_emotion != "neutral":
            final_emotion = kw_emotion
            final_intensity = kw_confidence
        else:
            final_emotion = "neutral"
            final_intensity = 0.0

        if final_emotion != "neutral":
            blended_intensity = max(self.intensity * 0.3, final_intensity)
            self.update_emotion(
                final_emotion,
                blended_intensity,
                data,
                predicted_valence=tf_valence if tw > 0 else None,
                predicted_arousal=tf_arousal if tw > 0 else None,
            )
        else:
            self._staleness_counter += 1
            if self._staleness_counter >= 3 and self.current_emotion != "neutral":
                self.current_emotion = "neutral"
                self.intensity = 0.0
                self.valence = 0.0
                self.arousal = 0.0

    def _extract_text(self, data: Dict[str, Any]) -> str:
        parts = []
        for key in ("content", "text", "thought"):
            val = data.get(key)
            if isinstance(val, str):
                parts.append(val)
            elif isinstance(val, dict):
                parts.append(str(val.get("text", "")))
        tags = data.get("tags", [])
        if isinstance(tags, list):
            parts.extend(str(t) for t in tags)
        return " ".join(parts).lower()

    def _detect_emotion(self, text: str) -> tuple:
        if not text.strip():
            return "neutral", 0.0

        scores: Dict[str, float] = {}
        for emotion, keywords in EMOTION_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text)
            if hits > 0:
                scores[emotion] = hits / len(keywords)

        if not scores:
            return "neutral", 0.0

        best = max(scores, key=scores.get)
        return best, min(1.0, scores[best] * 3.0)

    def update_emotion(
        self,
        label: str,
        intensity: float,
        context: Dict[str, Any] = None,
        predicted_valence: Optional[float] = None,
        predicted_arousal: Optional[float] = None,
    ):
        self.current_emotion = label
        self.intensity = min(1.0, max(0.0, intensity))
        if predicted_valence is not None:
            self.valence = float(predicted_valence)
        else:
            self.valence = VALENCE_MAP.get(label, 0.0)
        if predicted_arousal is not None:
            self.arousal = float(predicted_arousal)
        else:
            self.arousal = AROUSAL_MAP.get(label, 0.1)
        self._staleness_counter = 0
        self.history.append(
            {
                "timestamp": time.time(),
                "emotion": label,
                "intensity": self.intensity,
                "valence": self.valence,
                "arousal": self.arousal,
            }
        )
        if len(self.history) > self._MAX_HISTORY:
            self.history = self.history[-self._MAX_HISTORY :]

    def apply_decay(self, decay: float = 0.01):
        self.intensity = max(0.0, self.intensity - decay)
        self._staleness_counter += 1
        if self.intensity <= 0.0 and self.current_emotion != "neutral":
            self.current_emotion = "neutral"
            self.valence = 0.0
            self.arousal = 0.0

    def dominant_emotion(self) -> str:
        return self.current_emotion

    def compute_alignment(self, projected_emotions: Dict[str, Any]) -> float:
        if not projected_emotions:
            return 0.5
        values = [v for v in projected_emotions.values() if isinstance(v, (int, float))]
        if not values:
            return 0.5
        return min(1.0, sum(values) / len(values))

    def summarize(self) -> Dict[str, Any]:
        return {
            "dominant": self.current_emotion,
            "current_emotion": self.current_emotion,
            "intensity": round(self.intensity, 3),
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "staleness": min(1.0, self._staleness_counter / 20.0),
            "history_length": len(self.history),
            "learned_model": self.learned_model.stats(),
            "transformer_model": self.transformer_model.stats(),
        }

    def learn_from_cycle(
        self, context_data: Optional[Dict[str, Any]] = None, confirmed_emotion: Optional[str] = None
    ):
        ctx = context_data or self._last_context
        emo = confirmed_emotion or self.current_emotion
        if ctx and emo and emo != "neutral":
            self.learned_model.learn(ctx, emo, self.intensity)
            if self.transformer_model.available:
                text = self._extract_text(ctx)
                if text.strip():
                    self.transformer_model.train_step(text, emo, self.valence, self.arousal)

    def reset(self):
        self.current_emotion = "neutral"
        self.intensity = 0.0
        self.valence = 0.0
        self.arousal = 0.0
        self._staleness_counter = 0
        self.history.clear()
