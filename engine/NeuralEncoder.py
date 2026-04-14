"""
NeuralEncoder — Semantic embedding backbone for HaromaX6 (Phase 16).

Wraps a frozen pretrained sentence encoder (default **all-MiniLM-L6-v2**, 384-d hidden)
with a trainable projection to the system's 256-d embedding space.
Set env ``HAROMA_SEMANTIC_ENCODER`` to any Hugging Face ``AutoModel`` id used for
sentence embeddings (hidden size is read from ``config.hidden_size``).

Falls back to the original hash-bucket encoder when the pretrained
model is unavailable.

If PyTorch is unavailable the class degrades to a no-op stub.
"""

from __future__ import annotations

import os

from typing import Dict, Any, List, Optional
import hashlib
import re

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

_PRETRAINED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_PRETRAINED_DIM = 384

try:
    from transformers import AutoTokenizer, AutoModel

    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t for t in _SPLIT_RE.split(text.lower()) if len(t) > 1]


# ======================================================================
# Fallback: hash-bucket encoder (original Phase 8)
# ======================================================================


class _EncoderModel(nn.Module if _TORCH_AVAILABLE else object):
    """Token-embedding table + average pool + linear projector."""

    def __init__(self, vocab_size: int = 8192, embed_dim: int = 256):
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.projector = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.numel() == 0:
            return torch.zeros(self.projector.out_features)
        emb = self.embedding(token_ids).mean(dim=0)
        proj = torch.tanh(self.projector(emb))
        return F.normalize(proj, dim=0)


# ======================================================================
# Pretrained backbone + projection head (Phase 16)
# ======================================================================

if _TORCH_AVAILABLE:

    class _ProjectionHead(nn.Module):
        """Trainable Linear(384, 256) + LayerNorm + L2 normalize."""

        def __init__(self, in_dim: int = _PRETRAINED_DIM, out_dim: int = 256):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            self.ln = nn.LayerNorm(out_dim)
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.normalize(self.ln(self.linear(x)), dim=-1)


class _PretrainedBackbone:
    """Loads the frozen sentence encoder and provides encode()."""

    def __init__(self, model_id: Optional[str] = None):
        self.available = False
        self._tokenizer = None
        self._model = None
        self._model_id = (
            (model_id or os.environ.get("HAROMA_SEMANTIC_ENCODER", "") or "").strip()
            or _PRETRAINED_MODEL_NAME
        )
        self._hidden_dim = _PRETRAINED_DIM

        if not (_TORCH_AVAILABLE and _HF_AVAILABLE):
            return
        try:
            from engine.ModelCache import get_hf_semantic_encoder

            self._tokenizer, self._model = get_hf_semantic_encoder(self._model_id)
            self._model.eval()
            for p in self._model.parameters():
                if p.requires_grad:
                    p.requires_grad = False
            hid = getattr(self._model.config, "hidden_size", None)
            if isinstance(hid, int) and hid > 0:
                self._hidden_dim = hid
            _fab = _get_fabric()
            if _fab:
                self._model = _fab.register("pretrained_backbone", self._model)
            self.available = True
        except Exception:
            self.available = False

    def encode(self, texts: List[str]) -> "torch.Tensor":
        """Encode a batch of texts -> (batch, 384) tensor."""
        _fab = _get_fabric()
        encoded = self._tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        if _fab and _fab.device is not None:
            encoded = {k: v.to(_fab.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self._model(**encoded)
        token_embeddings = output.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        return F.normalize(pooled, dim=-1)


# ======================================================================
# Public interface
# ======================================================================


class NeuralEncoder:
    """Public interface consumed by Memory, Curiosity, Dream, and control.

    The ``force_mode`` parameter lets the ResourceAdaptiveConfig force
    the encoder into a specific mode regardless of library availability:
      - ``"hash_bucket"``  — lightweight fallback (tier 0 / embedded)
      - ``"pretrained"``   — sentence-transformer backbone (tier 1+)
      - ``None``           — auto-detect (original behavior)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        vocab_size: int = 8192,
        force_mode: Optional[str] = None,
        semantic_model_id: Optional[str] = None,
    ):
        self._embed_dim = embed_dim
        self._vocab_size = vocab_size
        self._train_steps: int = 0
        self._available = _TORCH_AVAILABLE
        self._pretrained = False

        self._backbone: Optional[_PretrainedBackbone] = None
        self._projection: Optional[Any] = None
        self._fallback_model: Optional[Any] = None
        self._optimizer: Optional[Any] = None

        if not self._available:
            return

        use_pretrained = force_mode != "hash_bucket"
        if use_pretrained:
            backbone = _PretrainedBackbone(model_id=semantic_model_id)
        else:
            backbone = None

        if backbone and backbone.available:
            self._backbone = backbone
            self._pretrained = True
            _in_dim = backbone._hidden_dim if backbone._hidden_dim else _PRETRAINED_DIM
            self._projection = _ProjectionHead(_in_dim, embed_dim)
            _fab = _get_fabric()
            if _fab:
                self._projection = _fab.register("encoder_projection", self._projection)
            self._optimizer = torch.optim.Adam(self._projection.parameters(), lr=5e-4)
        else:
            self._fallback_model = _EncoderModel(vocab_size, embed_dim)
            _fab = _get_fabric()
            if _fab:
                self._fallback_model = _fab.register("encoder_model", self._fallback_model)
            self._fallback_model.eval()
            self._optimizer = torch.optim.Adam(self._fallback_model.parameters(), lr=1e-3)

    @property
    def available(self) -> bool:
        return self._available

    @property
    def learned_weight(self) -> float:
        if not self._available:
            return 0.0
        if self._pretrained:
            return min(0.8, 0.3 + self._train_steps / 200.0 * 0.5)
        return min(0.8, self._train_steps / 200.0)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def semantic_backbone_id(self) -> str:
        """Hugging Face model id for the frozen backbone, or ``hash-bucket`` / empty."""
        if self._pretrained and self._backbone is not None:
            return str(getattr(self._backbone, "_model_id", "") or "").strip()
        if self._fallback_model is not None:
            return "hash-bucket"
        return ""

    def _to_ids(self, text: str) -> Optional["torch.Tensor"]:
        """Hash-bucket tokenization (fallback path only)."""
        if not self._available:
            return None
        tokens = _tokenize(text)
        _fab = _get_fabric()
        if not tokens:
            return _fab.tensor([], dtype=torch.long) if _fab else torch.tensor([], dtype=torch.long)
        ids = [int(hashlib.md5(t.encode()).hexdigest(), 16) % self._vocab_size for t in tokens]
        return _fab.tensor(ids, dtype=torch.long) if _fab else torch.tensor(ids, dtype=torch.long)

    def encode(self, text: str) -> Optional["np.ndarray"]:
        """Return L2-normalized embedding of shape (embed_dim,) or None."""
        if not self._available or not text:
            return None

        if self._pretrained and self._backbone is not None:
            with torch.no_grad():
                backbone_out = self._backbone.encode([text])
                projected = self._projection(backbone_out)
            return projected.squeeze(0).cpu().numpy().copy()

        ids = self._to_ids(text)
        if ids is None:
            return None
        with torch.no_grad():
            vec = self._fallback_model(ids)
        return vec.cpu().numpy().copy()

    def encode_batch(self, texts: List[str]) -> Optional["np.ndarray"]:
        """Return (n, embed_dim) matrix or None."""
        if not self._available or not texts:
            return None

        if self._pretrained and self._backbone is not None:
            with torch.no_grad():
                backbone_out = self._backbone.encode(texts)
                projected = self._projection(backbone_out)
            return projected.cpu().numpy().copy()

        rows = []
        for t in texts:
            v = self.encode(t)
            if v is not None:
                rows.append(v)
            else:
                rows.append(np.zeros(self._embed_dim))
        return np.stack(rows)

    def train_step(
        self, anchor_text: str, positive_texts: List[str], negative_texts: List[str]
    ) -> float:
        """Triplet-margin contrastive step. Returns loss value."""
        if not self._available:
            return 0.0
        if not anchor_text or not positive_texts or not negative_texts:
            return 0.0

        _fab = _get_fabric()

        if self._pretrained and self._backbone is not None:
            return self._train_pretrained(anchor_text, positive_texts, negative_texts, _fab)

        return self._train_fallback(anchor_text, positive_texts, negative_texts, _fab)

    def _train_pretrained(
        self, anchor: str, positives: List[str], negatives: List[str], _fab
    ) -> float:
        """Train projection head with frozen backbone."""
        all_texts = [anchor] + positives[:3] + negatives[:3]
        with torch.no_grad():
            backbone_out = self._backbone.encode(all_texts)

        self._projection.train()
        projected = self._projection(backbone_out)

        anchor_vec = projected[0]
        n_pos = min(3, len(positives))
        pos_vecs = projected[1 : 1 + n_pos]
        neg_vecs = projected[1 + n_pos :]

        total_loss = _fab.tensor(0.0) if _fab else torch.tensor(0.0)
        n_triplets = 0
        margin = 0.2

        for pv in pos_vecs:
            for nv in neg_vecs:
                d_pos = 1.0 - F.cosine_similarity(anchor_vec.unsqueeze(0), pv.unsqueeze(0))
                d_neg = 1.0 - F.cosine_similarity(anchor_vec.unsqueeze(0), nv.unsqueeze(0))
                loss = F.relu(d_pos - d_neg + margin)
                total_loss = total_loss + loss.squeeze()
                n_triplets += 1

        if n_triplets == 0:
            self._projection.eval()
            return 0.0

        total_loss = total_loss / n_triplets
        self._optimizer.zero_grad()
        if _fab:
            _fab.scale_loss(total_loss).backward()
            nn.utils.clip_grad_norm_(self._projection.parameters(), 1.0)
            _fab.scaler_step(self._optimizer)
            _fab.scaler_update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(self._projection.parameters(), 1.0)
            self._optimizer.step()

        self._train_steps += 1
        self._projection.eval()
        return float(total_loss.item())

    def _train_fallback(
        self, anchor: str, positives: List[str], negatives: List[str], _fab
    ) -> float:
        """Train hash-bucket encoder (original path)."""
        self._fallback_model.train()
        anchor_ids = self._to_ids(anchor)
        if anchor_ids is None or anchor_ids.numel() == 0:
            self._fallback_model.eval()
            return 0.0

        anchor_vec = self._fallback_model(anchor_ids)
        total_loss = _fab.tensor(0.0) if _fab else torch.tensor(0.0)
        n_triplets = 0
        margin = 0.2

        for pos_text in positives[:3]:
            pos_ids = self._to_ids(pos_text)
            if pos_ids is None or pos_ids.numel() == 0:
                continue
            pos_vec = self._fallback_model(pos_ids)
            for neg_text in negatives[:3]:
                neg_ids = self._to_ids(neg_text)
                if neg_ids is None or neg_ids.numel() == 0:
                    continue
                neg_vec = self._fallback_model(neg_ids)
                d_pos = 1.0 - F.cosine_similarity(anchor_vec.unsqueeze(0), pos_vec.unsqueeze(0))
                d_neg = 1.0 - F.cosine_similarity(anchor_vec.unsqueeze(0), neg_vec.unsqueeze(0))
                loss = F.relu(d_pos - d_neg + margin)
                total_loss = total_loss + loss.squeeze()
                n_triplets += 1

        if n_triplets == 0:
            self._fallback_model.eval()
            return 0.0

        total_loss = total_loss / n_triplets
        self._optimizer.zero_grad()
        if _fab:
            _fab.scale_loss(total_loss).backward()
            nn.utils.clip_grad_norm_(self._fallback_model.parameters(), 1.0)
            _fab.scaler_step(self._optimizer)
            _fab.scaler_update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(self._fallback_model.parameters(), 1.0)
            self._optimizer.step()

        self._train_steps += 1
        self._fallback_model.eval()
        return float(total_loss.item())

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "train_steps": self._train_steps,
            "embed_dim": self._embed_dim,
            "vocab_size": self._vocab_size,
            "pretrained": self._pretrained,
        }
        if self._available and self._pretrained and self._projection is not None:
            data["projection_state"] = {
                k: v.tolist() for k, v in self._projection.state_dict().items()
            }
        elif self._available and self._fallback_model is not None:
            data["model_state"] = {
                k: v.tolist() for k, v in self._fallback_model.state_dict().items()
            }
        return data

    def from_dict(self, data: Dict[str, Any]):
        saved_steps = data.get("train_steps", 0)
        saved_dim = data.get("embed_dim", self._embed_dim)
        saved_vocab = data.get("vocab_size", self._vocab_size)

        if not self._available:
            self._train_steps = saved_steps
            return

        saved_pretrained = data.get("pretrained", False)
        proj_state = data.get("projection_state")
        if proj_state and saved_pretrained and not self._pretrained:
            print(
                f"[NeuralEncoder] snapshot was pretrained but current instance is not; "
                f"projection weights will be skipped",
                flush=True,
            )
        if proj_state and self._pretrained and self._projection is not None:
            try:
                restored = {k: torch.tensor(v) for k, v in proj_state.items()}
                self._projection.load_state_dict(restored)
                self._projection.eval()
                self._train_steps = saved_steps
            except Exception as exc:
                print(
                    f"[NeuralEncoder] projection restore failed ({exc}); "
                    f"keeping train_steps={self._train_steps}",
                    flush=True,
                )
            return

        model_state = data.get("model_state")
        if model_state and self._fallback_model is not None:
            if saved_dim != self._embed_dim or saved_vocab != self._vocab_size:
                self._embed_dim = saved_dim
                self._vocab_size = saved_vocab
                self._fallback_model = _EncoderModel(saved_vocab, saved_dim)
                _fab = _get_fabric()
                if _fab:
                    self._fallback_model = _fab.register("encoder_model", self._fallback_model)
                self._optimizer = torch.optim.Adam(self._fallback_model.parameters(), lr=1e-3)
            try:
                restored = {k: torch.tensor(v) for k, v in model_state.items()}
                self._fallback_model.load_state_dict(restored)
                self._fallback_model.eval()
                self._train_steps = saved_steps
            except Exception as exc:
                print(
                    f"[NeuralEncoder] fallback model restore failed ({exc}); "
                    f"resetting train_steps to 0",
                    flush=True,
                )
                self._train_steps = 0
        else:
            self._train_steps = saved_steps

    def stats(self) -> Dict[str, Any]:
        backbone_label = "hash-bucket"
        backbone_hidden = None
        if self._pretrained and self._backbone is not None:
            backbone_label = getattr(self._backbone, "_model_id", _PRETRAINED_MODEL_NAME)
            backbone_hidden = getattr(self._backbone, "_hidden_dim", None)
        return {
            "available": self._available,
            "pretrained": self._pretrained,
            "train_steps": self._train_steps,
            "learned_weight": round(self.learned_weight, 3),
            "embed_dim": self._embed_dim,
            "vocab_size": self._vocab_size,
            "backbone": backbone_label,
            "backbone_hidden_dim": backbone_hidden,
        }
