"""
CognitiveBackbone — Transformer Cognitive State Encoder for HaromaX6 (Phase 16).

Produces a single 512-d state vector z_t from a standardized 64-d snapshot
of the cycle's cognitive state using a transformer with self-attention over
cognitive features and cross-attention to content embedding.

Two backbone architectures are available (selected via ``use_mla=True``):

  * **Classic** (default): ``nn.TransformerEncoder`` with LayerNorm + standard FFN.
  * **MLA-lite**: Clean-room implementation inspired by DeepSeek-V3 Multi-Head
    Latent Attention (MLA).  Uses low-rank KV compression, Pre-LN with RMSNorm,
    and SwiGLU feed-forward networks at HaromaX6 scale (d_model=256, 8 heads,
    6 layers).  This is *not* a port of DeepSeek weights or `kernel.py`; it
    adapts the *structural ideas* described in the DeepSeek-V3 technical report.

    Reference: DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv:2412.19437, 2024.
    Local reference code: C:\\Project\\DeepSeek-V3\\inference\\model.py (MLA class).
    No checkpoint compatibility with upstream DeepSeek models is claimed.

Training is multi-task:
  1. Outcome regression (predict cycle outcome score from z_t)
  2. Next-state prediction (predict next 64-d snapshot from z_t)
  3. Contrastive alignment with content embeddings

Gracefully degrades to None output when PyTorch is unavailable.
Falls back to MLP when old state is loaded.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import math
import threading

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH = True
except (ImportError, OSError):
    _TORCH = False

from engine.ComputeFabric import get_fabric as _get_fabric

_SNAPSHOT_DIM = 64
_Z_DIM = 512
_CONTENT_DIM = 256
_D_MODEL = 256
_N_HEADS = 8
_N_LAYERS = 6
_FF_DIM = 512

_STRATEGY_LABELS = ["reflect", "inquire", "express", "explore", "support"]
_INTENT_LABELS = ["utterance", "command", "greeting"]

_KV_LORA_RANK = 64


if _TORCH:
    # ==================================================================
    # Shared primitives (used by MLA-lite backbone)
    # ==================================================================

    class _RMSNorm(nn.Module):
        """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
            return (x.float() / rms).type_as(x) * self.weight

    class _SwiGLUFFN(nn.Module):
        """SwiGLU feed-forward: gate = SiLU(W1 x) * W3 x, out = W2 gate."""

        def __init__(self, dim: int, hidden_dim: int):
            super().__init__()
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    # ==================================================================
    # MLA-lite attention (DeepSeek-V3 inspired, clean-room at Haroma scale)
    # ==================================================================

    class _MLALiteAttention(nn.Module):
        """Multi-Head Latent Attention at small scale.

        Compresses KV into a shared low-rank latent, then expands to per-head
        K and V.  Q is projected directly.  Inspired by DeepSeek-V3's MLA
        (arXiv:2412.19437) — no RoPE split, no FP8, no distributed parallel.
        """

        def __init__(self, dim: int, n_heads: int, kv_lora_rank: int):
            super().__init__()
            self.dim = dim
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            self.kv_lora_rank = kv_lora_rank
            self.scale = self.head_dim**-0.5

            self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
            self.wkv_a = nn.Linear(dim, kv_lora_rank, bias=False)
            self.kv_norm = _RMSNorm(kv_lora_rank)
            self.wkv_b = nn.Linear(
                kv_lora_rank,
                n_heads * (self.head_dim + self.head_dim),
                bias=False,
            )
            self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz, seq_len, _ = x.size()

            q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)

            kv_compressed = self.wkv_a(x)
            kv_compressed = self.kv_norm(kv_compressed)
            kv = self.wkv_b(kv_compressed)
            kv = kv.view(bsz, seq_len, self.n_heads, self.head_dim * 2)
            k, v = kv.split(self.head_dim, dim=-1)

            scores = torch.einsum("bshd,bthd->bsht", q, k) * self.scale
            attn = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
            out = torch.einsum("bsht,bthd->bshd", attn, v)

            out = out.reshape(bsz, seq_len, self.n_heads * self.head_dim)
            return self.wo(out)

    class _MLALiteBlock(nn.Module):
        """Pre-LN transformer block: RMSNorm → MLA-lite → residual → RMSNorm → SwiGLU → residual."""

        def __init__(self, dim: int, n_heads: int, ff_dim: int, kv_lora_rank: int):
            super().__init__()
            self.attn_norm = _RMSNorm(dim)
            self.attn = _MLALiteAttention(dim, n_heads, kv_lora_rank)
            self.ffn_norm = _RMSNorm(dim)
            self.ffn = _SwiGLUFFN(dim, ff_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.attn_norm(x))
            x = x + self.ffn(self.ffn_norm(x))
            return x

    # ==================================================================
    # Full MLA-lite backbone (same I/O contract as _TransformerBackbone)
    # ==================================================================

    class _MLALiteBackbone(nn.Module):
        """MLA-lite cognitive state encoder.

        Same input/output contract as _TransformerBackbone:
        64 scalar tokens → self-attention stack → optional cross-attention
        on content → 512-d z_t.
        """

        def __init__(self):
            super().__init__()
            self._scalar_proj = nn.Linear(1, _D_MODEL)
            self._pos_embed = nn.Parameter(torch.randn(max(_SNAPSHOT_DIM, 48), _D_MODEL) * 0.02)

            self._layers = nn.ModuleList(
                [
                    _MLALiteBlock(_D_MODEL, _N_HEADS, _FF_DIM, _KV_LORA_RANK)
                    for _ in range(_N_LAYERS)
                ]
            )
            self._final_norm = _RMSNorm(_D_MODEL)

            self._cross_attn = nn.MultiheadAttention(
                embed_dim=_D_MODEL, num_heads=_N_HEADS, batch_first=True
            )
            self._content_proj = nn.Linear(_CONTENT_DIM, _D_MODEL)

            self._output_proj = nn.Linear(_D_MODEL, _Z_DIM)
            self._output_ln = _RMSNorm(_Z_DIM)

        def forward(
            self, snapshot: torch.Tensor, content_embedding: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            if snapshot.dim() == 1:
                snapshot = snapshot.unsqueeze(0)
            seq_len = snapshot.shape[1]

            tokens = self._scalar_proj(snapshot.unsqueeze(-1))
            tokens = tokens + self._pos_embed[:seq_len].unsqueeze(0)

            for layer in self._layers:
                tokens = layer(tokens)
            tokens = self._final_norm(tokens)

            if content_embedding is not None:
                if content_embedding.dim() == 1:
                    content_embedding = content_embedding.unsqueeze(0)
                content_kv = self._content_proj(content_embedding).unsqueeze(1)
                tokens, _ = self._cross_attn(tokens, content_kv, content_kv)

            pooled = tokens.mean(dim=1)
            z_t = self._output_ln(self._output_proj(pooled))
            return z_t

    # ==================================================================
    # Classic backbone (original Phase 16)
    # ==================================================================

    class _TransformerBackbone(nn.Module):
        """64 scalar tokens -> self-attention -> cross-attention on content -> 512-d z_t."""

        def __init__(self):
            super().__init__()
            self._scalar_proj = nn.Linear(1, _D_MODEL)
            self._pos_embed = nn.Parameter(torch.randn(max(_SNAPSHOT_DIM, 48), _D_MODEL) * 0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=_D_MODEL,
                nhead=_N_HEADS,
                dim_feedforward=_FF_DIM,
                dropout=0.1,
                batch_first=True,
            )
            self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=_N_LAYERS)

            self._cross_attn = nn.MultiheadAttention(
                embed_dim=_D_MODEL, num_heads=_N_HEADS, batch_first=True
            )
            self._content_proj = nn.Linear(_CONTENT_DIM, _D_MODEL)

            self._output_proj = nn.Linear(_D_MODEL, _Z_DIM)
            self._output_ln = nn.LayerNorm(_Z_DIM)

        def forward(
            self, snapshot: torch.Tensor, content_embedding: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            if snapshot.dim() == 1:
                snapshot = snapshot.unsqueeze(0)
            batch_size = snapshot.shape[0]
            seq_len = snapshot.shape[1]

            tokens = self._scalar_proj(snapshot.unsqueeze(-1))
            tokens = tokens + self._pos_embed[:seq_len].unsqueeze(0)

            tokens = self._encoder(tokens)

            if content_embedding is not None:
                if content_embedding.dim() == 1:
                    content_embedding = content_embedding.unsqueeze(0)
                content_kv = self._content_proj(content_embedding).unsqueeze(1)
                tokens, _ = self._cross_attn(tokens, content_kv, content_kv)

            pooled = tokens.mean(dim=1)
            z_t = self._output_ln(self._output_proj(pooled))
            return z_t

    class _OutcomeHead(nn.Module):
        """Predicts scalar outcome from z_t."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(_Z_DIM, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.net(z).squeeze(-1)

    class _NextStateHead(nn.Module):
        """Predicts next 64-d snapshot from z_t."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(_Z_DIM, 64),
                nn.ReLU(),
                nn.Linear(64, _SNAPSHOT_DIM),
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.net(z)


def build_snapshot(
    *,
    content_embedding: Optional[List[float]],
    embed_dim: int,
    valence: float,
    arousal: float,
    intensity: float,
    curiosity_score: float,
    prediction_error: float,
    dominant_drive_level: float,
    wm_load: float,
    outcome_prev: float,
    has_external: float,
    cycle_count: int,
    n_goals: int,
    kg_entity_count: int,
    self_surprise: float,
    emotion_streak: int,
    drift_score: float,
    strategy: str,
    intent: str,
    drive_levels: List[float],
    reasoning_depth: int,
    cf_depth: int,
    imagination_quality: float,
    metacog_prediction: float,
    steps_run_ratio: float,
    plan_active: bool,
    env_tick: int,
) -> List[float]:
    """Build the standardised 64-d cognitive snapshot."""
    emb = content_embedding or [0.0] * embed_dim
    stride = max(1, len(emb) // 8)
    compressed = [emb[i * stride] if i * stride < len(emb) else 0.0 for i in range(8)]

    vec = list(compressed)
    vec.append(valence)
    vec.append(arousal)
    vec.append(intensity)
    vec.append(curiosity_score)
    vec.append(prediction_error)
    vec.append(dominant_drive_level)
    vec.append(wm_load)
    vec.append(outcome_prev)
    vec.append(has_external)
    vec.append(min(1.0, math.log1p(cycle_count) / 10.0))
    vec.append(min(1.0, n_goals / 10.0))
    vec.append(min(1.0, kg_entity_count / 100.0))
    vec.append(self_surprise)
    vec.append(min(1.0, emotion_streak / 10.0))
    vec.append(drift_score)

    strat_oh = [0.0] * len(_STRATEGY_LABELS)
    if strategy in _STRATEGY_LABELS:
        strat_oh[_STRATEGY_LABELS.index(strategy)] = 1.0
    vec.extend(strat_oh)

    intent_oh = [0.0] * len(_INTENT_LABELS)
    _ik = (intent or "").strip().lower()
    if _ik in (
        "interrogative",
        "question",
        "declarative",
        "statement",
        "inquiry",
        "none",
        "exclamatory",
    ):
        _ik = "utterance"
    if _ik in ("imperative",):
        _ik = "command"
    if _ik in _INTENT_LABELS:
        intent_oh[_INTENT_LABELS.index(_ik)] = 1.0
    vec.extend(intent_oh)

    dl = (drive_levels + [0.0] * 5)[:5]
    vec.extend(dl)

    vec.append(min(1.0, reasoning_depth / 10.0))
    vec.append(min(1.0, cf_depth / 3.0))
    vec.append(imagination_quality)
    vec.append(metacog_prediction)
    vec.append(steps_run_ratio)
    vec.append(1.0 if plan_active else 0.0)
    vec.append(min(1.0, env_tick / 200.0))

    while len(vec) < _SNAPSHOT_DIM:
        vec.append(0.0)
    return vec[:_SNAPSHOT_DIM]


class CognitiveBackbone:
    """Shared cognitive encoder producing z_t for downstream modules.

    Parameters
    ----------
    use_mla : bool
        When True, use the MLA-lite backbone (RMSNorm + SwiGLU + low-rank KV
        attention inspired by DeepSeek-V3).  Default False preserves the
        original ``nn.TransformerEncoder`` path for checkpoint compatibility.
    """

    _RAMP_STEPS = 150
    _MAX_WEIGHT = 0.85

    def __init__(self, use_mla: bool = False):
        self.available = _TORCH
        self._use_mla = use_mla
        self._train_steps: int = 0
        self._outcome_buffer: List[Dict[str, Any]] = []
        self._buffer_cap = 1024
        self._prev_snapshot: Optional[List[float]] = None
        self._model_lock = threading.Lock() if _TORCH else None

        if _TORCH:
            if use_mla:
                self._encoder = _MLALiteBackbone()
            else:
                self._encoder = _TransformerBackbone()
            self._outcome_head = _OutcomeHead()
            self._next_state_head = _NextStateHead()

            _fab = _get_fabric()
            if _fab:
                self._encoder = _fab.register("backbone_encoder", self._encoder)
                self._outcome_head = _fab.register("backbone_outcome", self._outcome_head)
                self._next_state_head = _fab.register("backbone_next_state", self._next_state_head)

            params = (
                list(self._encoder.parameters())
                + list(self._outcome_head.parameters())
                + list(self._next_state_head.parameters())
            )
            self._optim = torch.optim.Adam(params, lr=5e-4)
        else:
            self._encoder = None
            self._outcome_head = None
            self._next_state_head = None
            self._optim = None

    @property
    def learned_weight(self) -> float:
        progress = min(1.0, self._train_steps / max(self._RAMP_STEPS, 1))
        return progress * self._MAX_WEIGHT

    def encode_state(
        self, snapshot: List[float], content_embedding: Optional[List[float]] = None
    ) -> Optional[List[float]]:
        """Encode a 64-d snapshot into 512-d z_t.  Returns None if unavailable."""
        if not self.available or self._encoder is None:
            return None
        if self.learned_weight < 0.01:
            return None
        if len(snapshot) < _SNAPSHOT_DIM:
            snapshot = snapshot + [0.0] * (_SNAPSHOT_DIM - len(snapshot))
        elif len(snapshot) > _SNAPSHOT_DIM:
            snapshot = snapshot[:_SNAPSHOT_DIM]
        _fab = _get_fabric()
        with self._model_lock:
            self._encoder.eval()
            with torch.no_grad():
                x = _fab.tensor(snapshot) if _fab else torch.tensor(snapshot, dtype=torch.float32)
                ce = None
                if content_embedding is not None:
                    ce = (
                        _fab.tensor(content_embedding)
                        if _fab
                        else torch.tensor(content_embedding, dtype=torch.float32)
                    )
                    if ce.shape[-1] != _CONTENT_DIM:
                        if ce.shape[-1] > _CONTENT_DIM:
                            ce = ce[..., :_CONTENT_DIM]
                        else:
                            pad = torch.zeros(
                                *ce.shape[:-1],
                                _CONTENT_DIM - ce.shape[-1],
                                dtype=ce.dtype,
                                device=ce.device,
                            )
                            ce = torch.cat([ce, pad], dim=-1)
                z = self._encoder(x, content_embedding=ce).squeeze(0)
        return z.tolist()

    def record_outcome(
        self,
        snapshot: List[float],
        outcome_score: float,
        next_snapshot: Optional[List[float]] = None,
        content_embedding: Optional[List[float]] = None,
    ):
        """Store a training sample."""
        self._outcome_buffer.append(
            {
                "snapshot": snapshot,
                "outcome": outcome_score,
                "next_snapshot": next_snapshot,
                "content_embedding": content_embedding,
            }
        )
        if len(self._outcome_buffer) > self._buffer_cap:
            self._outcome_buffer = self._outcome_buffer[-self._buffer_cap :]

    def train_step(self) -> Optional[float]:
        if not self.available or len(self._outcome_buffer) < 8:
            return None

        _fab = _get_fabric()
        import random

        batch_size = min(32, len(self._outcome_buffer))
        batch = random.sample(self._outcome_buffer, batch_size)

        snapshots = (
            _fab.tensor([b["snapshot"] for b in batch])
            if _fab
            else torch.tensor([b["snapshot"] for b in batch], dtype=torch.float32)
        )
        outcomes = (
            _fab.tensor([b["outcome"] for b in batch])
            if _fab
            else torch.tensor([b["outcome"] for b in batch], dtype=torch.float32)
        )

        has_ce = all(b.get("content_embedding") is not None for b in batch)
        content_embs = None
        if has_ce:
            content_embs = (
                _fab.tensor([b["content_embedding"] for b in batch])
                if _fab
                else torch.tensor([b["content_embedding"] for b in batch], dtype=torch.float32)
            )
            if content_embs is not None and content_embs.shape[-1] != _CONTENT_DIM:
                if content_embs.shape[-1] > _CONTENT_DIM:
                    content_embs = content_embs[..., :_CONTENT_DIM]
                else:
                    pad = torch.zeros(
                        *content_embs.shape[:-1],
                        _CONTENT_DIM - content_embs.shape[-1],
                        dtype=content_embs.dtype,
                        device=content_embs.device,
                    )
                    content_embs = torch.cat([content_embs, pad], dim=-1)

        with self._model_lock:
            self._encoder.train()

            z = self._encoder(snapshots, content_embedding=content_embs)
            pred_outcome = self._outcome_head(z)
            loss_outcome = nn.functional.mse_loss(pred_outcome, outcomes)

            loss_next = _fab.tensor(0.0) if _fab else torch.tensor(0.0, dtype=torch.float32)
            next_pairs = [
                (b["snapshot"], b["next_snapshot"])
                for b in batch
                if b.get("next_snapshot") is not None
            ]
            if next_pairs:
                ns_in = (
                    _fab.tensor([p[0] for p in next_pairs])
                    if _fab
                    else torch.tensor([p[0] for p in next_pairs], dtype=torch.float32)
                )
                ns_tgt = (
                    _fab.tensor([p[1] for p in next_pairs])
                    if _fab
                    else torch.tensor([p[1] for p in next_pairs], dtype=torch.float32)
                )
                ns_z = self._encoder(ns_in)
                ns_pred = self._next_state_head(ns_z)
                loss_next = nn.functional.mse_loss(ns_pred, ns_tgt)

            loss_contrastive = _fab.tensor(0.0) if _fab else torch.tensor(0.0, dtype=torch.float32)
            if content_embs is not None:
                ce_proj = self._encoder._content_proj(content_embs)
                ce_up = self._encoder._output_proj(ce_proj)
                loss_contrastive = 1.0 - F.cosine_similarity(z, ce_up, dim=-1).mean()

            loss = loss_outcome + 0.5 * loss_next + 0.3 * loss_contrastive

            params = (
                list(self._encoder.parameters())
                + list(self._outcome_head.parameters())
                + list(self._next_state_head.parameters())
            )
            self._optim.zero_grad()
            if _fab:
                _fab.scale_loss(loss).backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                _fab.scaler_step(self._optim)
                _fab.scaler_update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                self._optim.step()

        self._train_steps += 1
        return loss.item()

    def stats(self) -> Dict[str, Any]:
        arch = "mla_lite" if self._use_mla else "transformer"
        return {
            "available": self.available,
            "train_steps": self._train_steps,
            "learned_weight": round(self.learned_weight, 3),
            "buffer_size": len(self._outcome_buffer),
            "z_dim": _Z_DIM,
            "snapshot_dim": _SNAPSHOT_DIM,
            "architecture": arch,
        }

    def to_dict(self) -> Dict[str, Any]:
        arch = "mla_lite" if self._use_mla else "transformer"
        data: Dict[str, Any] = {
            "train_steps": self._train_steps,
            "architecture": arch,
        }
        if self.available and self._encoder is not None:
            data["encoder_state"] = {k: v.tolist() for k, v in self._encoder.state_dict().items()}
            data["outcome_head_state"] = {
                k: v.tolist() for k, v in self._outcome_head.state_dict().items()
            }
            data["next_state_head_state"] = {
                k: v.tolist() for k, v in self._next_state_head.state_dict().items()
            }
        return data

    def from_dict(self, data: Dict[str, Any]):
        saved_steps = data.get("train_steps", 0)
        if not self.available:
            self._train_steps = saved_steps
            return
        saved_arch = data.get("architecture", "transformer")
        current_arch = "mla_lite" if self._use_mla else "transformer"
        if saved_arch != current_arch:
            print(
                f"[CognitiveBackbone] architecture mismatch: saved={saved_arch}, "
                f"current={current_arch} — skipping weight restore",
                flush=True,
            )
            return
        loaded_all = True
        for key, module in [
            ("encoder_state", self._encoder),
            ("outcome_head_state", self._outcome_head),
            ("next_state_head_state", self._next_state_head),
        ]:
            state_data = data.get(key)
            if state_data and module is not None:
                try:
                    restored = {k: torch.tensor(v) for k, v in state_data.items()}
                    module.load_state_dict(restored)
                except Exception as _e:
                    print(f"[CognitiveBackbone] from_dict load failed for {key}: {_e}", flush=True)
                    loaded_all = False
            else:
                loaded_all = False
        if loaded_all:
            self._train_steps = saved_steps
        else:
            self._train_steps = 0
            print(
                "[CognitiveBackbone] partial load failure — resetting train_steps to 0", flush=True
            )
