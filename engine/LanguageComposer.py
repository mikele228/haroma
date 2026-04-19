"""
LanguageComposer — Self-Grown Voice for HaromaX6 (Phase 10).

Replaces template-based language generation with a learned
phrase-selection system.  A PhraseLexicon (seeded from current
templates, growing from experience) combined with a small neural
CompositionScorer learns which phrases work best in which contexts.

At earned weight 0 the output is identical to the old template path.
As weight increases the scorer guides selection and the lexicon grows
with novel phrases extracted from successful cycles.

If PyTorch is unavailable the class degrades to a no-op stub.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import threading
import json
import math
import os
import random
import re

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    _TORCH_AVAILABLE = False
    np = None  # type: ignore

from engine.ComputeFabric import get_fabric as _get_fabric
from engine.LLMBackend import LLMBackend

STRATEGIES = ["inform", "inquire", "empathize", "advance_goal", "reflect"]
INTERLOCUTOR_STYLES = [
    "unknown",
    "inquisitive",
    "directive",
    "expressive",
    "informative",
]

_STRATEGY_TO_IDX = {s: i for i, s in enumerate(STRATEGIES)}
_STYLE_TO_IDX = {s: i for i, s in enumerate(INTERLOCUTOR_STYLES)}

_SCALAR_CONTEXT = 13  # 2 emotion + 5 strategy + 5 style + 1 external
_KG_CONTEXT_DIM = 64
_DEFAULT_EMBED = 256
HIDDEN_DIM = 64

# Empty seeds: no template openers/closers/bridges in code (learning fills voice).
_EMOTIONS = (
    "joy",
    "wonder",
    "curiosity",
    "fear",
    "sadness",
    "anger",
    "resolve",
    "peace",
    "surprise",
    "neutral",
)
_SEED_OPENERS = {e: [""] for e in _EMOTIONS}
_SEED_CLOSERS = {s: "" for s in STRATEGIES}
_SEED_BRIDGES: List[str] = []

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ======================================================================
# Phrase Lexicon
# ======================================================================


class _PhraseEntry:
    __slots__ = ("text", "category", "tags", "use_count", "total_outcome", "is_seed")

    def __init__(
        self, text: str, category: str, tags: Optional[List[str]] = None, is_seed: bool = False
    ):
        self.text = text
        self.category = category
        self.tags: List[str] = tags or []
        self.use_count: int = 0
        self.total_outcome: float = 0.0
        self.is_seed = is_seed

    @property
    def avg_outcome(self) -> float:
        if self.use_count == 0:
            return 0.5
        return self.total_outcome / self.use_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "category": self.category,
            "tags": self.tags,
            "use_count": self.use_count,
            "total_outcome": round(self.total_outcome, 4),
            "is_seed": self.is_seed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> _PhraseEntry:
        p = cls(d["text"], d["category"], tags=d.get("tags", []), is_seed=d.get("is_seed", False))
        p.use_count = d.get("use_count", 0)
        p.total_outcome = d.get("total_outcome", 0.0)
        return p


class PhraseLexicon:
    def __init__(self):
        self._phrases: List[_PhraseEntry] = []
        self._text_index: Dict[str, int] = {}
        self._seed()

    def _seed(self):
        for emotion, variants in _SEED_OPENERS.items():
            for text in variants:
                if not (text or "").strip():
                    continue
                self._add_internal(text, "opener", [emotion], is_seed=True)
        for strategy, text in _SEED_CLOSERS.items():
            if not (text or "").strip():
                continue
            self._add_internal(text, "closer", [strategy], is_seed=True)
        for text in _SEED_BRIDGES:
            if not (text or "").strip():
                continue
            self._add_internal(text, "bridge", [], is_seed=True)

    def _add_internal(
        self, text: str, category: str, tags: List[str], is_seed: bool = False
    ) -> int:
        key = text.strip().lower()
        if key in self._text_index:
            return self._text_index[key]
        idx = len(self._phrases)
        self._phrases.append(_PhraseEntry(text, category, tags=tags, is_seed=is_seed))
        self._text_index[key] = idx
        return idx

    def add_phrase(self, text: str, category: str, tags: Optional[List[str]] = None) -> int:
        return self._add_internal(text, category, tags or [])

    def get_candidates(self, category: str, tags: Optional[List[str]] = None) -> List[tuple]:
        """Return [(idx, PhraseEntry)] for the category, optionally filtered."""
        results = []
        tag_set = set(tags) if tags else None
        for i, p in enumerate(self._phrases):
            if p.category != category:
                continue
            if tag_set and not tag_set.intersection(p.tags):
                continue
            results.append((i, p))
        if not results and tag_set:
            return self.get_candidates(category, tags=None)
        return results

    def reinforce(self, idx: int, outcome_score: float):
        if 0 <= idx < len(self._phrases):
            p = self._phrases[idx]
            p.use_count += 1
            p.total_outcome += outcome_score

    def prune(self, min_uses: int = 5, min_avg_outcome: float = 0.3):
        keep = []
        new_index: Dict[str, int] = {}
        for p in self._phrases:
            if p.is_seed or p.use_count < min_uses or p.avg_outcome >= min_avg_outcome:
                new_idx = len(keep)
                keep.append(p)
                new_index[p.text.strip().lower()] = new_idx
        self._phrases = keep
        self._text_index = new_index

    def get_phrase(self, idx: int) -> Optional[_PhraseEntry]:
        if 0 <= idx < len(self._phrases):
            return self._phrases[idx]
        return None

    @property
    def openers(self) -> List[_PhraseEntry]:
        return [p for p in self._phrases if p.category == "opener"]

    @property
    def closers(self) -> List[_PhraseEntry]:
        return [p for p in self._phrases if p.category == "closer"]

    @property
    def bridges(self) -> List[_PhraseEntry]:
        return [p for p in self._phrases if p.category == "bridge"]

    def total_count(self) -> int:
        return len(self._phrases)

    def avg_outcome(self) -> float:
        used = [p for p in self._phrases if p.use_count > 0]
        if not used:
            return 0.0
        return sum(p.avg_outcome for p in used) / len(used)

    def to_dict(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self._phrases]

    def from_dict(self, data: List[Dict[str, Any]]):
        self._phrases.clear()
        self._text_index.clear()
        for d in data:
            p = _PhraseEntry.from_dict(d)
            idx = len(self._phrases)
            self._phrases.append(p)
            self._text_index[p.text.strip().lower()] = idx


# ======================================================================
# Composition Scorer (PyTorch MLP)
# ======================================================================


class _CompositionScorerNet(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, input_dim: int = 525):
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(x))
        return torch.sigmoid(self.fc2(h)).squeeze(-1)


# ======================================================================
# Generative Decoder (Phase 13 GRU)
# ======================================================================

_PAD_TOKEN = "<pad>"
_SOS_TOKEN = "<sos>"
_EOS_TOKEN = "<eos>"
_UNK_TOKEN = "<unk>"
_SPECIAL_TOKENS = [_PAD_TOKEN, _SOS_TOKEN, _EOS_TOKEN, _UNK_TOKEN]
_MAX_VOCAB = 4000
_WORD_SPLIT = re.compile(r"[^a-z']+")


class WordVocabulary:
    """Word-level vocab that grows from experience, capped at _MAX_VOCAB."""

    def __init__(self):
        self._word2idx: Dict[str, int] = {}
        self._idx2word: List[str] = []
        for tok in _SPECIAL_TOKENS:
            self._add(tok)
        self._seed_common()

    def _add(self, word: str) -> int:
        if word in self._word2idx:
            return self._word2idx[word]
        if len(self._idx2word) >= _MAX_VOCAB:
            return self._word2idx.get(_UNK_TOKEN, 3)
        idx = len(self._idx2word)
        self._idx2word.append(word)
        self._word2idx[word] = idx
        return idx

    def _seed_common(self):
        seed_words = set()
        for variants in _SEED_OPENERS.values():
            for phrase in variants:
                seed_words.update(_WORD_SPLIT.split(phrase.lower()))
        for phrase in _SEED_CLOSERS.values():
            seed_words.update(_WORD_SPLIT.split(phrase.lower()))
        for phrase in _SEED_BRIDGES:
            seed_words.update(_WORD_SPLIT.split(phrase.lower()))
        common = [
            "i",
            "you",
            "we",
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "this",
            "that",
            "it",
            "and",
            "or",
            "but",
            "not",
            "in",
            "on",
            "with",
            "for",
            "to",
            "of",
            "from",
            "by",
            "at",
            "as",
            "so",
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "which",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "can",
            "could",
            "should",
            "may",
            "might",
            "must",
            "be",
            "been",
            "being",
            "am",
            "know",
            "think",
            "feel",
            "see",
            "hear",
            "understand",
            "wonder",
            "believe",
            "want",
            "need",
            "like",
            "love",
            "find",
            "make",
            "take",
            "give",
            "tell",
            "say",
            "ask",
            "new",
            "good",
            "great",
            "well",
            "much",
            "very",
            "more",
            "here",
            "there",
            "now",
            "then",
            "just",
            "also",
            "still",
            "about",
            "into",
            "over",
            "through",
            "between",
            "under",
            "something",
            "nothing",
            "everything",
            "anything",
            "my",
            "your",
            "our",
            "its",
            "their",
            "his",
            "her",
            "me",
            "him",
            "them",
            "us",
            "one",
            "two",
            "some",
            "all",
            "many",
            "each",
            "every",
            "if",
            "because",
            "since",
            "while",
            "though",
            "although",
            "perhaps",
            "maybe",
            "yet",
            "already",
            "always",
            "never",
        ]
        seed_words.update(common)
        for w in sorted(seed_words):
            if w and len(w) > 0:
                self._add(w)

    def encode_text(self, text: str) -> List[int]:
        words = _WORD_SPLIT.split(text.lower())
        unk_id = self._word2idx[_UNK_TOKEN]
        return [self._word2idx.get(w, unk_id) for w in words if w]

    def grow_from_text(self, text: str):
        for w in _WORD_SPLIT.split(text.lower()):
            if w and len(w) > 1:
                self._add(w)

    def decode_ids(self, ids: List[int]) -> str:
        eos_id = self._word2idx[_EOS_TOKEN]
        words = []
        for i in ids:
            if i == eos_id:
                break
            if 0 <= i < len(self._idx2word):
                tok = self._idx2word[i]
                if tok not in _SPECIAL_TOKENS:
                    words.append(tok)
        return " ".join(words)

    @property
    def size(self) -> int:
        return len(self._idx2word)

    @property
    def sos_id(self) -> int:
        return self._word2idx[_SOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self._word2idx[_EOS_TOKEN]

    def to_dict(self) -> Dict[str, Any]:
        return {"words": self._idx2word}

    def from_dict(self, data: Dict[str, Any]):
        words = data.get("words", [])
        self._word2idx.clear()
        self._idx2word.clear()
        for w in words:
            idx = len(self._idx2word)
            self._idx2word.append(w)
            self._word2idx[w] = idx


_DECODER_D_MODEL = 128
_DECODER_N_HEADS = 4
_DECODER_N_LAYERS = 4
_DECODER_FF_DIM = 256
_DECODER_MAX_LEN = 256
_N_MEMORY_TOKENS = 4


def _sinusoidal_pos_encoding(max_len: int, d_model: int) -> "torch.Tensor":
    """Pre-compute sinusoidal positional encoding."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class _TransformerDecoderNet(nn.Module if _TORCH_AVAILABLE else object):
    """Context-conditioned transformer decoder for text generation."""

    def __init__(self, context_dim: int = 269, vocab_size: int = 500, word_embed_dim: int = 64):
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        self._d_model = _DECODER_D_MODEL
        self._vocab_size = vocab_size

        self.word_embedding = nn.Embedding(vocab_size, word_embed_dim)
        self.word_proj = nn.Linear(word_embed_dim, _DECODER_D_MODEL)
        self.register_buffer(
            "_pos_enc", _sinusoidal_pos_encoding(_DECODER_MAX_LEN, _DECODER_D_MODEL)
        )

        self.context_proj = nn.Linear(context_dim, _N_MEMORY_TOKENS * _DECODER_D_MODEL)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=_DECODER_D_MODEL,
            nhead=_DECODER_N_HEADS,
            dim_feedforward=_DECODER_FF_DIM,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=_DECODER_N_LAYERS)

        self.output_proj = nn.Linear(_DECODER_D_MODEL, vocab_size)

        nn.init.xavier_uniform_(self.word_embedding.weight)
        nn.init.xavier_uniform_(self.word_proj.weight)
        nn.init.xavier_uniform_(self.context_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def _make_causal_mask(self, seq_len: int) -> "torch.Tensor":
        return nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=self.word_embedding.weight.device
        )

    def _encode_context(self, context: "torch.Tensor") -> "torch.Tensor":
        """Project context to N_MEMORY_TOKENS key-value tokens."""
        if context.dim() == 1:
            context = context.unsqueeze(0)
        batch = context.shape[0]
        memory = self.context_proj(context)
        return memory.view(batch, _N_MEMORY_TOKENS, _DECODER_D_MODEL)

    def forward(self, token_ids: "torch.Tensor", context: "torch.Tensor") -> "torch.Tensor":
        """token_ids: (batch, seq_len), context: (batch, context_dim) -> logits."""
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        seq_len = token_ids.shape[1]
        emb = self.word_proj(self.word_embedding(token_ids))
        emb = emb + self._pos_enc[:seq_len].unsqueeze(0)

        memory = self._encode_context(context)
        causal_mask = self._make_causal_mask(seq_len)
        decoded = self.decoder(emb, memory, tgt_mask=causal_mask)
        return self.output_proj(decoded)

    def generate(
        self,
        context: "torch.Tensor",
        sos_id: int,
        eos_id: int,
        max_len: int = 40,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> "List[int]":
        """Autoregressive generation with top-k sampling."""
        if context.dim() == 1:
            context = context.unsqueeze(0)
        memory = self._encode_context(context)
        device = self.word_embedding.weight.device

        generated: List[int] = []
        ids = torch.tensor([[sos_id]], dtype=torch.long, device=device)

        for _ in range(max_len):
            seq_len = ids.shape[1]
            emb = self.word_proj(self.word_embedding(ids))
            emb = emb + self._pos_enc[:seq_len].unsqueeze(0)
            causal_mask = self._make_causal_mask(seq_len)
            decoded = self.decoder(emb, memory, tgt_mask=causal_mask)

            logits = self.output_proj(decoded[:, -1, :]).squeeze(0)
            logits = logits / max(temperature, 0.1)

            if top_k > 0 and top_k < logits.shape[-1]:
                topk_vals, _ = torch.topk(logits, top_k)
                logits[logits < topk_vals[-1]] = float("-inf")

            seen = set(generated[-3:]) if generated else set()
            for prev_id in seen:
                if 0 <= prev_id < logits.shape[-1]:
                    logits[prev_id] -= 1.5

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            if next_id == eos_id:
                break
            generated.append(next_id)
            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
            )

        return generated

    def resize_vocab(self, new_size: int):
        old_size = self.word_embedding.num_embeddings
        if new_size <= old_size:
            return
        old_embed_weight = self.word_embedding.weight.data
        new_embed = nn.Embedding(new_size, self.word_embedding.embedding_dim)
        nn.init.xavier_uniform_(new_embed.weight)
        new_embed.weight.data[:old_size] = old_embed_weight
        self.word_embedding = new_embed

        old_proj_weight = self.output_proj.weight.data
        old_proj_bias = self.output_proj.bias.data
        new_proj = nn.Linear(_DECODER_D_MODEL, new_size)
        nn.init.xavier_uniform_(new_proj.weight)
        new_proj.weight.data[:old_size] = old_proj_weight
        new_proj.bias.data[:old_size] = old_proj_bias
        self.output_proj = new_proj
        self._vocab_size = new_size


# ======================================================================
# Knowledge Context Encoder (Phase 14)
# ======================================================================


class _KnowledgeContextEncoder(nn.Module if _TORCH_AVAILABLE else object):
    """Encodes up to 5 KG triples into a fixed 64-d vector."""

    def __init__(self, embed_dim: int = _DEFAULT_EMBED, output_dim: int = _KG_CONTEXT_DIM):
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        self.proj = nn.Linear(embed_dim, output_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, triple_vectors: torch.Tensor) -> torch.Tensor:
        """triple_vectors: (N, embed_dim) — mean-pooled triple embeddings.
        Returns: (output_dim,) vector."""
        if triple_vectors.shape[0] == 0:
            return torch.zeros(
                self.proj.out_features, dtype=self.proj.weight.dtype, device=self.proj.weight.device
            )
        pooled = triple_vectors.mean(dim=0)
        return torch.tanh(self.proj(pooled))


# ======================================================================
# Language Composer
# ======================================================================


class LanguageComposer:
    def __init__(self, encoder=None, llm_backend=None, data_lock: Optional[threading.Lock] = None):
        self._encoder = encoder
        embed_dim = getattr(encoder, "_embed_dim", _DEFAULT_EMBED) if encoder else _DEFAULT_EMBED
        self._phrase_dim = embed_dim
        self._kg_dim = _KG_CONTEXT_DIM
        self._context_dim = embed_dim + _SCALAR_CONTEXT + self._kg_dim
        self._z_dim = 512
        self._full_context_dim = self._context_dim + self._z_dim
        self._input_dim = self._full_context_dim + self._phrase_dim
        self._lexicon = PhraseLexicon()
        self._train_steps: int = 0
        self._available = _TORCH_AVAILABLE

        if self._available:
            self._scorer = _CompositionScorerNet(input_dim=self._input_dim)
            self._scorer.eval()
            self._kg_encoder = _KnowledgeContextEncoder(
                embed_dim=embed_dim, output_dim=self._kg_dim
            )
            _fab = _get_fabric()
            if _fab:
                self._scorer = _fab.register("composition_scorer", self._scorer)
                self._kg_encoder = _fab.register("kg_context_encoder", self._kg_encoder)
            self._optimizer = torch.optim.Adam(self._scorer.parameters(), lr=0.001)
        else:
            self._scorer = None
            self._optimizer = None
            self._kg_encoder = None

        self._pending: List[Dict[str, Any]] = []
        self._data_lock = data_lock if data_lock is not None else threading.Lock()

        # Phase 16 transformer decoder
        self._vocab = WordVocabulary()
        self._gen_train_steps: int = 0
        self._gen_pending: List[Dict[str, Any]] = []
        self._last_llm_prompt: str = ""
        self._last_llm_response: str = ""
        if self._available:
            self._decoder = _TransformerDecoderNet(
                context_dim=self._full_context_dim, vocab_size=self._vocab.size, word_embed_dim=64
            )
            self._decoder.eval()
            _fab = _get_fabric()
            if _fab:
                self._decoder = _fab.register("generative_decoder", self._decoder)
            self._gen_optimizer = torch.optim.Adam(self._decoder.parameters(), lr=5e-4)
        else:
            self._decoder = None
            self._gen_optimizer = None

        self._llm = llm_backend if llm_backend is not None else LLMBackend()

        self._offline_assets_bootstrapped = False

    # ------------------------------------------------------------------
    # Offline assets (downloaded corpora -> vocab + generative seed)
    # ------------------------------------------------------------------

    _BOOTSTRAP_EMOTION_VA = {
        "joy": (0.55, 0.55),
        "anger": (-0.55, 0.72),
        "fear": (-0.45, 0.78),
        "sadness": (-0.52, 0.35),
        "surprise": (0.15, 0.65),
        "disgust": (-0.48, 0.45),
        "neutral": (0.0, 0.25),
    }

    def bootstrap_offline_training_assets(
        self,
        project_root: Optional[str] = None,
        max_gen_seed: int = 96,
        seed: int = 17,
        min_text_len: int = 12,
    ) -> Dict[str, Any]:
        """Merge `data/cognitive/word_vocabulary.json` and seed `_gen_pending`
        from `data/training/dialog_training.json` if present.

        Call once after persistence restore so saved weights stay canonical
        while offline text augments the word table and generative queue."""
        out: Dict[str, Any] = {
            "vocab_merged": 0,
            "gen_seeded": 0,
            "skipped": False,
        }
        if self._offline_assets_bootstrapped:
            out["skipped"] = True
            return out
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vocab_path = os.path.join(project_root, "data", "cognitive", "word_vocabulary.json")
        dialog_path = os.path.join(project_root, "data", "training", "dialog_training.json")

        pre_vocab = self._vocab.size
        if os.path.isfile(vocab_path):
            try:
                with open(vocab_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                words = payload.get("words", [])
                for w in words:
                    if not isinstance(w, str) or len(w) < 2:
                        continue
                    if w in _SPECIAL_TOKENS or w.startswith("<"):
                        continue
                    n0 = self._vocab.size
                    self._vocab.grow_from_text(w)
                    if self._vocab.size > n0:
                        out["vocab_merged"] += 1
            except (OSError, json.JSONDecodeError, TypeError):
                pass

        if (
            self._available
            and self._decoder is not None
            and self._vocab.size > self._decoder.word_embedding.num_embeddings
        ):
            self._decoder.resize_vocab(self._vocab.size)
            self._gen_optimizer = torch.optim.Adam(self._decoder.parameters(), lr=5e-4)

        rows: List[Dict[str, Any]] = []
        if os.path.isfile(dialog_path):
            try:
                with open(dialog_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    rows = [r for r in raw if isinstance(r, dict)]
            except (OSError, json.JSONDecodeError, TypeError):
                rows = []

        if rows:
            rng = random.Random(seed)
            rng.shuffle(rows)
            for row in rows:
                if out["gen_seeded"] >= max_gen_seed:
                    break
                text = (row.get("text") or "").strip()
                if len(text) < min_text_len:
                    continue
                emotion = (row.get("emotion") or "neutral").lower()
                va = self._BOOTSTRAP_EMOTION_VA.get(emotion, self._BOOTSTRAP_EMOTION_VA["neutral"])
                ctx_txt = (row.get("context") or "").strip()
                phrase_emb = None
                if self._encoder and getattr(self._encoder, "available", False):
                    try:
                        cue = ctx_txt if len(ctx_txt) > 8 else text
                        phrase_emb = self._encoder.encode(cue[:512])
                    except Exception:
                        phrase_emb = None
                ctx = self._build_context(
                    phrase_emb, va[0], va[1], "inform", "unknown", 0.0, None, None
                )
                if ctx is None:
                    continue
                self.record_text_outcome(ctx, text, 0.55)
                out["gen_seeded"] += 1

        if (
            self._available
            and self._decoder is not None
            and self._vocab.size > self._decoder.word_embedding.num_embeddings
        ):
            self._decoder.resize_vocab(self._vocab.size)
            self._gen_optimizer = torch.optim.Adam(self._decoder.parameters(), lr=5e-4)

        self._offline_assets_bootstrapped = True
        out["vocab_size_before"] = pre_vocab
        out["vocab_size_after"] = self._vocab.size
        return out

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._available or self._llm.available

    @property
    def llm_available(self) -> bool:
        return self._llm.available

    @property
    def learned_weight(self) -> float:
        if self._llm.available:
            return 1.0
        if not self._available:
            return 0.0
        return min(0.8, self._train_steps / 150.0)

    # ------------------------------------------------------------------
    # Context vector construction
    # ------------------------------------------------------------------

    def _build_context(
        self,
        content_embedding,
        emotion_valence: float,
        emotion_arousal: float,
        strategy: str,
        interlocutor_style: str,
        has_external: float,
        knowledge_triples: Optional[List[tuple]] = None,
        z_t: Optional[List[float]] = None,
    ) -> Optional[List[float]]:
        if content_embedding is not None:
            embed = list(content_embedding)
        else:
            embed = [0.0] * self._phrase_dim

        strat_oh = [0.0] * len(STRATEGIES)
        idx = _STRATEGY_TO_IDX.get(strategy, 4)
        strat_oh[idx] = 1.0

        style_oh = [0.0] * len(INTERLOCUTOR_STYLES)
        sidx = _STYLE_TO_IDX.get(interlocutor_style, 0)
        style_oh[sidx] = 1.0

        kg_vec = self._encode_triples(knowledge_triples)

        ctx = (
            embed
            + [emotion_valence, emotion_arousal]
            + strat_oh
            + style_oh
            + [has_external]
            + kg_vec
        )
        if z_t is not None:
            ctx = ctx + list(z_t)
        else:
            ctx = ctx + [0.0] * self._z_dim
        return ctx

    def _encode_triples(self, triples: Optional[List[tuple]] = None) -> List[float]:
        if not triples or not self._available or self._kg_encoder is None:
            return [0.0] * self._kg_dim

        if not self._encoder or not self._encoder.available:
            return [0.0] * self._kg_dim

        vecs = []
        for triple in triples[:5]:
            parts = [str(p) for p in triple]
            combined = " ".join(parts)
            vec = self._encoder.encode(combined)
            if vec is not None:
                vecs.append(vec)

        if not vecs:
            return [0.0] * self._kg_dim

        with torch.no_grad():
            _fab = _get_fabric()
            stacked = (
                _fab.tensor(vecs, dtype=torch.float32)
                if _fab
                else torch.tensor(vecs, dtype=torch.float32)
            )
            kg_vec = self._kg_encoder(stacked)
            return kg_vec.tolist()

    @staticmethod
    def select_relevant_triples(
        knowledge_summary: Optional[Dict[str, Any]],
        knowledge_graph=None,
        nlu_entities: Optional[List[str]] = None,
        max_triples: int = 5,
    ) -> List[tuple]:
        if not knowledge_graph:
            return []

        triples: List[tuple] = []
        entities_to_query = []

        if nlu_entities:
            entities_to_query.extend(nlu_entities[:5])

        if knowledge_summary:
            top_ents = knowledge_summary.get("top_entities", [])
            for e in top_ents[:3]:
                if e not in entities_to_query:
                    entities_to_query.append(e)

        for ent_name in entities_to_query:
            if hasattr(knowledge_graph, "query_relations"):
                rels = knowledge_graph.query_relations(ent_name)
                if isinstance(rels, list):
                    for rel in rels[:3]:
                        if isinstance(rel, dict):
                            triple = (
                                rel.get("subject", ent_name),
                                rel.get("predicate", "relates_to"),
                                rel.get("object", "?"),
                            )
                        elif isinstance(rel, (list, tuple)) and len(rel) >= 3:
                            triple = (str(rel[0]), str(rel[1]), str(rel[2]))
                        else:
                            continue
                        triples.append(triple)
            if len(triples) >= max_triples:
                break

        return triples[:max_triples]

    def _encode_phrase(self, text: str) -> List[float]:
        if self._encoder and self._encoder.available:
            vec = self._encoder.encode(text)
            if vec is not None:
                return list(vec)
        return [0.0] * self._phrase_dim

    def _score_phrase(self, context: List[float], phrase_embed: List[float]) -> float:
        if not self._available or self._scorer is None:
            return 0.5
        full = context + phrase_embed
        with torch.no_grad():
            _fab = _get_fabric()
            t = (
                _fab.tensor(full, dtype=torch.float32)
                if _fab
                else torch.tensor(full, dtype=torch.float32)
            )
            return float(self._scorer(t))

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _select_from(self, candidates: List[tuple], context: List[float]) -> Optional[tuple]:
        """Return (idx, PhraseEntry) using temperature-weighted scoring."""
        if not candidates:
            return None

        if not self._available or self.learned_weight < 0.05:
            return random.choice(candidates)

        scores = []
        for idx, phrase in candidates:
            pe = self._encode_phrase(phrase.text)
            s = self._score_phrase(context, pe)
            scores.append(s)

        temperature = max(0.3, 1.0 - self.learned_weight)
        max_s = max(scores)
        exp_scores = [math.exp((s - max_s) / temperature) for s in scores]
        total = sum(exp_scores)
        if total == 0:
            return random.choice(candidates)
        probs = [e / total for e in exp_scores]

        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return candidates[i]
        return candidates[-1]

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(
        self,
        emotion: str,
        strategy: str,
        content_embedding,
        interlocutor_style: str,
        has_external: bool,
        content_elements: List[str],
        emotion_valence: float = 0.0,
        emotion_arousal: float = 0.0,
        knowledge_triples: Optional[List[tuple]] = None,
        z_t: Optional[List[float]] = None,
        cognitive_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._available:
            return None

        ctx = self._build_context(
            content_embedding,
            emotion_valence,
            emotion_arousal,
            strategy,
            interlocutor_style,
            float(has_external),
            knowledge_triples=knowledge_triples,
            z_t=z_t,
        )
        if ctx is None:
            return None

        gw = self.generative_weight
        if gw > 0.1 and random.random() < gw:
            temp = max(0.3, 1.0 - gw * 0.5)
            gen_text = self.generate_text(ctx, max_len=40, temperature=temp)
            if gen_text:
                return {
                    "text": gen_text,
                    "opener_idx": -1,
                    "closer_idx": -1,
                    "bridge_idxs": [],
                    "generative": True,
                }

        opener_candidates = self._lexicon.get_candidates("opener", tags=[emotion])
        selected_opener = self._select_from(opener_candidates, ctx)
        opener_idx = selected_opener[0] if selected_opener else -1
        opener_text = selected_opener[1].text if selected_opener else ""

        closer_candidates = self._lexicon.get_candidates("closer", tags=[strategy])
        selected_closer = self._select_from(closer_candidates, ctx)
        closer_idx = selected_closer[0] if selected_closer else -1
        closer_text = selected_closer[1].text if selected_closer else ""

        bridge_idxs: List[int] = []
        bridge_texts: List[str] = []
        if len(content_elements) > 1:
            bridge_candidates = self._lexicon.get_candidates("bridge")
            for _ in range(len(content_elements) - 1):
                sel = self._select_from(bridge_candidates, ctx)
                if sel:
                    bridge_idxs.append(sel[0])
                    bridge_texts.append(sel[1].text)

        parts: List[str] = []
        if opener_text:
            parts.append(opener_text)

        for i, element in enumerate(content_elements):
            if element:
                parts.append(f"{element}. ")
            if i < len(bridge_texts):
                parts.append(bridge_texts[i])

        if closer_text:
            parts.append(closer_text)

        return {
            "text": "".join(parts).strip(),
            "opener_idx": opener_idx,
            "closer_idx": closer_idx,
            "bridge_idxs": bridge_idxs,
        }

    # ------------------------------------------------------------------
    # Fast composition (System-1 path, no LLM / no torch required)
    # ------------------------------------------------------------------

    def compose_fast(
        self, emotion: str, strategy: str, content_elements: List[str]
    ) -> Dict[str, Any]:
        """Lightweight template composition for the fast-path chat handler.

        Uses only the phrase lexicon (openers/closers/bridges).  No LLM,
        no torch context vector, no scorer.  Runs in microseconds.
        """
        import random as _rng

        opener_variants = _SEED_OPENERS.get(emotion, _SEED_OPENERS.get("neutral", [""]))
        opener = _rng.choice(opener_variants) if opener_variants else ""

        closer = _SEED_CLOSERS.get(strategy, "")

        parts: List[str] = []
        if opener:
            parts.append(opener)

        for i, element in enumerate(content_elements):
            if element:
                parts.append(element)
                if not element.endswith((".", "!", "?")):
                    parts.append(". ")
                else:
                    parts.append(" ")
            if i < len(content_elements) - 1:
                bridge = _rng.choice(_SEED_BRIDGES) if _SEED_BRIDGES else ""
                if bridge:
                    parts.append(bridge)

        if closer:
            parts.append(closer)

        return {
            "text": "".join(parts).strip(),
            "opener_idx": -1,
            "closer_idx": -1,
            "bridge_idxs": [],
            "fast_path": True,
        }

    # ------------------------------------------------------------------
    # LLM critique → lexicon / reward (on-demand; not BackgroundAgent-scheduled)
    # ------------------------------------------------------------------

    _CRITIQUE_PROMPT_TEMPLATE = (
        "You are a language coach reviewing a conversational AI's response.\n"
        "The AI's personality is warm, reflective, and emotionally aware.\n\n"
        "USER INPUT: {user_input}\n"
        "AI RESPONSE: {response}\n"
        "EMOTION: {emotion}\n"
        "STRATEGY: {strategy}\n\n"
        "Tasks:\n"
        "1. Rate the response quality from 0.0 to 1.0.\n"
        "2. Write 2-3 improved alternative phrasings (one per line, "
        "prefixed with PHRASE:).\n"
        "3. Write one short insight about what could be better "
        "(prefixed with INSIGHT:).\n\n"
        "Respond ONLY with the rating, phrases, and insight."
    )

    def learn_from_llm(
        self,
        turns: List[Dict[str, Any]],
        max_turns: int = 6,
    ) -> Dict[str, Any]:
        """Background learning: ask the LLM to critique recent organic
        responses and extract improved phrases into the lexicon.

        *turns* should be a list of dicts with keys:
            speaker, content, response, emotion, tags

        Returns a metrics dict (phrases_added, critiques, insights).
        """
        if not self._llm.available:
            return {"skipped": True, "reason": "llm_unavailable"}

        usable = [
            t for t in turns if t.get("response") and t.get("content") and len(t["response"]) > 10
        ][-max_turns:]

        if not usable:
            return {"skipped": True, "reason": "no_usable_turns"}

        phrases_added = 0
        critiques: List[Dict[str, Any]] = []
        insights: List[str] = []

        for turn in usable:
            prompt = self._CRITIQUE_PROMPT_TEMPLATE.format(
                user_input=turn["content"][:400],
                response=turn["response"][:400],
                emotion=turn.get("emotion", "neutral"),
                strategy=",".join(turn.get("tags", [])[:3]) or "inform",
            )

            raw = self._llm.generate(prompt, max_tokens=300, temperature=0.6)
            if not raw:
                continue

            score = self._parse_critique_score(raw)
            new_phrases = self._parse_critique_phrases(raw)
            insight = self._parse_critique_insight(raw)

            emotion = turn.get("emotion", "neutral")
            strategy_tag = (turn.get("tags") or ["inform"])[0]

            for phrase_text in new_phrases:
                if len(phrase_text.split()) < 3:
                    continue
                self.extract_phrases(phrase_text, emotion, strategy_tag, max(score, 0.7))
                phrases_added += 1

            if score > 0 and turn["response"]:
                self._llm.record_outcome(prompt, turn["response"], score)

            critique_entry = {
                "input": turn["content"][:120],
                "score": score,
                "phrases": len(new_phrases),
            }
            critiques.append(critique_entry)

            if insight:
                insights.append(insight)

        if critiques:
            self._llm.train_reward_model()

        return {
            "phrases_added": phrases_added,
            "critiques": critiques,
            "insights": insights,
            "turns_reviewed": len(usable),
        }

    @staticmethod
    def _parse_critique_score(text: str) -> float:
        """Extract a 0.0-1.0 rating from LLM critique output."""
        import re as _re

        for line in text.split("\n"):
            line = line.strip()
            m = _re.search(r"\b([01]\.?\d*)\b", line)
            if m:
                try:
                    val = float(m.group(1))
                    if 0.0 <= val <= 1.0:
                        return val
                except ValueError:
                    pass
            if line and not line.startswith(("PHRASE:", "INSIGHT:")):
                break
        return 0.5

    @staticmethod
    def _parse_critique_phrases(text: str) -> List[str]:
        """Extract PHRASE: lines from LLM critique output."""
        phrases = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("PHRASE:"):
                phrase = stripped[7:].strip().strip('"').strip("'")
                if phrase and len(phrase) > 5:
                    phrases.append(phrase)
        return phrases[:6]

    @staticmethod
    def _parse_critique_insight(text: str) -> str:
        """Extract INSIGHT: line from LLM critique output."""
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("INSIGHT:"):
                return stripped[8:].strip()
        return ""

    # ------------------------------------------------------------------
    # Outcome recording
    # ------------------------------------------------------------------

    def record_outcome(
        self, composition_meta: Dict[str, Any], outcome_score: float, context_features: List[float]
    ):
        opener_idx = composition_meta.get("opener_idx", -1)
        closer_idx = composition_meta.get("closer_idx", -1)
        bridge_idxs = composition_meta.get("bridge_idxs", [])

        if opener_idx >= 0:
            self._lexicon.reinforce(opener_idx, outcome_score)
        if closer_idx >= 0:
            self._lexicon.reinforce(closer_idx, outcome_score)
        for bidx in bridge_idxs:
            self._lexicon.reinforce(bidx, outcome_score)

        phrase_idxs = (
            ([opener_idx] if opener_idx >= 0 else [])
            + ([closer_idx] if closer_idx >= 0 else [])
            + bridge_idxs
        )

        for pidx in phrase_idxs:
            phrase = self._lexicon.get_phrase(pidx)
            if phrase:
                pe = self._encode_phrase(phrase.text)
                with self._data_lock:
                    self._pending.append(
                        {
                            "context": context_features,
                            "phrase_embed": pe,
                            "outcome": outcome_score,
                        }
                    )

    def extract_phrases(self, text: str, emotion: str, strategy: str, outcome_score: float):
        """Extract novel phrases from a successful utterance."""
        if outcome_score < 0.6:
            return

        sentences = _SENTENCE_SPLIT.split(text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        if not sentences:
            return

        for i, sentence in enumerate(sentences):
            words = sentence.split()
            if len(words) < 3:
                continue

            key = sentence.strip().lower()
            if key in self._lexicon._text_index:
                continue

            if i == 0:
                category = "opener"
                tags = [emotion]
            elif i == len(sentences) - 1:
                category = "closer"
                tags = [strategy]
            else:
                category = "bridge"
                tags = [emotion, strategy]

            self._lexicon.add_phrase(sentence + " ", category, tags)

    def train_step(self) -> float:
        if not self._available or self._scorer is None:
            return 0.0
        with self._data_lock:
            if not self._pending:
                return 0.0
            batch = list(self._pending[-20:])
            self._pending.clear()

        self._scorer.train()
        total_loss = 0.0
        n = 0

        _fab = _get_fabric()
        for item in batch:
            full = item["context"] + item["phrase_embed"]
            x = (
                _fab.tensor(full, dtype=torch.float32)
                if _fab
                else torch.tensor(full, dtype=torch.float32)
            )
            target = (
                _fab.tensor([item["outcome"]], dtype=torch.float32)
                if _fab
                else torch.tensor([item["outcome"]], dtype=torch.float32)
            )

            pred = self._scorer(x)
            loss = F.mse_loss(pred, target)

            self._optimizer.zero_grad()
            if _fab:
                _fab.scale_loss(loss).backward()
                nn.utils.clip_grad_norm_(self._scorer.parameters(), 1.0)
                _fab.scaler_step(self._optimizer)
                _fab.scaler_update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self._scorer.parameters(), 1.0)
                self._optimizer.step()

            total_loss += float(loss.item())
            n += 1

        self._train_steps += 1
        self._scorer.eval()

        if self._train_steps % 50 == 0:
            self._lexicon.prune()

        return total_loss / max(n, 1)

    # ------------------------------------------------------------------
    # Generative language (Phase 13)
    # ------------------------------------------------------------------

    @property
    def generative_weight(self) -> float:
        if not self._available or self._decoder is None:
            return 0.0
        return min(0.8, self._gen_train_steps / 200.0)

    def generate_text(
        self, context: List[float], max_len: int = 40, temperature: float = 0.7
    ) -> Optional[str]:
        if not self._available or self._decoder is None:
            return None

        with torch.no_grad():
            _fab = _get_fabric()
            ctx_t = (
                _fab.tensor(context, dtype=torch.float32)
                if _fab
                else torch.tensor(context, dtype=torch.float32)
            )
            generated_ids = self._decoder.generate(
                ctx_t,
                sos_id=self._vocab.sos_id,
                eos_id=self._vocab.eos_id,
                max_len=max_len,
                temperature=temperature,
                top_k=20,
            )

        text = self._vocab.decode_ids(generated_ids)
        if not text.strip():
            return None
        if not text.rstrip().endswith((".", "!", "?")):
            text = text.rstrip() + "."
        return text[0].upper() + text[1:] if text else None

    def record_text_outcome(self, context_features: List[float], text: str, outcome_score: float):
        if outcome_score < 0.4 or not text:
            return
        self._vocab.grow_from_text(text)
        token_ids = self._vocab.encode_text(text)
        if len(token_ids) < 3:
            return
        with self._data_lock:
            self._gen_pending.append(
                {
                    "context": context_features,
                    "token_ids": token_ids,
                    "score": outcome_score,
                }
            )
            if len(self._gen_pending) > 256:
                self._gen_pending = self._gen_pending[-256:]

    def train_generative_step(self) -> float:
        if not self._available or self._decoder is None:
            return 0.0
        with self._data_lock:
            if len(self._gen_pending) < 4:
                return 0.0
            batch = list(self._gen_pending[-32:])

        if self._vocab.size > self._decoder.word_embedding.num_embeddings:
            self._decoder.resize_vocab(self._vocab.size)
            self._gen_optimizer = torch.optim.Adam(self._decoder.parameters(), lr=5e-4)
        self._decoder.train()
        _fab = _get_fabric()

        max_seq = max(len(s["token_ids"]) + 1 for s in batch)
        max_seq = min(max_seq, _DECODER_MAX_LEN - 1)

        pad_id = self._vocab._word2idx[_PAD_TOKEN]
        input_batch = []
        target_batch = []
        ctx_batch = []

        for sample in batch:
            ids = sample["token_ids"][: max_seq - 1]
            inp_ids = [self._vocab.sos_id] + ids
            tgt_ids = ids + [self._vocab.eos_id]
            while len(inp_ids) < max_seq:
                inp_ids.append(pad_id)
                tgt_ids.append(pad_id)
            input_batch.append(inp_ids)
            target_batch.append(tgt_ids)
            ctx_batch.append(sample["context"])

        inp_t = (
            _fab.tensor(input_batch, dtype=torch.long)
            if _fab
            else torch.tensor(input_batch, dtype=torch.long)
        )
        tgt_t = (
            _fab.tensor(target_batch, dtype=torch.long)
            if _fab
            else torch.tensor(target_batch, dtype=torch.long)
        )
        ctx_t = _fab.tensor(ctx_batch) if _fab else torch.tensor(ctx_batch, dtype=torch.float32)

        logits = self._decoder(inp_t, ctx_t)
        logits_flat = logits.view(-1, logits.shape[-1])
        tgt_flat = tgt_t.view(-1)
        loss = F.cross_entropy(logits_flat, tgt_flat, ignore_index=pad_id)

        self._gen_optimizer.zero_grad()
        if _fab:
            _fab.scale_loss(loss).backward()
            nn.utils.clip_grad_norm_(self._decoder.parameters(), 1.0)
            _fab.scaler_step(self._gen_optimizer)
            _fab.scaler_update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self._decoder.parameters(), 1.0)
            self._gen_optimizer.step()

        self._decoder.eval()
        self._gen_train_steps += 1
        return loss.item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "train_steps": self._train_steps,
            "lexicon": self._lexicon.to_dict(),
            "gen_train_steps": self._gen_train_steps,
            "vocabulary": self._vocab.to_dict(),
            "phrase_dim": self._phrase_dim,
            "kg_dim": self._kg_dim,
        }
        if self._available and self._scorer is not None:
            state = self._scorer.state_dict()
            data["scorer_state"] = {k: v.tolist() for k, v in state.items()}
        if self._available and self._decoder is not None:
            data["decoder_state"] = {k: v.tolist() for k, v in self._decoder.state_dict().items()}
        if self._available and self._kg_encoder is not None:
            data["kg_encoder_state"] = {
                k: v.tolist() for k, v in self._kg_encoder.state_dict().items()
            }
        return data

    def from_dict(self, data: Dict[str, Any]):
        self._train_steps = data.get("train_steps", 0)
        self._gen_train_steps = data.get("gen_train_steps", 0)

        lexicon_data = data.get("lexicon")
        if lexicon_data and isinstance(lexicon_data, list):
            self._lexicon.from_dict(lexicon_data)

        vocab_data = data.get("vocabulary")
        if vocab_data and isinstance(vocab_data, dict):
            self._vocab.from_dict(vocab_data)

        saved_phrase_dim = data.get("phrase_dim", self._phrase_dim)
        saved_kg_dim = data.get("kg_dim", 0)
        dim_changed = saved_phrase_dim != self._phrase_dim or saved_kg_dim != self._kg_dim

        if not self._available:
            return

        _fab = _get_fabric()

        scorer_state = data.get("scorer_state")
        if scorer_state and self._scorer is not None and not dim_changed:
            try:
                restored = {k: torch.tensor(v) for k, v in scorer_state.items()}
                self._scorer.load_state_dict(restored)
                self._scorer.eval()
            except (RuntimeError, Exception):
                self._scorer = _CompositionScorerNet(input_dim=self._input_dim)
                if _fab:
                    self._scorer = _fab.register("composition_scorer", self._scorer)
                self._scorer.eval()
                self._optimizer = torch.optim.Adam(self._scorer.parameters(), lr=0.001)
                self._train_steps = 0
        elif dim_changed:
            self._scorer = _CompositionScorerNet(input_dim=self._input_dim)
            if _fab:
                self._scorer = _fab.register("composition_scorer", self._scorer)
            self._scorer.eval()
            self._optimizer = torch.optim.Adam(self._scorer.parameters(), lr=0.001)
            self._train_steps = 0

        decoder_state = data.get("decoder_state")
        if decoder_state and self._decoder is not None and not dim_changed:
            try:
                restored = {k: torch.tensor(v) for k, v in decoder_state.items()}
                self._decoder.load_state_dict(restored)
                self._decoder.eval()
            except (RuntimeError, Exception):
                self._decoder = _TransformerDecoderNet(
                    context_dim=self._full_context_dim,
                    vocab_size=self._vocab.size,
                    word_embed_dim=64,
                )
                if _fab:
                    self._decoder = _fab.register("generative_decoder", self._decoder)
                self._decoder.eval()
                self._gen_optimizer = torch.optim.Adam(self._decoder.parameters(), lr=5e-4)
                self._gen_train_steps = 0
        elif dim_changed:
            self._decoder = _TransformerDecoderNet(
                context_dim=self._full_context_dim, vocab_size=self._vocab.size, word_embed_dim=64
            )
            if _fab:
                self._decoder = _fab.register("generative_decoder", self._decoder)
            self._decoder.eval()
            self._gen_optimizer = torch.optim.Adam(self._decoder.parameters(), lr=5e-4)
            self._gen_train_steps = 0

        kg_state = data.get("kg_encoder_state")
        if kg_state and self._kg_encoder is not None and not dim_changed:
            try:
                restored = {k: torch.tensor(v) for k, v in kg_state.items()}
                self._kg_encoder.load_state_dict(restored)
            except (RuntimeError, Exception):
                self._kg_encoder = _KnowledgeContextEncoder(
                    embed_dim=self._phrase_dim, output_dim=self._kg_dim
                )
                if _fab:
                    self._kg_encoder = _fab.register("kg_context_encoder", self._kg_encoder)
        elif dim_changed and self._kg_encoder is not None:
            self._kg_encoder = _KnowledgeContextEncoder(
                embed_dim=self._phrase_dim, output_dim=self._kg_dim
            )
            if _fab:
                self._kg_encoder = _fab.register("kg_context_encoder", self._kg_encoder)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "available": self._available,
            "train_steps": self._train_steps,
            "learned_weight": round(self.learned_weight, 3),
            "lexicon_size": {
                "openers": len(self._lexicon.openers),
                "closers": len(self._lexicon.closers),
                "bridges": len(self._lexicon.bridges),
            },
            "total_phrases": self._lexicon.total_count(),
            "avg_phrase_outcome": round(self._lexicon.avg_outcome(), 3),
            "gen_train_steps": self._gen_train_steps,
            "generative_weight": round(self.generative_weight, 3),
            "vocab_size": self._vocab.size,
            "phrase_dim": self._phrase_dim,
            "kg_dim": self._kg_dim,
            "context_dim": self._context_dim,
            "full_context_dim": self._full_context_dim,
            "decoder_architecture": "transformer",
            "kg_grounded": self._kg_encoder is not None,
            "llm": self._llm.stats(),
        }
