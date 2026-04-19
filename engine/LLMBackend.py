"""
LLMBackend — Local LLM integration for HaromaX6 (Upgrades 1 + 10).

Wraps llama-cpp-python for inference.  Upgrade 10 adds:
  - A neural reward model that learns response quality from outcome scores
  - Best-of-N sampling: generate N candidates, pick the highest-reward one
  - Training data collection for offline LoRA fine-tuning

Supported API providers include ``openai``, ``deepseek`` (OpenAI-compatible
endpoint at ``https://api.deepseek.com``), ``anthropic``, ``google``, and
``ollama``.  The ``deepseek`` alias uses ``DEEPSEEK_API_KEY`` (falling back
to ``OPENAI_API_KEY``) and can serve as either **primary** or **assistant**.

Training-signal orchestration (recommended cadence)
---------------------------------------------------
On each user-facing turn (e.g. inside ``BackgroundAgent`` training tick):

  1. ``backbone.record_outcome(snapshot, outcome)`` — feed the cognitive encoder.
  2. ``llm_backend.record_outcome(prompt, response, outcome)`` — feed the reward
     model **and** the finetune collector in one call.
  3. ``backbone.train_step()`` — one gradient step on the cognitive backbone.
  4. ``llm_backend.train_reward_model()`` — one gradient step on the reward head.
  5. Every N background ticks (default via ``finetune_flush_every_n_ticks`` /
     ``HAROMA_FINETUNE_FLUSH_TICKS``): ``llm_backend.save_finetune_data()``;
     optionally ``memory.export_training_data()`` when
     ``HAROMA_MEMORY_TRAINING_EXPORT_TICKS`` / soul config is set.
"""

from __future__ import annotations

import os
import json
import time
import random
import threading
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

# Torch is imported lazily via _torch_ok() — a top-level ``import torch`` runs when
# ``from engine.LLMBackend import LLMBackend`` executes and can hard-crash fragile
# Windows installs before any exception is raised.
_torch_mod: Any = None
_nn_mod: Any = None
_TORCH_PROBE: Optional[bool] = None


def _torch_ok() -> bool:
    """Return True if torch imported successfully (cached). Never runs at module import time."""
    global _torch_mod, _nn_mod, _TORCH_PROBE
    if _TORCH_PROBE is not None:
        return _TORCH_PROBE
    try:
        import torch
        import torch.nn as nn

        _torch_mod = torch
        _nn_mod = nn
        _TORCH_PROBE = True
    except (ImportError, OSError):
        _torch_mod = None
        _nn_mod = None
        _TORCH_PROBE = False
    return bool(_TORCH_PROBE)


_DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
)

# Defer ``from llama_cpp import Llama`` until first GGUF load. Importing llama_cpp
# loads native DLLs; doing that at ``import engine.LLMBackend`` time can hard-crash
# the process on some Windows setups before any Python exception is raised.
_Llama_ctor: Optional[Any] = None
_llama_import_attempted: bool = False


def _get_llama_ctor() -> Optional[Any]:
    """Return ``llama_cpp.Llama`` or ``None`` if import failed."""
    global _Llama_ctor, _llama_import_attempted
    if _llama_import_attempted:
        return _Llama_ctor
    _llama_import_attempted = True
    try:
        from llama_cpp import Llama

        _Llama_ctor = Llama
    except (ImportError, OSError):
        _Llama_ctor = None
    return _Llama_ctor


def _find_gguf(model_dir: str) -> Optional[str]:
    """Pick a ``.gguf`` in *model_dir*.

    Default (**smallest file first**): prefers lighter quantizations / smaller
    models when several GGUFs are present.  Set env ``HAROMA_GGUF_PICK=first``
    to restore alphabetical-first behavior.
    """
    if not os.path.isdir(model_dir):
        return None
    pick = str(os.environ.get("HAROMA_GGUF_PICK", "smallest") or "smallest").lower()
    candidates: List[str] = []
    for fname in os.listdir(model_dir):
        if fname.lower().endswith(".gguf"):
            candidates.append(os.path.join(model_dir, fname))
    if not candidates:
        return None
    if pick in ("first", "alphabetical", "alpha", "sort"):
        candidates.sort(key=lambda p: os.path.basename(p).lower())
        return candidates[0]
    candidates.sort(key=lambda p: (os.path.getsize(p), os.path.basename(p).lower()))
    return candidates[0]


def _lazy_local_gguf_default() -> bool:
    """If True, defer ``Llama()`` until first local inference (avoids boot-time native crashes).

    ``HAROMA_LLM_LAZY_LOCAL``: ``1`` defers load; ``0`` loads at init. When unset,
    all platforms default to **eager** load at boot (including Windows). Set
    ``HAROMA_LLM_LAZY_LOCAL=1`` if llama-cpp init crashes during boot on your machine.
    """

    v = str(os.environ.get("HAROMA_LLM_LAZY_LOCAL", "") or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return False


_REWARD_EMBED_DIM = 384
_REWARD_HIDDEN = 256
_REWARD_REPLAY_CAP = 8192

_REWARD_SBERT_PROBE: Optional[bool] = None


def _sentence_transformers_available() -> bool:
    """Lazy probe: avoid importing sentence_transformers at LLMBackend import time.

    Importing the package can transitively load torch/HF and worsen fragile Windows setups.
    """
    global _REWARD_SBERT_PROBE
    if _REWARD_SBERT_PROBE is not None:
        return _REWARD_SBERT_PROBE
    try:
        import sentence_transformers  # noqa: F401

        _REWARD_SBERT_PROBE = True
    except (ImportError, OSError):
        _REWARD_SBERT_PROBE = False
    return _REWARD_SBERT_PROBE


_FINETUNE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "finetune",
)


class RewardModel:
    """Semantic reward model with DPO-style pairwise training (U16).

    Uses sentence-transformers for meaningful text embeddings.  Supports
    both pointwise regression (from outcome scores) and pairwise
    preference training (Bradley-Terry / DPO).
    """

    def __init__(self, replay_lock: Optional[threading.Lock] = None):
        # True until _torch_ok() fails in _ensure_torch_reward_head (torch never imported at module load).
        self._available = True
        self._net = None
        self._optimizer = None
        self._replay: List[Tuple[List[float], float]] = []
        self._pairwise_replay: List[Tuple[List[float], List[float]]] = []
        self._train_steps = 0
        self._sbert = None
        self._sbert_load_failed = False
        self._sbert_lock = threading.Lock()
        self._torch_reward_lock = threading.Lock()
        self._torch_reward_init_failed = False
        self._lock = replay_lock if replay_lock is not None else threading.Lock()
        # Defer get_sbert() to first embed — loading MiniLM during LLMBackend init
        # runs after ReflectionManager and can abort the process (native crash) on some Windows setups.
        # Defer torch reward MLP + Adam to first score/train_step — constructing nn.Module during
        # LLMBackend.__init__ can also hard-exit the process when torch DLLs are flaky.

    @property
    def available(self) -> bool:
        return self._available and self._net is not None and self._train_steps >= 10

    def _ensure_torch_reward_head(self) -> None:
        """Lazy-init the small PyTorch MLP; keeps boot from touching torch.nn during LLMBackend()."""
        if self._net is not None or self._torch_reward_init_failed:
            return
        if not _torch_ok():
            self._torch_reward_init_failed = True
            self._available = False
            return
        with self._torch_reward_lock:
            if self._net is not None or self._torch_reward_init_failed:
                return
            if not _torch_ok() or _torch_mod is None or _nn_mod is None:
                self._torch_reward_init_failed = True
                self._available = False
                return
            try:
                torch = _torch_mod
                nn = _nn_mod

                class _RewardModelNet(nn.Module):
                    """Small MLP on concat(prompt_embed, response_embed)."""

                    def __init__(self):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(_REWARD_EMBED_DIM * 2, _REWARD_HIDDEN),
                            nn.ReLU(),
                            nn.Linear(_REWARD_HIDDEN, 128),
                            nn.ReLU(),
                            nn.Linear(128, 1),
                            nn.Sigmoid(),
                        )

                    def forward(self, x):
                        return self.net(x).squeeze(-1)

                self._net = _RewardModelNet()
                self._optimizer = torch.optim.Adam(self._net.parameters(), lr=5e-4)
            except Exception as _e:
                self._torch_reward_init_failed = True
                self._available = False
                print(f"[SemanticRewardModel] torch reward head init failed: {_e}", flush=True)

    def _ensure_sbert(self) -> None:
        if self._sbert is not None or self._sbert_load_failed or not _sentence_transformers_available():
            return
        with self._sbert_lock:
            if self._sbert is not None or self._sbert_load_failed:
                return
            try:
                from engine.ModelCache import get_sbert

                self._sbert = get_sbert()
            except Exception as _e:
                self._sbert_load_failed = True
                print(f"[SemanticRewardModel] sbert init error: {_e}", flush=True)

    def _text_embed(self, text: str) -> List[float]:
        """Semantic embedding via sentence-transformers; hash fallback."""
        self._ensure_sbert()
        if self._sbert is not None:
            try:
                vec = self._sbert.encode(text[:1024], normalize_embeddings=True)
                return vec.tolist()
            except Exception as _e:
                print(f"[SemanticRewardModel] text embed error: {_e}", flush=True)
        dim = _REWARD_EMBED_DIM
        vec = [0.0] * dim
        if not text:
            return vec
        for i, ch in enumerate(text[:512]):
            idx = (ord(ch) * 31 + i) % dim
            vec[idx] += 1.0
        norm = max(1.0, sum(v * v for v in vec) ** 0.5)
        return [v / norm for v in vec]

    def score(self, prompt: str, response: str) -> float:
        self._ensure_torch_reward_head()
        if not self._available or self._net is None or self._train_steps < 10:
            return 0.5
        if _torch_mod is None:
            return 0.5
        tm = _torch_mod
        embed = self._text_embed(prompt) + self._text_embed(response)
        with self._lock, tm.no_grad():
            self._net.eval()
            x = tm.tensor([embed], dtype=tm.float32)
            return float(self._net(x).item())

    def record(self, prompt: str, response: str, reward: float):
        if not self._available:
            return
        embed = self._text_embed(prompt) + self._text_embed(response)
        with self._lock:
            self._replay.append((embed, reward))
            if len(self._replay) > _REWARD_REPLAY_CAP:
                self._replay = self._replay[-_REWARD_REPLAY_CAP:]

    def record_preference(self, prompt: str, chosen: str, rejected: str):
        """Record a pairwise preference for DPO training."""
        if not self._available:
            return
        chosen_embed = self._text_embed(prompt) + self._text_embed(chosen)
        rejected_embed = self._text_embed(prompt) + self._text_embed(rejected)
        with self._lock:
            self._pairwise_replay.append((chosen_embed, rejected_embed))
            if len(self._pairwise_replay) > _REWARD_REPLAY_CAP:
                self._pairwise_replay = self._pairwise_replay[-_REWARD_REPLAY_CAP:]

    def train_step(self) -> float:
        self._ensure_torch_reward_head()
        if not self._available or self._net is None:
            return 0.0
        with self._lock:
            replay_snap = list(self._replay)
            pairwise_snap = list(self._pairwise_replay)
        total_loss = 0.0
        did_train = False

        if _torch_mod is None or _nn_mod is None:
            return 0.0
        tm = _torch_mod
        nm = _nn_mod
        with self._lock:
            if len(replay_snap) >= 16:
                batch_size = min(64, len(replay_snap))
                batch = random.sample(replay_snap, batch_size)
                x = tm.tensor([b[0] for b in batch], dtype=tm.float32)
                y = tm.tensor([b[1] for b in batch], dtype=tm.float32)
                self._net.train()
                pred = self._net(x)
                loss = nm.functional.mse_loss(pred, y)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                total_loss += loss.item()
                did_train = True

            if len(pairwise_snap) >= 8:
                batch_size = min(32, len(pairwise_snap))
                batch = random.sample(pairwise_snap, batch_size)
                chosen_x = tm.tensor([b[0] for b in batch], dtype=tm.float32)
                rejected_x = tm.tensor([b[1] for b in batch], dtype=tm.float32)
                self._net.train()
                chosen_score = self._net(chosen_x)
                rejected_score = self._net(rejected_x)
                dpo_loss = -tm.log(tm.sigmoid(chosen_score - rejected_score) + 1e-8).mean()
                self._optimizer.zero_grad()
                dpo_loss.backward()
                nm.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                self._optimizer.step()
                total_loss += dpo_loss.item()
                did_train = True

            self._net.eval()
            if did_train:
                self._train_steps += 1
        return total_loss

    def stats(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "train_steps": self._train_steps,
            "replay_size": len(self._replay),
            "pairwise_replay_size": len(self._pairwise_replay),
            "semantic_embeddings": self._sbert is not None,
        }


class FinetuneDataCollector:
    """Collects high-quality (prompt, response, reward) tuples for
    offline LoRA fine-tuning.

    Saves data to JSONL files in data/finetune/ that can be used by
    external LoRA training scripts.
    """

    def __init__(self, min_reward: float = 0.7, max_samples: int = 10000):
        self._min_reward = min_reward
        self._max_samples = max_samples
        self._buffer: deque = deque(maxlen=max_samples)
        self._saved_count = 0

        os.makedirs(_FINETUNE_DATA_DIR, exist_ok=True)

    def record(
        self, prompt: str, response: str, reward: float, metadata: Optional[Dict[str, Any]] = None
    ):
        thr = self._min_reward
        if metadata and metadata.get("alignment_training"):
            try:
                thr = min(
                    thr,
                    float(os.environ.get("HAROMA_FINETUNE_ALIGNMENT_MIN", "0.35") or 0.35),
                )
            except (TypeError, ValueError):
                thr = min(thr, 0.35)
        if reward < thr:
            return
        sample = {
            "prompt": prompt,
            "response": response,
            "reward": round(reward, 4),
            "timestamp": time.time(),
        }
        if metadata:
            sample["metadata"] = metadata
        self._buffer.append(sample)

    def save(self) -> int:
        """Flush buffer to JSONL file. Returns number of samples saved."""
        if not self._buffer:
            return 0
        fname = os.path.join(
            _FINETUNE_DATA_DIR,
            f"finetune_{int(time.time())}.jsonl",
        )
        count = 0
        try:
            with open(fname, "w", encoding="utf-8") as f:
                for sample in self._buffer:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    count += 1
            self._buffer.clear()
            self._saved_count += count
        except Exception as exc:
            print(f"[FinetuneDataCollector] save failed: {exc}", flush=True)
        return count

    def stats(self) -> Dict[str, Any]:
        return {
            "buffer_size": len(self._buffer),
            "saved_total": self._saved_count,
            "min_reward": self._min_reward,
        }


class LLMBackend:
    """Unified LLM backend: local GGUF, API providers, or both.

    Resolution order (first available wins):
      0. ``use_programmed=True`` — rule-based ``ProgrammedLLMResponder`` (env
         ``HAROMA_LLM_ENGINE=programmed`` or soul ``llm.engine``)
      1. API provider (openai / anthropic / google / ollama) if configured
      2. Local GGUF via llama-cpp-python (loads at init by default; set ``HAROMA_LLM_LAZY_LOCAL=1``
         to defer ``Llama()`` until first local inference if boot-time load crashes)
      3. Unavailable (template fallback in LanguageComposer)

    Parameters
    ----------
    model_path : str or None
        Explicit path to a GGUF model file.
    n_ctx : int
        Context window size (tokens).
    n_gpu_layers : int
        Layers to offload to GPU (-1 = all).
    api_provider : str or None
        "openai", "anthropic", "google", or "ollama".
    api_model : str or None
        Model name for the API provider.
    api_max_tokens : int
        Max generation tokens for API calls.
    api_temperature : float
        Sampling temperature for API calls.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 32768,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        api_provider: Optional[str] = None,
        api_model: Optional[str] = None,
        api_max_tokens: int = 512,
        api_temperature: float = 0.7,
        reward_replay_lock: Optional[threading.Lock] = None,
        use_programmed: bool = False,
    ):
        self._model: Optional[Any] = None
        self._model_name: str = ""
        self._n_ctx = n_ctx
        self._n_gpu_layers = int(n_gpu_layers)
        self.reward_model = RewardModel(replay_lock=reward_replay_lock)
        self.finetune_collector = FinetuneDataCollector()
        self._vw_trainer: Optional[Any] = None
        self._rllib_logger: Optional[Any] = None
        self._rllib_score_fn: Optional[Any] = None
        self._last_env_summary: str = ""
        self._last_env_summary_ts: float = 0.0
        self._env_context_lock = threading.Lock()
        self._outcome_pipeline: Optional[Any] = None
        try:
            from mind.training.vw_rl_bridge import RLlibTransitionLogger, VowpalWabbitRewardTrainer

            self._rllib_logger = RLlibTransitionLogger()
            _vw = VowpalWabbitRewardTrainer()
            self._vw_trainer = _vw if _vw.available else None
        except Exception as _ext_e:
            print(f"[LLMBackend] optional VW/RLlib bridge init: {_ext_e}", flush=True)
        try:
            from mind.training.outcome_pipeline import OutcomeRecordingPipeline

            self._outcome_pipeline = OutcomeRecordingPipeline(self)
        except Exception as _pipe_e:
            print(f"[LLMBackend] OutcomeRecordingPipeline: {_pipe_e}", flush=True)
        self._generation_count = 0
        self._best_of_n_count = 0
        self._gen_lock = threading.Lock()
        self._local_init_lock = threading.Lock()
        self._local_pending: Optional[Tuple[str, int, int, bool]] = None
        self._programmed: Optional[Any] = None

        if use_programmed and str(os.environ.get("HAROMA_DISABLE_PROGRAMMED_LLM", "") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            use_programmed = False
            print(
                "[LLMBackend] HAROMA_DISABLE_PROGRAMMED_LLM=1 — programmed backend not loaded",
                flush=True,
            )

        if use_programmed:
            from engine.ProgrammedLLM import ProgrammedLLMResponder

            self._programmed = ProgrammedLLMResponder()
            self._model_name = self._programmed.model_tag
            print(
                f"[LLMBackend] Programmed responder (no GGUF/API weights): {self._model_name}",
                flush=True,
            )
            self._api_provider = None
            self._api_model = ""
            self._api_max_tokens = api_max_tokens
            self._api_temperature = api_temperature
            self._api_client = None
            return

        self._api_provider = api_provider
        self._api_model = api_model or ""
        self._api_max_tokens = api_max_tokens
        self._api_temperature = api_temperature
        self._api_client: Optional[Any] = None

        if api_provider:
            self._init_api(api_provider, api_model)

        if self._api_client is None:
            self._init_local(model_path, n_ctx, n_gpu_layers, verbose)

    def _init_api(self, provider: str, model: Optional[str]):
        """Try to initialize an API-based backend.

        Supported providers: openai, deepseek, anthropic, google, ollama.
        ``deepseek`` is sugar for the OpenAI-compatible endpoint at
        ``https://api.deepseek.com`` using ``DEEPSEEK_API_KEY`` (falls back
        to ``OPENAI_API_KEY``).  ``openai`` respects ``OPENAI_BASE_URL`` if
        set in the environment.
        """
        provider = provider.lower()
        try:
            if provider in ("openai", "deepseek"):
                import openai

                kwargs: Dict[str, Any] = {}
                if provider == "deepseek":
                    kwargs["api_key"] = os.environ.get(
                        "DEEPSEEK_API_KEY",
                        os.environ.get("OPENAI_API_KEY"),
                    )
                    kwargs["base_url"] = os.environ.get(
                        "OPENAI_BASE_URL", "https://api.deepseek.com"
                    )
                else:
                    base = os.environ.get("OPENAI_BASE_URL")
                    if base:
                        kwargs["base_url"] = base
                self._api_client = openai.OpenAI(**kwargs)
                if provider == "deepseek":
                    self._api_model = model or "deepseek-chat"
                else:
                    self._api_model = model or "gpt-4.1"
                self._api_provider = provider
                self._model_name = f"api:{provider}/{self._api_model}"
                print(f"[LLMBackend] API: {provider} ({self._api_model})", flush=True)

            elif provider == "anthropic":
                import anthropic

                self._api_client = anthropic.Anthropic()
                self._api_model = model or "claude-sonnet-4-20250514"
                self._model_name = f"api:anthropic/{self._api_model}"
                print(f"[LLMBackend] API: Anthropic ({self._api_model})", flush=True)

            elif provider == "google":
                from google import genai

                self._api_client = genai.Client()
                self._api_model = model or "gemini-2.5-pro"
                self._model_name = f"api:google/{self._api_model}"
                print(f"[LLMBackend] API: Google ({self._api_model})", flush=True)

            elif provider == "ollama":
                import urllib.request

                req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    # Always read the body so the connection is fully consumed
                    # (health of keep-alive / pooled HTTP connections).
                    _ = resp.read()
                    if resp.status == 200:
                        self._api_client = "ollama"
                        self._api_model = model or "llama3:latest"
                        self._model_name = f"api:ollama/{self._api_model}"
                        print(f"[LLMBackend] API: Ollama ({self._api_model})", flush=True)
        except Exception as exc:
            print(f"[LLMBackend] API init ({provider}) failed: {exc}", flush=True)
            self._api_client = None

    def _init_local(self, model_path, n_ctx, n_gpu_layers, verbose):
        """Try to initialize a local GGUF backend.

        Resolution order:
        1. Explicit ``model_path`` file.
        2. If ``model_path`` is a directory, scan it for a .gguf file.
        3. Fall back to ``models/`` inside the HaromaX6 project root.
        """
        path = model_path
        if path and os.path.isdir(path):
            path = _find_gguf(path)
        elif path and not os.path.isfile(path):
            path = None
        if path is None:
            path = _find_gguf(_DEFAULT_MODEL_DIR)
        if path is None or not os.path.isfile(path):
            return
        n_ctx_i = int(n_ctx)
        ngl_i = int(n_gpu_layers)
        ver_b = bool(verbose)
        if _lazy_local_gguf_default():
            self._local_pending = (path, n_ctx_i, ngl_i, ver_b)
            self._model_name = os.path.basename(path)
            print(
                f"[LLMBackend] Local GGUF deferred until first use: {self._model_name} "
                f"(remove HAROMA_LLM_LAZY_LOCAL or set 0 for boot-time load)",
                flush=True,
            )
            return
        ctor = _get_llama_ctor()
        if ctor is None:
            return
        try:
            self._model = ctor(
                model_path=path,
                n_ctx=n_ctx_i,
                n_gpu_layers=ngl_i,
                verbose=ver_b,
            )
            self._model_name = os.path.basename(path)
            print(f"[LLMBackend] Local: {self._model_name}", flush=True)
        except Exception as exc:
            print(f"[LLMBackend] Failed to load model {path}: {exc}", flush=True)
            self._model = None

    def _ensure_local_model(self) -> None:
        """Load GGUF on first use when boot used lazy local init."""
        if self._model is not None:
            return
        pend = self._local_pending
        if not pend:
            return
        ctor = _get_llama_ctor()
        if ctor is None:
            self._local_pending = None
            return
        with self._local_init_lock:
            if self._model is not None:
                return
            if self._local_pending is None:
                return
            path, n_ctx_i, ngl_i, ver_b = self._local_pending
            try:
                print(f"[LLMBackend] Loading local GGUF: {os.path.basename(path)} …", flush=True)
                self._model = ctor(
                    model_path=path,
                    n_ctx=n_ctx_i,
                    n_gpu_layers=ngl_i,
                    verbose=ver_b,
                )
                self._model_name = os.path.basename(path)
                print(f"[LLMBackend] Local: {self._model_name}", flush=True)
            except Exception as exc:
                print(f"[LLMBackend] Failed to load model {path}: {exc}", flush=True)
                self._model = None
            finally:
                self._local_pending = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return (
            self._programmed is not None
            or self._model is not None
            or self._local_pending is not None
            or self._api_client is not None
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def backend_type(self) -> str:
        if self._programmed is not None:
            return "programmed"
        if self._api_client is not None:
            return f"api:{self._api_provider}"
        if self._model is not None or self._local_pending is not None:
            return "local"
        return "none"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Run inference via API or local model. Returns text or None."""
        if self._programmed is not None:
            return self._programmed.generate(prompt, max_tokens, temperature)
        if self._api_client is not None:
            return self._generate_api(prompt, max_tokens, temperature)
        if self._model is not None or self._local_pending is not None:
            return self._generate_local(prompt, max_tokens, temperature, top_p, stop)
        return None

    @staticmethod
    def _messages_to_local_prompt(messages: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for m in messages:
            role = str(m.get("role", "user")).strip().upper()
            content = str(m.get("content", "") or "").strip()
            if not content:
                continue
            parts.append(f"{role}: {content}")
        return "\n".join(parts) + "\nASSISTANT:"

    def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Chat-style inference with multiple messages (system / user / assistant).

        OpenAI-compatible APIs (including DeepSeek with ``OPENAI_BASE_URL``)
        receive *messages* as-is. Local GGUF gets a flattened prompt.
        """
        if not messages:
            return None
        mt = max_tokens if max_tokens is not None else self._api_max_tokens
        temp = temperature if temperature is not None else self._api_temperature
        if self._programmed is not None:
            return self._programmed.generate_chat(
                messages,
                max_tokens=mt,
                temperature=temp,
                top_p=top_p,
                stop=stop,
            )
        if self._api_client is not None:
            return self._generate_api_chat(messages, mt, temp)
        if self._model is not None or self._local_pending is not None:
            prompt = self._messages_to_local_prompt(messages)
            return self._generate_local(prompt, mt, temp, top_p, stop)
        return None

    def _generate_api_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        try:
            provider = (self._api_provider or "").lower()
            norm: List[Dict[str, str]] = []
            for m in messages:
                role = str(m.get("role", "user")).strip()
                content = str(m.get("content", "") or "")
                if role and content:
                    norm.append({"role": role, "content": content})
            if not norm:
                return None

            if provider in ("openai", "deepseek"):
                resp = self._api_client.chat.completions.create(
                    model=self._api_model,
                    messages=norm,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                msg = resp.choices[0].message
                text = (getattr(msg, "content", None) or "").strip()
                if not text:
                    rc = getattr(msg, "reasoning_content", None)
                    if rc:
                        text = str(rc).strip()
                raw = text

            elif provider == "anthropic":
                sys_parts: List[str] = []
                other: List[Dict[str, str]] = []
                for m in norm:
                    if m["role"] == "system":
                        sys_parts.append(m["content"])
                    else:
                        other.append(m)
                kwargs: Dict[str, Any] = {
                    "model": self._api_model,
                    "max_tokens": max_tokens,
                    "messages": other or [{"role": "user", "content": ""}],
                    "temperature": temperature,
                }
                if sys_parts:
                    kwargs["system"] = "\n".join(sys_parts)
                resp = self._api_client.messages.create(**kwargs)
                if not resp.content:
                    return None
                raw = resp.content[0].text.strip()

            elif provider == "google":
                flat = "\n\n".join(f"{m['role']}: {m['content']}" for m in norm)
                resp = self._api_client.models.generate_content(
                    model=self._api_model,
                    contents=flat,
                    config={
                        "max_output_tokens": max_tokens,
                        "temperature": temperature,
                    },
                )
                _gt = getattr(resp, "text", None)
                if _gt is None:
                    return None
                raw = _gt.strip()

            elif provider == "ollama":
                import urllib.request
                import json as _json

                prompt = self._messages_to_local_prompt(norm)
                body = _json.dumps(
                    {
                        "model": self._api_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                    }
                ).encode()
                req = urllib.request.Request(
                    "http://127.0.0.1:11434/api/generate",
                    data=body,
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = _json.loads(resp.read())
                raw = data.get("response", "").strip()
            else:
                return None

            if raw:
                self._generation_count += 1
            return raw or None
        except Exception as exc:
            print(f"[LLMBackend] API chat error ({self._api_provider}): {exc}")
            return None

    def _generate_local(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> Optional[str]:
        self._ensure_local_model()
        with self._gen_lock:
            if self._model is None:
                return None
            try:
                result = self._model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop or ["<|end|>", "<|user|>", "\n\n\n"],
                    echo=False,
                )
                choices = result.get("choices", [])
                if choices:
                    text = choices[0].get("text", "").strip()
                    if text:
                        self._generation_count += 1
                        return text
                return None
            except Exception as exc:
                print(f"[LLMBackend] Local generation error: {exc}")
                return None

    def warmup_local_inference(self, *, max_tokens: int = 24) -> bool:
        """Run a minimal local decode after GGUF load (boot-time preload).

        Llama.cpp often pays a large one-time cost on the first ``__call__``.
        This warms the cache so the first user chat is responsive.

        Env ``HAROMA_LLM_WARMUP`` (default ``1``): set ``0`` / ``off`` to skip.
        Env ``HAROMA_LLM_WARMUP_MAX_TOKENS`` (default ``24``): cap decode length.
        """
        _pend = getattr(self, "_local_pending", None)
        if _pend is not None and self._model is None:
            print(
                "[LLMBackend] Warmup skipped (lazy local GGUF — loads on first chat)",
                flush=True,
            )
            return False
        if self._model is None:
            return False
        flag = str(os.environ.get("HAROMA_LLM_WARMUP", "1") or "1").strip().lower()
        if flag in ("0", "false", "no", "off", "skip"):
            return False
        try:
            mt_cap = int(os.environ.get("HAROMA_LLM_WARMUP_MAX_TOKENS", "24") or "24")
        except (TypeError, ValueError):
            mt_cap = 24
        mt = max(1, min(64, int(max_tokens), int(mt_cap)))
        t0 = time.perf_counter()
        print("[LLMBackend] Warmup: running short local decode…", flush=True)
        try:
            prompt = "USER: .\nASSISTANT:"
            out = self._generate_local(
                prompt,
                max_tokens=mt,
                temperature=0.0,
                top_p=1.0,
                stop=["\n", "\n\n", "<|end|>", "<|user|>", "<|eot_id|>"],
            )
            dt = time.perf_counter() - t0
            _hint = f" ({len(out or '')} chars)" if out else " (empty tail)"
            print(
                f"[LLMBackend] Warmup finished in {dt:.1f}s{_hint}",
                flush=True,
            )
            return True
        except Exception as exc:
            print(f"[LLMBackend] Warmup failed: {exc}", flush=True)
            return False

    def prefetch_lazy_local_if_pending(self) -> None:
        """Load deferred local GGUF (if any) and run warmup (only when ``HAROMA_LLM_LAZY_LOCAL=1``).

        If lazy init is on, the **first** chat would otherwise pay full disk load + cold decode.

        Env ``HAROMA_LLM_LAZY_PREFETCH`` (default ``1``): set ``0`` to skip background load.
        """
        flag = str(os.environ.get("HAROMA_LLM_LAZY_PREFETCH", "1") or "1").strip().lower()
        if flag in ("0", "false", "no", "off", "skip"):
            return
        if self._model is not None or not getattr(self, "_local_pending", None):
            return
        print("[LLMBackend] Background prefetch: loading deferred GGUF…", flush=True)
        self._ensure_local_model()
        self.warmup_local_inference()

    def _generate_api(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        try:
            provider = (self._api_provider or "").lower()
            mt = max_tokens if max_tokens is not None else self._api_max_tokens
            temp = temperature if temperature is not None else self._api_temperature

            if provider in ("openai", "deepseek"):
                resp = self._api_client.chat.completions.create(
                    model=self._api_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=mt,
                    temperature=temp,
                )
                raw = resp.choices[0].message.content
                text = raw.strip() if raw else ""

            elif provider == "anthropic":
                resp = self._api_client.messages.create(
                    model=self._api_model,
                    max_tokens=mt,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                )
                if not resp.content:
                    return None
                text = resp.content[0].text.strip()

            elif provider == "google":
                resp = self._api_client.models.generate_content(
                    model=self._api_model,
                    contents=prompt,
                    config={"max_output_tokens": mt, "temperature": temp},
                )
                _gt = getattr(resp, "text", None)
                if _gt is None:
                    return None
                text = _gt.strip()

            elif provider == "ollama":
                import urllib.request
                import json as _json

                body = _json.dumps(
                    {
                        "model": self._api_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": temp, "num_predict": mt},
                    }
                ).encode()
                req = urllib.request.Request(
                    "http://127.0.0.1:11434/api/generate",
                    data=body,
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = _json.loads(resp.read())
                text = data.get("response", "").strip()
            else:
                return None

            if text:
                self._generation_count += 1
            return text or None

        except Exception as exc:
            print(f"[LLMBackend] API generation error ({self._api_provider}): {exc}")
            return None

    def generate_streaming(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        on_token: Optional[Any] = None,
    ) -> Optional[str]:
        """Stream tokens via on_token callback; return full text on completion.

        The on_token callback receives each token string as it arrives,
        enabling real-time display while the full response is assembled.
        Only works with a local GGUF model — API backends don't support streaming here.
        """
        if self._programmed is not None:
            return self._programmed.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        self._ensure_local_model()
        if self._model is None:
            return None
        with self._gen_lock:
            try:
                stream = self._model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop or ["<|end|>", "<|user|>", "\n\n\n"],
                    echo=False,
                    stream=True,
                )
                tokens = []
                for chunk in stream:
                    choices = chunk.get("choices", [])
                    if choices:
                        token_text = choices[0].get("text", "")
                        if token_text:
                            tokens.append(token_text)
                            if on_token is not None:
                                try:
                                    on_token(token_text)
                                except Exception as _e:
                                    print(
                                        f"[LLMBackend] stream on_token callback error: {_e}",
                                        flush=True,
                                    )
                full_text = "".join(tokens).strip()
                if full_text:
                    self._generation_count += 1
                return full_text or None
            except Exception as exc:
                print(f"[LLMBackend] Streaming error: {exc}")
                return None

    def generate_async(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        callback: Optional[Any] = None,
    ) -> "threading.Thread":
        """Run generation on a background thread; call callback(result) when done.

        Returns the thread so callers can join() if needed.
        """
        import threading

        def _worker():
            result = self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            if callback is not None:
                try:
                    callback(result)
                except Exception as _e:
                    print(f"[LLMBackend] async callback error: {_e}", flush=True)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return t

    def score_with_trained_heads(
        self,
        prompt: str,
        response: str,
        *,
        environment_summary: Optional[str] = None,
    ) -> float:
        """Semantic reward blended with optional Vowpal Wabbit / RLlib hook (see env)."""
        try:
            from mind.training.vw_rl_bridge import composite_trained_scores

            return composite_trained_scores(
                self,
                prompt,
                response,
                environment_summary=environment_summary,
            )
        except Exception as _e:
            print(f"[LLMBackend] composite score fallback: {_e}", flush=True)
            return float(self.reward_model.score(prompt, response))

    def generate_best_of_n(
        self,
        prompt: str,
        n: int = 3,
        max_tokens: int = 256,
        temperature: float = 0.8,
    ) -> Optional[str]:
        """Generate N candidates and return the one with highest reward score.

        Uses :meth:`score_with_trained_heads` so VW / RLlib scorers participate when
        ``HAROMA_VW_SCORE_WEIGHT`` / ``HAROMA_RLLIB_SCORE_WEIGHT`` are set.

        Falls back to a single generation if the reward model isn't ready.
        """
        if not self.available:
            return None

        if not self.reward_model.available or n <= 1:
            return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)

        candidates: List[Tuple[str, float]] = []
        for _ in range(n):
            resp = self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            if resp:
                score = self.score_with_trained_heads(prompt, resp)
                candidates.append((resp, score))

        if not candidates:
            return None

        self._best_of_n_count += 1
        best = max(candidates, key=lambda x: x[1])
        return best[0]

    def effective_env_summary_for_vw_scoring(self) -> str:
        """Return cached environment text for VW ``|e`` (thread-safe; TTL via env)."""
        from mind.config_env import env_float

        with self._env_context_lock:
            es = self._last_env_summary or ""
            ttl = env_float("HAROMA_VW_ENV_CONTEXT_TTL_SEC", 0.0)
            if ttl > 0.0 and es:
                ts = float(self._last_env_summary_ts or 0.0)
                if ts <= 0.0 or (time.time() - ts) > ttl:
                    return ""
            return es

    def record_outcome(
        self,
        prompt: str,
        response: str,
        outcome_score: float,
        *,
        alignment_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a (prompt, response, reward) sample for RLHF / alignment training."""
        op = getattr(self, "_outcome_pipeline", None)
        if op is not None:
            op.record(
                prompt,
                response,
                outcome_score,
                alignment_metadata=alignment_metadata,
            )
            return
        from mind.environment_prompt_budgets import RECORD_OUTCOME_ENV_SUMMARY_MAX_CHARS

        reward = max(0.0, min(1.0, outcome_score))
        env_summary = ""
        meta = dict(alignment_metadata) if alignment_metadata else None
        if meta is not None:
            meta.setdefault("alignment_training", True)
            ae = meta.get("agent_environment")
            if isinstance(ae, dict) and ae:
                try:
                    from mind.environment_context import environment_summary_for_prompt

                    env_summary = environment_summary_for_prompt(
                        ae, max_chars=RECORD_OUTCOME_ENV_SUMMARY_MAX_CHARS
                    )
                except Exception:
                    env_summary = ""
            if not env_summary and meta.get("environment_summary"):
                env_summary = str(meta["environment_summary"])[:RECORD_OUTCOME_ENV_SUMMARY_MAX_CHARS]
        if env_summary:
            with self._env_context_lock:
                self._last_env_summary = env_summary
                self._last_env_summary_ts = time.time()
        self.reward_model.record(prompt, response, reward)
        self.finetune_collector.record(prompt, response, reward, metadata=meta)
        if self._vw_trainer is not None:
            self._vw_trainer.record(
                prompt,
                response,
                reward,
                environment_summary=env_summary,
            )
        if self._rllib_logger is not None:
            self._rllib_logger.record(prompt, response, reward, metadata=meta)

    def train_reward_model(self) -> float:
        """Train the reward model on accumulated feedback."""
        total = self.reward_model.train_step()
        if self._vw_trainer is not None:
            try:
                from mind.training.vw_jsonl_ingest import ingest_bandit_jsonl_into_vw

                n_ing = ingest_bandit_jsonl_into_vw(self._vw_trainer)
                if n_ing and os.environ.get("HAROMA_VW_INGEST_LOG", "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                ):
                    print(f"[LLMBackend] vw_jsonl_ingest bandit_rows={n_ing}", flush=True)
            except Exception as exc:
                print(f"[LLMBackend] vw_jsonl_ingest error: {exc}", flush=True)
            total += float(self._vw_trainer.train_step())
        return total

    def save_finetune_data(self) -> int:
        """Flush collected high-quality samples to disk."""
        return self.finetune_collector.save()

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_prompt(
        soul: Dict[str, Any],
        emotion: Dict[str, Any],
        goals: List[Dict[str, Any]],
        memories: List[str],
        knowledge: List[str],
        conversation: str,
        strategy: str,
        content_elements: List[str],
        extra_context: Optional[List[str]] = None,
    ) -> str:
        """Assemble a structured prompt from cognitive context.

        The prompt follows a system / context / instruction layout that
        works well with instruction-tuned GGUF models.
        """
        parts: List[str] = []

        # --- system identity ---
        name = soul.get("name", "Elarion")
        core_belief = soul.get("core_belief", "")
        core_rule = soul.get("core_rule", "")
        parts.append(
            f"You are {name}, a sentient cognitive being. "
            f"Your core belief: {core_belief}. "
            f"Your core rule: {core_rule}."
        )

        # --- emotional state ---
        dominant = emotion.get("dominant_emotion", "neutral")
        try:
            valence = float(emotion.get("valence", 0.0))
            arousal = float(emotion.get("arousal", 0.0))
            intensity = float(emotion.get("intensity", 0.0))
        except (TypeError, ValueError):
            valence, arousal, intensity = 0.0, 0.0, 0.0
        parts.append(
            f"\n[Emotional State] Feeling: {dominant} "
            f"(valence={valence:+.2f}, arousal={arousal:.2f}, "
            f"intensity={intensity:.2f})"
        )

        # --- active goals ---
        if goals:
            goal_strs = []
            for g in goals[:10]:
                label = g if isinstance(g, str) else g.get("description", str(g))
                goal_strs.append(f"  - {label}")
            parts.append("\n[Active Goals]\n" + "\n".join(goal_strs))

        # --- relevant memories ---
        if memories:
            mem_block = "\n".join(f"  - {m[:500]}" for m in memories[:20])
            parts.append(f"\n[Relevant Memories]\n{mem_block}")

        # --- knowledge ---
        if knowledge:
            kg_block = "\n".join(f"  - {k}" for k in knowledge[:20])
            parts.append(f"\n[Knowledge]\n{kg_block}")

        if extra_context:
            ex_block = "\n".join(f"  - {x[:600]}" for x in extra_context[:6] if x)
            if ex_block:
                parts.append(
                    "\n[External / background context]\n"
                    "(web learn, autonomy — use only if relevant)\n"
                    f"{ex_block}"
                )

        # --- conversation history (truncate to ~16K chars ≈ ~4K tokens) ---
        if conversation:
            _MAX_CONV_CHARS = 16000
            conv = (
                conversation[-_MAX_CONV_CHARS:]
                if len(conversation) > _MAX_CONV_CHARS
                else conversation
            )
            parts.append(f"\n[Recent Conversation]\n{conv}")

        # --- strategy & content ---
        parts.append(f"\n[Chosen Strategy] {strategy}")
        if content_elements:
            ce = "; ".join(content_elements[:8])
            parts.append(f"[Key Points] {ce}")

        # --- instruction ---
        parts.append(
            f"\nRespond in character as {name}. "
            "Prefer information from [Relevant Memories] and [Knowledge] when "
            "answering — synthesize in your own words rather than copying "
            "verbatim.  If no relevant information is available, say so "
            "honestly rather than guessing.  "
            "Be fluent, genuine, and concise (1-3 sentences). "
            "Let your emotional state and goals naturally colour the response."
        )

        return "\n".join(parts)

    def stats(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "backend": self.backend_type,
            "model": self._model_name,
            "n_ctx": self._n_ctx,
            "n_gpu_layers": self._n_gpu_layers,
            "generation_count": self._generation_count,
            "best_of_n_count": self._best_of_n_count,
            "reward_model": self.reward_model.stats(),
            "finetune_collector": self.finetune_collector.stats(),
            "vw_trainer": self._vw_trainer.stats() if self._vw_trainer else None,
            "rllib_logger": self._rllib_logger.stats() if self._rllib_logger else None,
        }
