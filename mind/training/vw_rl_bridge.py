"""Optional Vowpal Wabbit + RLlib-oriented training hooks for HaromaX6.

Vowpal Wabbit (``vowpalwabbit`` package): online squared-loss regression on
``(prompt, response[, environment]) -> reward`` when ``HAROMA_VW_REWARD=1``.
Optional host context is hashed in namespace ``|e`` (compact environment summary)
so predictions are contextual when ``agent_environment`` is recorded via
``LLMBackend.record_outcome``.

RLlib: does not run Ray inside the agent loop by default (heavy dependency).
When ``HAROMA_RLLIB_LOG_TRANSITIONS=1``, append JSON lines compatible with
offline RL ingestion (prompt/response/reward as text bandit-style steps).

Install extras::

    pip install vowpalwabbit
    pip install 'ray[rllib]'

Env
---
* ``HAROMA_VW_REWARD`` — enable VW learner (``1``/``true``).
* ``HAROMA_VW_OPTS`` — extra CLI args for ``pyvw.vw(...)`` (default: squared loss).
* ``HAROMA_RLLIB_LOG_TRANSITIONS`` — append JSONL transitions (``1``/``true``).
* ``HAROMA_RLLIB_TRANSITIONS_PATH`` — output file (default ``data/rllib/transitions.jsonl``).

Scoring (best-of-N, etc.)
-------------------------
* ``HAROMA_VW_SCORE_WEIGHT`` — blend VW ``predict`` into the semantic reward score
  in ``[0, 1]`` (``0`` = torch only; e.g. ``0.6`` favors VW after it has learned).
* ``HAROMA_RLLIB_SCORE_FN`` — ``importable.module:callable`` returning
  ``float`` for ``(prompt, response)`` from your RLlib / offline training wrapper.
* ``HAROMA_RLLIB_SCORE_WEIGHT`` — blend factor for that callable (``0`` = off).

Mitigations
-----------
* ``HAROMA_VW_ENV_CONTEXT_TTL_SEC`` — after this many seconds, cached environment
  text is not passed to VW at scoring time (``0`` = no expiry). Reduces stale
  ``|e`` context when the host does not send fresh ``agent_environment`` every turn.
* ``HAROMA_RLLIB_LOG_FULL_AGENT_ENV`` — when not set / ``0``, JSONL ``info`` omits
  the full ``agent_environment`` blob (flat fingerprint/domain/version still logged).
* ``HAROMA_RLLIB_ENV_SUMMARY_LOG_CHARS`` — max chars for ``environment_summary``
  in JSONL (default ``2000``).
* ``HAROMA_RLLIB_LOG_FULL_ALIGNMENT_DIAG`` — when ``1``, keep full ``alignment`` dict
  in JSONL ``info``; otherwise store a compact summary (default: summary only).
"""

from __future__ import annotations

import importlib
import json
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from mind.config_env import env_float, env_int, env_truthy
from mind.environment_prompt_budgets import VW_ENV_NAMESPACE_MAX_CHARS

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
_DEFAULT_RLLIB_LOG = os.path.join(_PROJECT_ROOT, "data", "rllib", "transitions.jsonl")


def _sanitize_text(s: str, max_chars: int) -> str:
    t = " ".join((s or "").replace("|", " ").replace("\n", " ").split())
    return t[:max_chars]


def load_rllib_score_callable() -> Optional[Callable[..., Any]]:
    """Load ``HAROMA_RLLIB_SCORE_FN`` = ``some.module:score_fn`` (optional)."""
    spec = str(os.environ.get("HAROMA_RLLIB_SCORE_FN", "") or "").strip()
    if not spec or ":" not in spec:
        return None
    mod_name, _, fn_name = spec.partition(":")
    mod_name = mod_name.strip()
    fn_name = fn_name.strip()
    if not mod_name or not fn_name:
        return None
    try:
        m = importlib.import_module(mod_name)
        fn = getattr(m, fn_name, None)
        if callable(fn):
            return fn
    except Exception as exc:
        print(f"[RLlibScore] import {spec!r} failed: {exc}", flush=True)
    return None


def composite_trained_scores(
    llm_backend: Any,
    prompt: str,
    response: str,
    *,
    environment_summary: Optional[str] = None,
) -> float:
    """Blend PyTorch reward with optional VW + optional RLlib hook (env-weighted).

    When *environment_summary* is omitted, uses
    :meth:`engine.LLMBackend.effective_env_summary_for_vw_scoring` when present;
    otherwise falls back to cached fields + TTL (for lightweight test doubles).
    """
    s = float(llm_backend.reward_model.score(prompt, response))
    es = environment_summary
    if es is None:
        getter = getattr(llm_backend, "effective_env_summary_for_vw_scoring", None)
        if callable(getter):
            es = getter()
        else:
            es = str(getattr(llm_backend, "_last_env_summary", "") or "")
            ttl = env_float("HAROMA_VW_ENV_CONTEXT_TTL_SEC", 0.0)
            if ttl > 0.0 and es:
                ts = float(getattr(llm_backend, "_last_env_summary_ts", 0.0) or 0.0)
                if ts <= 0.0 or (time.time() - ts) > ttl:
                    es = ""
    vw_w = max(0.0, min(1.0, env_float("HAROMA_VW_SCORE_WEIGHT", 0.0)))
    vw_t = getattr(llm_backend, "_vw_trainer", None)
    if vw_w > 0 and vw_t is not None:
        vp = vw_t.predict(prompt, response, environment_summary=es)
        if vp is not None:
            s = (1.0 - vw_w) * s + vw_w * float(vp)
    rl_w = max(0.0, min(1.0, env_float("HAROMA_RLLIB_SCORE_WEIGHT", 0.0)))
    if rl_w > 0:
        fn = getattr(llm_backend, "_rllib_score_fn", None)
        if fn is None:
            fn = load_rllib_score_callable()
            try:
                llm_backend._rllib_score_fn = fn
            except Exception:
                fn = None
        if fn is not None:
            try:
                rp = float(fn(prompt, response))
                rp = max(0.0, min(1.0, rp))
                s = (1.0 - rl_w) * s + rl_w * rp
            except Exception as exc:
                print(f"[RLlibScore] callable error: {exc}", flush=True)
    return max(0.0, min(1.0, s))


class VowpalWabbitRewardTrainer:
    """Online VW regression on hashed text features (optional dependency)."""

    def __init__(self) -> None:
        self._vw: Any = None
        self._lock = threading.Lock()
        self._pending: deque[str] = deque(maxlen=8192)
        self._learned = 0
        self._last_loss_sum = 0.0
        self._last_batches = 0

        if not env_truthy("HAROMA_VW_REWARD", False):
            return

        # Squared loss on [0, 1] labels: use identity (default link), not logistic.
        opts = os.environ.get("HAROMA_VW_OPTS") or "--loss_function squared --quiet"
        try:
            from vowpalwabbit import pyvw

            self._vw = pyvw.vw(opts)
            print(f"[VowpalWabbitRewardTrainer] initialized: {opts!r}", flush=True)
        except Exception as exc:
            print(f"[VowpalWabbitRewardTrainer] vowpalwabbit unavailable: {exc}", flush=True)
            self._vw = None

    @property
    def available(self) -> bool:
        return self._vw is not None

    def record(
        self,
        prompt: str,
        response: str,
        reward: float,
        *,
        environment_summary: str = "",
    ) -> None:
        if not self._vw:
            return
        try:
            lab = max(0.0, min(1.0, float(reward)))
        except (TypeError, ValueError):
            lab = 0.5
        p = _sanitize_text(prompt, 600)
        r = _sanitize_text(response, 600)
        line = f"{lab} |p {p} |q {r}"
        if environment_summary:
            line += f" |e {_sanitize_text(environment_summary, VW_ENV_NAMESPACE_MAX_CHARS)}"
        with self._lock:
            self._pending.append(line)

    def train_step(self) -> float:
        """Learn up to 128 pending examples; return mean absolute residual proxy."""
        if not self._vw:
            return 0.0
        batch: List[str] = []
        with self._lock:
            n = min(128, len(self._pending))
            for _ in range(n):
                batch.append(self._pending.popleft())
        if not batch:
            return 0.0

        err_sum = 0.0
        n_ok = 0
        vw = self._vw
        for line in batch:
            try:
                parts = line.split(None, 1)
                try:
                    lab = float(parts[0])
                except (IndexError, ValueError):
                    lab = 0.5
                infer = parts[1] if len(parts) > 1 else ""
                pred = vw.predict(infer) if infer else 0.0
                if isinstance(pred, list):
                    pred = pred[0] if pred else 0.0
                ex = vw.parse(line)
                vw.learn(ex)
                ex.finish()
                err_sum += abs(float(pred) - lab)
                n_ok += 1
            except Exception:
                continue

        self._learned += n_ok
        self._last_loss_sum = err_sum
        self._last_batches = n_ok
        return err_sum / max(1, n_ok)

    def predict(
        self,
        prompt: str,
        response: str,
        *,
        environment_summary: str = "",
    ) -> Optional[float]:
        """Return VW prediction in ``[0, 1]`` if trained, else ``None``.

        Must mirror :meth:`record` feature namespaces (``|p``, ``|q``, optional ``|e``)
        so train and inference see the same geometry.
        """
        if not self._vw or self._learned < 1:
            return None
        p = _sanitize_text(prompt, 600)
        r = _sanitize_text(response, 600)
        line = f"|p {p} |q {r}"
        if environment_summary:
            line += f" |e {_sanitize_text(environment_summary, VW_ENV_NAMESPACE_MAX_CHARS)}"
        try:
            pr = self._vw.predict(line)
            if isinstance(pr, list):
                pr = pr[0] if pr else 0.0
            return max(0.0, min(1.0, float(pr)))
        except Exception:
            return None

    def stats(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "learned_examples": self._learned,
            "last_batch_mae": (
                round(self._last_loss_sum / max(1, self._last_batches), 6)
                if self._last_batches
                else None
            ),
            "pending": len(self._pending),
        }

    def finish(self) -> None:
        if self._vw is not None:
            try:
                self._vw.finish()
            except Exception:
                pass
            self._vw = None


def _alignment_summary_only(full: Any) -> Dict[str, Any]:
    """Compact alignment diagnostics for JSONL when full log is off."""
    if not isinstance(full, dict):
        return {"_note": "non_dict_alignment"}
    keys = (
        "outcome_score",
        "blend_weight",
        "deliberative_used",
        "deliberative_unit",
        "chosen_id",
        "blended_reward",
    )
    slim = {k: full[k] for k in keys if k in full}
    if isinstance(full.get("pillars"), dict) and full["pillars"]:
        slim["pillars"] = {
            k: round(float(v), 4) if isinstance(v, (int, float)) else v
            for k, v in list(full["pillars"].items())[:12]
        }
    return slim


def _transition_info_payload(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten ``agent_environment`` for JSONL while keeping records bounded."""
    if not metadata:
        return {}
    out: Dict[str, Any] = dict(metadata)
    ae = out.get("agent_environment")
    if isinstance(ae, dict):
        fp = ae.get("fingerprint")
        dom = ae.get("domain")
        ver = ae.get("version")
        if ver is None:
            ver = ae.get("schema_version")
        out.setdefault("agent_environment_fp", fp)
        out.setdefault("agent_environment_domain", dom)
        out.setdefault("agent_environment_version", ver)
    if not env_truthy("HAROMA_RLLIB_LOG_FULL_AGENT_ENV", False):
        out.pop("agent_environment", None)
    if not env_truthy("HAROMA_RLLIB_LOG_FULL_ALIGNMENT_DIAG", False) and "alignment" in out:
        out["alignment"] = _alignment_summary_only(out.get("alignment"))
    es = out.get("environment_summary")
    if isinstance(es, str):
        cap = max(1, min(env_int("HAROMA_RLLIB_ENV_SUMMARY_LOG_CHARS", 2000), 500_000))
        if len(es) > cap:
            out["environment_summary"] = es[: cap - 1] + "…"
    return out


class RLlibTransitionLogger:
    """Append bandit-style JSON lines for Ray RLlib / offline RL pipelines."""

    def __init__(self) -> None:
        self._enabled = env_truthy("HAROMA_RLLIB_LOG_TRANSITIONS", False)
        self._path = os.environ.get("HAROMA_RLLIB_TRANSITIONS_PATH") or _DEFAULT_RLLIB_LOG
        self._lock = threading.Lock()
        self._count = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record(
        self,
        prompt: str,
        response: str,
        reward: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._enabled:
            return
        try:
            rw = max(0.0, min(1.0, float(reward)))
        except (TypeError, ValueError):
            rw = 0.5
        rec = {
            "type": "bandit_step",
            "obs": _sanitize_text(prompt, 4000),
            "action": _sanitize_text(response, 4000),
            "reward": rw,
            "done": True,
            "info": _transition_info_payload(metadata),
        }
        line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
        try:
            _dir = os.path.dirname(self._path)
            if _dir:
                os.makedirs(_dir, exist_ok=True)
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(line)
                self._count += 1
        except Exception as exc:
            print(f"[RLlibTransitionLogger] write failed: {exc}", flush=True)

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "path": self._path,
            "writes": self._count,
        }


