"""Offline bridge between Gymnasium rollouts and Haroma bandit JSONL.

Writes ``type: bandit_step`` lines compatible with :class:`RLlibTransitionLogger`
and :func:`mind.training.rllib_jsonl_offline.iter_bandit_steps`.

Optional: train a small scikit-learn linear scorer and emit a module usable with
``HAROMA_RLLIB_SCORE_FN`` (see docs/gymnasium-bridge.md).
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from mind.training.rllib_jsonl_offline import (
    default_transitions_path,
    iter_bandit_steps,
    load_bandit_steps,
    summarize_file,
)
from mind.training.vw_rl_bridge import _sanitize_text, _transition_info_payload


def bandit_record_dict(
    prompt: str,
    response: str,
    reward: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Single JSON object matching :meth:`RLlibTransitionLogger.record` output shape."""
    try:
        rw = max(0.0, min(1.0, float(reward)))
    except (TypeError, ValueError):
        rw = 0.5
    return {
        "type": "bandit_step",
        "obs": _sanitize_text(prompt, 4000),
        "action": _sanitize_text(response, 4000),
        "reward": rw,
        "done": True,
        "info": _transition_info_payload(metadata),
    }


def write_bandit_steps_to_jsonl(
    path: str,
    steps: Iterable[Mapping[str, Any]],
    *,
    append: bool = True,
) -> int:
    """Write bandit_step lines. Each *step* may contain:

    - ``prompt`` or ``obs`` (string)
    - ``response`` or ``action`` (string)
    - ``reward`` (float)
    - optional ``metadata`` or merge top-level keys into info via ``info`` key

    Returns number of lines written.
    """
    n = 0
    mode = "a" if append and os.path.isfile(path) else "w"
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for raw in steps:
            if not isinstance(raw, Mapping):
                continue
            prompt = str(raw.get("prompt") or raw.get("obs") or "")
            response = str(raw.get("response") or raw.get("action") or "")
            rw = raw.get("reward", 0.5)
            meta = raw.get("metadata")
            if meta is None and isinstance(raw.get("info"), dict):
                meta = raw["info"]
            elif meta is None:
                meta = {}
            # Pass reward through; bandit_record_dict matches RLlibTransitionLogger coercion
            rec = bandit_record_dict(prompt, response, rw, metadata=meta if meta else None)
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def read_bandit_steps(
    path: Optional[str] = None,
    *,
    limit: int = 100_000,
) -> List[Dict[str, Any]]:
    """Alias for :func:`mind.training.rllib_jsonl_offline.load_bandit_steps`."""
    return load_bandit_steps(path, limit=limit)


def summarize_bandit_jsonl(path: Optional[str] = None) -> Dict[str, Any]:
    """Alias for :func:`mind.training.rllib_jsonl_offline.summarize_file`."""
    return summarize_file(path)


def iter_gymnasium_rollout_records(
    prompts: List[str],
    responses: List[str],
    rewards: List[float],
    infos: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> Iterator[Dict[str, Any]]:
    """Zip parallel lists into dicts suitable for :func:`write_bandit_steps_to_jsonl`."""
    if not (len(prompts) == len(responses) == len(rewards)):
        raise ValueError("prompts, responses, rewards must have equal length")
    if infos is not None and len(infos) != len(prompts):
        raise ValueError("infos length must match prompts")
    for i in range(len(prompts)):
        d: Dict[str, Any] = {
            "prompt": prompts[i],
            "response": responses[i],
            "reward": rewards[i],
        }
        if infos is not None and infos[i]:
            d["metadata"] = infos[i]
        yield d


def _pair_obs_action(obs: str, action: str) -> str:
    return f"{obs or ''} ||| {action or ''}"


def train_linear_scorer_sklearn(
    jsonl_path: str,
    out_dir: str,
    *,
    limit: int = 50_000,
) -> Tuple[str, str]:
    """Fit Ridge on hashed ``obs ||| action`` text; save pickle + loader module.

    Requires ``pip install scikit-learn`` (see requirements-rl.txt).

    Returns ``(module_path, pkl_path)``. Set ``HAROMA_RLLIB_SCORE_FN`` to
    ``haroma_linear_scorer:score`` with ``PYTHONPATH`` including *out_dir*.

    Raises ``ImportError`` if sklearn is not installed.
    """
    try:
        import numpy as np
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
    except ImportError as e:
        raise ImportError(
            "train_linear_scorer_sklearn requires scikit-learn: pip install scikit-learn"
        ) from e

    steps = load_bandit_steps(jsonl_path, limit=limit)
    if len(steps) < 4:
        raise ValueError(f"need at least 4 bandit_step rows, got {len(steps)}")

    texts = [
        _pair_obs_action(str(s.get("obs") or ""), str(s.get("action") or "")) for s in steps
    ]
    y_list: List[float] = []
    for s in steps:
        try:
            y_list.append(float(s.get("reward", 0.0)))
        except (TypeError, ValueError):
            y_list.append(0.0)
    y_arr = np.asarray(y_list, dtype=np.float64)

    pipe = Pipeline(
        [
            ("vec", HashingVectorizer(n_features=8192, alternate_sign=False)),
            ("ridge", Ridge(alpha=1.0, random_state=0)),
        ]
    )
    pipe.fit(texts, y_arr)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "haroma_linear_scorer")
    pkl_path = base + ".pkl"
    mod_path = base + ".py"
    with open(pkl_path, "wb") as f:
        pickle.dump(pipe, f, protocol=4)

    mod_src = f'''"""Auto-generated linear scorer for HAROMA_RLLIB_SCORE_FN.

Usage:
  set HAROMA_RLLIB_SCORE_FN=haroma_linear_scorer:score
  set PYTHONPATH to the directory containing this file.
"""
from __future__ import annotations

import os
import pickle

_DIR = os.path.dirname(os.path.abspath(__file__))
_PKL = os.path.join(_DIR, {os.path.basename(pkl_path)!r})

_model = None

def _get_model():
    global _model
    if _model is None:
        with open(_PKL, "rb") as f:
            _model = pickle.load(f)
    return _model

def score(prompt: str, response: str) -> float:
    """Return reward in [0, 1] for Haroma composite scoring."""
    m = _get_model()
    text = (prompt or "") + " ||| " + (response or "")
    v = float(m.predict([text])[0])
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v
'''
    with open(mod_path, "w", encoding="utf-8") as f:
        f.write(mod_src)

    return mod_path, pkl_path


def write_constant_scorer_module(out_path: str, value: float = 0.5) -> str:
    """Write a minimal scorer module (constant in [0,1]) for testing or baselines."""
    try:
        v = max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        v = 0.5
    src = f'''"""Constant scorer for HAROMA_RLLIB_SCORE_FN."""

def score(prompt: str, response: str) -> float:
    return {v!r}
'''
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(src)
    return out_path


__all__ = [
    "bandit_record_dict",
    "default_transitions_path",
    "iter_bandit_steps",
    "iter_gymnasium_rollout_records",
    "read_bandit_steps",
    "summarize_bandit_jsonl",
    "train_linear_scorer_sklearn",
    "write_bandit_steps_to_jsonl",
    "write_constant_scorer_module",
]
