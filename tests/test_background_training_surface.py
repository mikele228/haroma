"""core.training_surface + RuntimeSignals training-effect flag."""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

import numpy as np

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from agents.runtime_signals import RuntimeSignals
from core.self_model_train_batch import SelfModelTrainBatch
from core.training_surface import build_background_train_map, _self_model_background_train_step


class _Anything:
    """Stand-in ``SharedResources`` so train lambdas resolve attributes without boot."""

    def __getattr__(self, name: str):
        return None


def test_build_background_train_map_module_count():
    m = build_background_train_map(_Anything())
    assert len(m) == 17
    assert [t[0] for t in m][:3] == ["encoder", "backbone", "attention"]
    assert [t[0] for t in m][-1] == "llm_reward"


def test_self_model_background_train_step_uses_batch():
    class _SM:
        available = True

        def train_step(self, emb, prev, actual):
            return 0.25

    s = SimpleNamespace()
    s.self_model = _SM()
    s._self_model_last_train_ctx = SelfModelTrainBatch(
        embedding=np.array([0.1, 0.2], dtype=np.float32),
        prev_state={"valence": 0.0},
        actual_state={"valence": 0.1},
    )
    assert _self_model_background_train_step(s) == 0.25


def test_runtime_signals_last_training_had_effect():
    class _Sh:
        def __init__(self):
            self._http_chat_lock = threading.Lock()
            self._http_chat_inflight = 0

    sh = _Sh()
    sig = RuntimeSignals(sh)
    sig.record_background_training_completed(had_effect=False)
    snap0 = sig.snapshot()
    assert snap0["last_background_training_had_effect"] is False
    sig.record_background_training_completed(had_effect=True)
    snap1 = sig.snapshot()
    assert snap1["last_background_training_had_effect"] is True
