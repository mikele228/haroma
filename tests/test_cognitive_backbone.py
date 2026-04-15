"""Tests for CognitiveBackbone — classic and MLA-lite architectures."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests._import_guard import skip_unless_torch_imports

skip_unless_torch_imports()

import pytest

try:
    import torch
except (ImportError, OSError):
    pytest.skip("torch unavailable", allow_module_level=True)

from engine.CognitiveBackbone import CognitiveBackbone, build_snapshot, _SNAPSHOT_DIM, _Z_DIM

pytestmark = pytest.mark.torch

# ── helpers ──────────────────────────────────────────────────────────


def _dummy_snapshot() -> list:
    return [0.1 * i for i in range(_SNAPSHOT_DIM)]


def _dummy_content_embedding(dim: int = 256) -> list:
    return [0.01 * i for i in range(dim)]


_SNAPSHOT_KWARGS = dict(
    content_embedding=None,
    embed_dim=256,
    valence=0.3,
    arousal=0.5,
    intensity=0.4,
    curiosity_score=0.6,
    prediction_error=0.1,
    dominant_drive_level=0.5,
    wm_load=0.4,
    outcome_prev=0.7,
    has_external=1.0,
    cycle_count=42,
    n_goals=3,
    kg_entity_count=15,
    self_surprise=0.2,
    emotion_streak=2,
    drift_score=0.05,
    strategy="reflect",
    intent="utterance",
    drive_levels=[0.5, 0.3, 0.4, 0.2, 0.1],
    reasoning_depth=2,
    cf_depth=1,
    imagination_quality=0.6,
    metacog_prediction=0.5,
    steps_run_ratio=0.8,
    plan_active=False,
    env_tick=10,
)


# ── build_snapshot ───────────────────────────────────────────────────


def test_build_snapshot_length():
    snap = build_snapshot(**_SNAPSHOT_KWARGS)
    assert len(snap) == _SNAPSHOT_DIM


# ── Classic backbone ─────────────────────────────────────────────────


class TestClassicBackbone:
    def test_encode_state_shape(self):
        bb = CognitiveBackbone(use_mla=False)
        bb._train_steps = 200
        z = bb.encode_state(_dummy_snapshot())
        assert z is not None
        assert len(z) == _Z_DIM

    def test_encode_state_with_content(self):
        bb = CognitiveBackbone(use_mla=False)
        bb._train_steps = 200
        z = bb.encode_state(_dummy_snapshot(), _dummy_content_embedding())
        assert z is not None
        assert len(z) == _Z_DIM

    def test_train_step_no_nan(self):
        bb = CognitiveBackbone(use_mla=False)
        for i in range(12):
            bb.record_outcome(
                _dummy_snapshot(),
                outcome_score=0.5 + 0.01 * i,
                content_embedding=_dummy_content_embedding(),
            )
        loss = bb.train_step()
        assert loss is not None
        assert not (loss != loss), "loss is NaN"

    def test_to_from_dict_roundtrip(self):
        bb = CognitiveBackbone(use_mla=False)
        bb._train_steps = 5
        data = bb.to_dict()
        assert data["architecture"] == "transformer"

        bb2 = CognitiveBackbone(use_mla=False)
        bb2.from_dict(data)
        assert bb2._train_steps == 5

    def test_from_dict_arch_mismatch_skips(self):
        bb = CognitiveBackbone(use_mla=False)
        bb._train_steps = 10
        data = bb.to_dict()

        bb_mla = CognitiveBackbone(use_mla=True)
        bb_mla.from_dict(data)
        assert bb_mla._train_steps == 0


# ── MLA-lite backbone ────────────────────────────────────────────────


class TestMLALiteBackbone:
    def test_encode_state_shape(self):
        bb = CognitiveBackbone(use_mla=True)
        bb._train_steps = 200
        z = bb.encode_state(_dummy_snapshot())
        assert z is not None
        assert len(z) == _Z_DIM

    def test_encode_state_with_content(self):
        bb = CognitiveBackbone(use_mla=True)
        bb._train_steps = 200
        z = bb.encode_state(_dummy_snapshot(), _dummy_content_embedding())
        assert z is not None
        assert len(z) == _Z_DIM

    def test_train_step_no_nan(self):
        bb = CognitiveBackbone(use_mla=True)
        for i in range(12):
            bb.record_outcome(
                _dummy_snapshot(),
                outcome_score=0.5 + 0.01 * i,
                content_embedding=_dummy_content_embedding(),
            )
        loss = bb.train_step()
        assert loss is not None
        assert not (loss != loss), "loss is NaN"

    def test_to_from_dict_roundtrip(self):
        bb = CognitiveBackbone(use_mla=True)
        bb._train_steps = 7
        data = bb.to_dict()
        assert data["architecture"] == "mla_lite"

        bb2 = CognitiveBackbone(use_mla=True)
        bb2.from_dict(data)
        assert bb2._train_steps == 7

    def test_stats_architecture(self):
        bb = CognitiveBackbone(use_mla=True)
        assert bb.stats()["architecture"] == "mla_lite"

    def test_content_dim_padding(self):
        """Content embedding shorter than 256-d should be padded."""
        bb = CognitiveBackbone(use_mla=True)
        bb._train_steps = 200
        z = bb.encode_state(_dummy_snapshot(), _dummy_content_embedding(dim=128))
        assert z is not None
        assert len(z) == _Z_DIM

    def test_content_dim_truncation(self):
        """Content embedding longer than 256-d should be truncated."""
        bb = CognitiveBackbone(use_mla=True)
        bb._train_steps = 200
        z = bb.encode_state(_dummy_snapshot(), _dummy_content_embedding(dim=384))
        assert z is not None
        assert len(z) == _Z_DIM
