"""Unit tests for mind.cycle_flow.organic_confidence and threshold constant."""

from __future__ import annotations

import pytest

from mind.cycle_flow import (
    organic_confidence,
    ORGANIC_PACKED_LLM_SKIP_THRESHOLD,
)


class TestOrganicConfidence:
    """Core behaviour of the organic-confidence scorer."""

    def test_empty_inputs_returns_zero(self):
        assert organic_confidence([], None, None) == 0.0

    def test_empty_lists_and_empty_dicts(self):
        assert (
            organic_confidence(
                [], {"inferences": []}, {"relevance": 0.0, "norm_compatibility": 0.99}
            )
            == 0.0
        )

    def test_memory_confidence_ignored(self):
        """Recall confidence must not skip the LLM (nodes default to 1.0)."""
        mems = [{"confidence": 0.99}, {"confidence": 1.0}]
        assert organic_confidence(mems, None, None) == 0.0
        assert organic_confidence(mems, {"inferences": []}, {}) == 0.0

    def test_reasoning_confidence_only(self):
        rr = {"inferences": [{"confidence": 0.85}, {"confidence": 0.6}]}
        result = organic_confidence([], rr)
        assert result == pytest.approx(0.85)
        assert result < ORGANIC_PACKED_LLM_SKIP_THRESHOLD

    def test_high_reasoning_above_threshold(self):
        rr = {"inferences": [{"confidence": 0.92}]}
        result = organic_confidence([], rr)
        assert result >= ORGANIC_PACKED_LLM_SKIP_THRESHOLD

    def test_appraisal_relevance_wins(self):
        ap = {"relevance": 0.91, "norm_compatibility": 0.4}
        result = organic_confidence([], None, ap)
        assert result == pytest.approx(0.91)
        assert result >= ORGANIC_PACKED_LLM_SKIP_THRESHOLD

    def test_appraisal_norm_compatibility_ignored_for_skip(self):
        ap = {"relevance": 0.1, "norm_compatibility": 0.99}
        result = organic_confidence([], None, ap)
        assert result == pytest.approx(0.1)

    def test_max_across_reasoning_and_relevance(self):
        mems = [{"confidence": 1.0}]
        rr = {"inferences": [{"confidence": 0.7}]}
        ap = {"relevance": 0.8, "norm_compatibility": 0.99}
        result = organic_confidence(mems, rr, ap)
        assert result == pytest.approx(0.8)

    def test_clamping_above_one(self):
        rr = {"inferences": [{"confidence": 1.5}]}
        assert organic_confidence([], rr) == pytest.approx(1.0)

    def test_clamping_below_zero(self):
        rr = {"inferences": [{"confidence": -0.3}]}
        assert organic_confidence([], rr) == 0.0

    def test_non_numeric_inference_ignored(self):
        rr = {"inferences": [{"confidence": "bad"}, {"confidence": 0.4}]}
        assert organic_confidence([], rr) == pytest.approx(0.4)

    def test_missing_confidence_key_on_inference(self):
        rr = {"inferences": [{"subject": "a", "predicate": "b", "object": "c"}]}
        assert organic_confidence([], rr) == 0.0

    def test_none_recalled_memories(self):
        assert organic_confidence(None) == 0.0


class TestThresholdConstant:
    def test_default_value(self):
        assert ORGANIC_PACKED_LLM_SKIP_THRESHOLD == pytest.approx(0.9)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("HAROMA_ORGANIC_LLM_SKIP", "0.75")
        from importlib import reload
        import mind.cycle_flow as cf

        reload(cf)
        assert cf.ORGANIC_PACKED_LLM_SKIP_THRESHOLD == pytest.approx(0.75)
        monkeypatch.delenv("HAROMA_ORGANIC_LLM_SKIP", raising=False)
        reload(cf)
