"""Tests for integrative brain-like state."""

from mind.humanoid_brain_state import compute_brain_like_state, maybe_log_brain_state


def test_compute_returns_expected_keys():
    s = compute_brain_like_state(
        affect={"intensity": 0.5, "arousal": 0.2, "valence": 0.1},
        curiosity={"novelty_score": 0.3, "uncertainty_score": 0.2},
        drives={"sleep": 0.4},
        dominant_drive="sleep",
        embodied_modulation={"novelty_bias": 0.0, "curiosity_damping": 1.0},
        self_surprise={"overall_surprise": 0.1, "accuracy": 0.8},
        appraisal={"relevance": 0.2, "overrides": False},
        drift_score=0.05,
        prediction_error=0.15,
    )
    for k in (
        "arousal_index",
        "exploration_index",
        "stability_index",
        "plasticity_index",
        "consolidation_pressure",
        "mood_tilt",
        "narrative",
    ):
        assert k in s
    assert s["schema_version"] == 1


def test_maybe_log_brain_state_no_crash(monkeypatch):
    maybe_log_brain_state("x", {"narrative": "ok"})
