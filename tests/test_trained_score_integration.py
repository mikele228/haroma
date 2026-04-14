"""Composite scoring: torch + optional VW + optional RLlib hook."""

from __future__ import annotations

from types import SimpleNamespace


def test_composite_torch_only_when_weights_zero(monkeypatch):
    monkeypatch.delenv("HAROMA_VW_SCORE_WEIGHT", raising=False)
    monkeypatch.delenv("HAROMA_RLLIB_SCORE_WEIGHT", raising=False)
    from mind.training.vw_rl_bridge import composite_trained_scores

    be = SimpleNamespace(
        reward_model=SimpleNamespace(score=lambda p, r: 0.42),
        _vw_trainer=None,
        _rllib_score_fn=None,
    )
    assert composite_trained_scores(be, "a", "b") == 0.42


def test_composite_blends_vowpal_when_weight_and_predict(monkeypatch):
    monkeypatch.setenv("HAROMA_VW_SCORE_WEIGHT", "0.5")

    class _VW:
        def predict(self, p, r, **kwargs):
            return 1.0

    from mind.training.vw_rl_bridge import composite_trained_scores

    be = SimpleNamespace(
        reward_model=SimpleNamespace(score=lambda p, r: 0.0),
        _vw_trainer=_VW(),
        _rllib_score_fn=None,
    )
    # (1-0.5)*0 + 0.5*1 = 0.5
    assert abs(composite_trained_scores(be, "p", "r") - 0.5) < 1e-6


def test_composite_rllib_hook(monkeypatch, tmp_path):
    monkeypatch.setenv("HAROMA_RLLIB_SCORE_WEIGHT", "0.5")
    hook = tmp_path / "rllib_hook.py"
    hook.write_text(
        "def my_score(p, r):\n    return 0.8\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("HAROMA_RLLIB_SCORE_FN", "rllib_hook:my_score")

    from mind.training.vw_rl_bridge import composite_trained_scores

    be = SimpleNamespace(
        reward_model=SimpleNamespace(score=lambda p, r: 0.0),
        _vw_trainer=None,
        _rllib_score_fn=None,
    )
    # (1-0.5)*0 + 0.5*0.8 = 0.4
    assert abs(composite_trained_scores(be, "p", "r") - 0.4) < 1e-6


def test_rllib_transition_info_flattens_agent_environment(monkeypatch):
    monkeypatch.delenv("HAROMA_RLLIB_LOG_FULL_AGENT_ENV", raising=False)
    from mind.training.vw_rl_bridge import _transition_info_payload

    p = _transition_info_payload(
        {
            "agent_environment": {
                "fingerprint": "abc123",
                "domain": "lab",
                "version": 1,
            },
            "alignment_training": True,
        }
    )
    assert p["agent_environment_fp"] == "abc123"
    assert p["agent_environment_domain"] == "lab"
    assert "agent_environment" not in p


def test_rllib_env_summary_respects_small_char_cap(monkeypatch):
    monkeypatch.setenv("HAROMA_RLLIB_ENV_SUMMARY_LOG_CHARS", "12")
    from mind.training.vw_rl_bridge import _transition_info_payload

    long = "x" * 50
    p = _transition_info_payload({"environment_summary": long})
    assert len(p["environment_summary"]) <= 12
    assert p["environment_summary"].endswith("…")


def test_rllib_transition_info_keeps_full_agent_environment_when_enabled(monkeypatch):
    monkeypatch.setenv("HAROMA_RLLIB_LOG_FULL_AGENT_ENV", "1")
    from mind.training.vw_rl_bridge import _transition_info_payload

    ae = {"fingerprint": "x", "domain": "d"}
    p = _transition_info_payload({"agent_environment": ae})
    assert p.get("agent_environment") == ae


def test_rllib_alignment_slimmed_by_default(monkeypatch):
    monkeypatch.delenv("HAROMA_RLLIB_LOG_FULL_ALIGNMENT_DIAG", raising=False)
    from mind.training.vw_rl_bridge import _transition_info_payload

    big = {"outcome_score": 0.5, "blend_weight": 0.3, "extra_noise": "x" * 500}
    p = _transition_info_payload({"alignment": big})
    assert "extra_noise" not in p.get("alignment", {})


def test_rllib_alignment_full_when_flag(monkeypatch):
    monkeypatch.setenv("HAROMA_RLLIB_LOG_FULL_ALIGNMENT_DIAG", "1")
    from mind.training.vw_rl_bridge import _transition_info_payload

    big = {"outcome_score": 0.5, "extra_noise": "y" * 100}
    p = _transition_info_payload({"alignment": big})
    assert p["alignment"].get("extra_noise") == "y" * 100


def test_vw_cached_env_expires_after_ttl(monkeypatch):
    monkeypatch.setenv("HAROMA_VW_SCORE_WEIGHT", "0.5")
    monkeypatch.setenv("HAROMA_VW_ENV_CONTEXT_TTL_SEC", "100")
    import engine.LLMBackend as lb_mod

    captured = []

    class _VW:
        def predict(self, p, r, **kwargs):
            captured.append(kwargs.get("environment_summary", ""))
            return 0.25

    from engine.LLMBackend import LLMBackend
    from mind.training.vw_rl_bridge import composite_trained_scores

    b = LLMBackend(use_programmed=True)
    b._vw_trainer = _VW()
    monkeypatch.setattr(lb_mod.time, "time", lambda: 1000.0)
    b._last_env_summary = "room a"
    b._last_env_summary_ts = 1000.0

    composite_trained_scores(b, "p", "r")
    assert captured[-1] == "room a"

    monkeypatch.setattr(lb_mod.time, "time", lambda: 1200.0)
    composite_trained_scores(b, "p", "r")
    assert captured[-1] == ""


def test_llm_backend_score_with_trained_heads_programmed(monkeypatch):
    monkeypatch.delenv("HAROMA_VW_SCORE_WEIGHT", raising=False)
    monkeypatch.delenv("HAROMA_RLLIB_SCORE_WEIGHT", raising=False)
    from engine.LLMBackend import LLMBackend

    b = LLMBackend(use_programmed=True)
    s = b.score_with_trained_heads("x", "y")
    assert 0.0 <= s <= 1.0
