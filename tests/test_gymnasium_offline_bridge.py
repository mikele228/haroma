"""Offline bandit JSONL bridge and optional sklearn scorer."""

from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def test_write_and_read_bandit_jsonl(tmp_path):
    from mind.training.gymnasium_offline_bridge import (
        read_bandit_steps,
        summarize_bandit_jsonl,
        write_bandit_steps_to_jsonl,
    )

    p = str(tmp_path / "t.jsonl")
    n = write_bandit_steps_to_jsonl(
        p,
        [
            {"prompt": "hello", "response": "hi", "reward": 0.7},
            {"obs": "q", "action": "a", "reward": 0.2, "metadata": {"x": 1}},
        ],
        append=False,
    )
    assert n == 2
    rows = read_bandit_steps(p, limit=10)
    assert len(rows) == 2
    assert rows[0]["type"] == "bandit_step"
    assert rows[0]["obs"] == "hello"
    assert rows[0]["action"] == "hi"
    assert abs(float(rows[0]["reward"]) - 0.7) < 1e-6
    assert rows[1]["obs"] == "q"
    assert rows[1]["action"] == "a"
    summ = summarize_bandit_jsonl(p)
    assert summ["count"] == 2
    assert summ["mean_reward"] is not None
    assert abs(summ["mean_reward"] - 0.45) < 1e-6


def test_write_bandit_jsonl_invalid_reward_coerces_like_logger(tmp_path):
    from mind.training.gymnasium_offline_bridge import read_bandit_steps, write_bandit_steps_to_jsonl

    p = str(tmp_path / "bad_reward.jsonl")
    n = write_bandit_steps_to_jsonl(
        p,
        [{"prompt": "a", "response": "b", "reward": "not_numeric"}],
        append=False,
    )
    assert n == 1
    rows = read_bandit_steps(p, limit=5)
    assert len(rows) == 1
    assert abs(float(rows[0]["reward"]) - 0.5) < 1e-6


def test_bandit_record_dict_matches_logger_shape():
    from mind.training.gymnasium_offline_bridge import bandit_record_dict

    d = bandit_record_dict("p", "r", 0.33, metadata={"alignment": {"outcome_score": 0.5}})
    assert d["type"] == "bandit_step"
    assert "obs" in d and "action" in d
    assert d["done"] is True
    # RLlibTransitionLogger uses same keys
    assert set(d.keys()) >= {"type", "obs", "action", "reward", "done", "info"}


def test_iter_gymnasium_rollout_records():
    from mind.training.gymnasium_offline_bridge import iter_gymnasium_rollout_records

    recs = list(
        iter_gymnasium_rollout_records(
            ["a", "b"],
            ["x", "y"],
            [1.0, 0.0],
            infos=[{"k": 1}, None],
        )
    )
    assert len(recs) == 2
    assert recs[0]["metadata"] == {"k": 1}


def test_constant_scorer_module_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("HAROMA_RLLIB_SCORE_WEIGHT", "0.5")
    from mind.training.gymnasium_offline_bridge import write_constant_scorer_module
    from mind.training.vw_rl_bridge import composite_trained_scores, load_rllib_score_callable

    mod = tmp_path / "cscorer.py"
    write_constant_scorer_module(str(mod), value=0.8)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("HAROMA_RLLIB_SCORE_FN", "cscorer:score")

    fn = load_rllib_score_callable()
    assert fn is not None
    assert fn("a", "b") == 0.8

    be = __import__("types").SimpleNamespace(
        reward_model=__import__("types").SimpleNamespace(score=lambda p, r: 0.0),
        _vw_trainer=None,
        _rllib_score_fn=None,
    )
    assert abs(composite_trained_scores(be, "p", "r") - 0.4) < 1e-6


@pytest.mark.skipif(
    os.environ.get("HAROMA_SKIP_SKLEARN", "").strip().lower() in ("1", "true", "yes"),
    reason="HAROMA_SKIP_SKLEARN set",
)
def test_train_linear_scorer_sklearn_roundtrip(tmp_path):
    sklearn = pytest.importorskip("sklearn")
    del sklearn  # noqa: F841

    from mind.training.gymnasium_offline_bridge import train_linear_scorer_sklearn, write_bandit_steps_to_jsonl

    p = str(tmp_path / "data.jsonl")
    write_bandit_steps_to_jsonl(
        p,
        [
            {"prompt": "hello", "response": "there", "reward": 0.9},
            {"prompt": "hello", "response": "x", "reward": 0.1},
            {"prompt": "bye", "response": "z", "reward": 0.8},
            {"prompt": "bye", "response": "w", "reward": 0.2},
        ],
        append=False,
    )
    out = str(tmp_path / "out")
    mod_path, pkl_path = train_linear_scorer_sklearn(p, out)
    assert os.path.isfile(mod_path) and os.path.isfile(pkl_path)

    import importlib.util

    spec = importlib.util.spec_from_file_location("haroma_linear_scorer", mod_path)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    s = m.score("hello", "there")
    assert 0.0 <= s <= 1.0
