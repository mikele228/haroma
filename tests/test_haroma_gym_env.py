"""HaromaBanditChatEnv with mocked HTTP (no live server)."""

from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


@pytest.fixture
def gymnasium_available():
    pytest.importorskip("gymnasium")


def test_haroma_bandit_env_reset_step(gymnasium_available, monkeypatch):
    from mind.training.haroma_gym_env import HaromaBanditChatEnv

    calls: list = []

    def fake_post(base_url, message, *, depth="normal", async_=False, timeout=600.0, extra_fields=None):
        calls.append({"message": message, "async": async_})
        return {"response": "yes that is ok", "cycle": 1}, 200

    monkeypatch.setattr("mind.training.haroma_gym_env.post_chat", fake_post)

    env = HaromaBanditChatEnv(
        "http://127.0.0.1:8193",
        candidate_messages=["hello", "status"],
        tasks=["TaskA"],
        chat_timeout_sec=30.0,
    )
    obs, info = env.reset(seed=0)
    assert obs.shape == (1,)
    assert float(obs.sum()) == 1.0

    obs2, reward, terminated, truncated, step_info = env.step(0)
    assert len(calls) == 1
    assert "TaskA" in calls[0]["message"] or calls[0]["message"].strip()
    assert 0.0 <= reward <= 1.0
    assert terminated is True
    assert truncated is False
    assert "response" not in step_info or step_info.get("http_status") == 200
    assert obs2.shape == obs.shape


def test_haroma_bandit_env_depth_legacy_ignored(gymnasium_available):
    from mind.training.haroma_gym_env import HaromaBanditChatEnv

    env = HaromaBanditChatEnv(
        "http://127.0.0.1:8193",
        candidate_messages=["x"],
        depth="fast",
    )
    assert env._depth == "normal"


def test_haroma_bandit_env_http_error_zero_reward(gymnasium_available, monkeypatch):
    from mind.training.haroma_gym_env import HaromaBanditChatEnv

    def fake_post(*a, **k):
        return {"error": "server not ready"}, 503

    monkeypatch.setattr("mind.training.haroma_gym_env.post_chat", fake_post)

    env = HaromaBanditChatEnv(
        "http://127.0.0.1:8193",
        candidate_messages=["x"],
    )
    env.reset(seed=0)
    _obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 0.0
    assert terminated is True
    assert info.get("http_status") == 503


def test_post_chat_helpers_use_urllib(monkeypatch):
    """Smoke: post_chat delegates to same path as health (mock urllib)."""
    from bridge import haroma_client as hc

    class _Resp:
        status = 200

        def read(self):
            return b'{"response":"hi","cycle":3}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    def fake_urlopen(req, timeout=15, context=None):
        assert b"/chat" in req.full_url.encode() or "/chat" in getattr(req, "full_url", "")
        return _Resp()

    monkeypatch.setattr(hc.urllib.request, "urlopen", fake_urlopen)
    body, code = hc.post_chat("http://localhost:8193", "ping", async_=False, timeout=5.0)
    assert code == 200
    assert body.get("response") == "hi"
