"""integrations.sim — universal simulation protocol and registry."""

from __future__ import annotations

import os
import sys


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def test_null_backend_roundtrip():
    from integrations.sim import create_backend

    b = create_backend("null")
    assert b.backend_id() == "null"
    r = b.reset(seed=1)
    assert r.get("ok") is True
    s = b.step({"action_type": "noop"})
    assert s.get("step_index") == 1
    b.close()


def test_load_backend_from_env_default(monkeypatch):
    monkeypatch.delenv("HAROMA_SIM_BACKEND", raising=False)
    from integrations.sim import load_backend_from_env

    b = load_backend_from_env()
    assert b.backend_id() == "null"


def test_http_json_backend_mock(monkeypatch):
    from integrations.sim.backends.http_json_backend import HttpJsonSimulationBackend
    from unittest.mock import patch

    calls = []

    def fake_post(url, body, *, timeout=30.0):
        calls.append(("post", url, body))
        return {"ok": True, "echo": body}, 200

    monkeypatch.setenv("HAROMA_SIM_HTTP_BASE_URL", "http://test.local:1")
    b = HttpJsonSimulationBackend.from_env()
    with patch(
        "integrations.sim.backends.http_json_backend._post_json",
        side_effect=fake_post,
    ):
        r = b.reset(seed=2)
        assert r.get("ok") is True
        b.step({"move": "north"})
        assert any("action" in str(c) for c in calls)


def test_register_custom_backend():
    from integrations.sim import create_backend, register_backend
    from integrations.sim.backends.null_backend import NullSimulationBackend

    register_backend("custom_null", lambda **_: NullSimulationBackend())
    b = create_backend("custom_null")
    assert b.backend_id() == "null"


def test_importable_backend_from_file(monkeypatch, tmp_path):
    import importlib.util

    p = tmp_path / "sim_x.py"
    p.write_text(
        """
class X:
    def __init__(self, n=0):
        self.n = int(n)
    def backend_id(self):
        return "x"
    def capabilities(self):
        return {}
    def reset(self, seed=None, **kwargs):
        return {"ok": True}
    def step(self, action):
        return {"ok": True}
    def observe(self, **kwargs):
        return {}
    def close(self):
        pass
""",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("sim_x_mod", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["sim_x_mod"] = mod
    monkeypatch.setenv("HAROMA_SIM_BACKEND", "sim_x_mod:X")
    monkeypatch.setenv("HAROMA_SIM_BACKEND_KWARGS", '{"n": 3}')
    from integrations.sim import load_backend_from_env

    b = load_backend_from_env()
    assert b.backend_id() == "x"
    assert b.n == 3
    del sys.modules["sim_x_mod"]


def test_merge_extensions():
    from integrations.sim.universal import merge_simulation_into_extensions, simulation_summary_for_prompt

    ext = merge_simulation_into_extensions(
        {"x": 1},
        backend_id="http_json",
        bundle={"description": "hello", "observation": {"rgb": "ref"}},
    )
    assert ext["simulation"]["backend_id"] == "http_json"
    assert "hello" in simulation_summary_for_prompt({"description": "hello world"})
