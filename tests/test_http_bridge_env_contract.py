"""HTTP contract tests for embodiment integrator routes (mocked ``shared``; no full boot).

Covers ``POST /agent/environment`` and ``POST /robot/bridge/feedback`` shapes expected
by bridge hosts — see ``docs/architecture-audit.md`` (integration surface).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests._import_guard import prepare_test_imports

prepare_test_imports(__file__)


@pytest.fixture
def bridge_client():
    import mind.elarion_server_v2 as srv

    calls: dict = {"env": [], "feedback": []}

    class Shared:
        def set_agent_environment(self, raw):
            calls["env"].append(raw)
            return {"ok": True, "stored": True}

        def agent_environment_status(self):
            return {"ok": True, "extensions": {"robot_bridge": {"test": True}}}

        def merge_robot_bridge_feedback(self, raw):
            calls["feedback"].append(raw)
            return {"ok": True, "merged": 1}

    shared = Shared()
    boot = SimpleNamespace(
        shared=shared,
        input_agent=SimpleNamespace(stats=lambda: {}),
        persona_agents=[],
        bus=SimpleNamespace(stats=lambda: {}),
    )
    from mind.server_state import get_haroma_server_state

    st = get_haroma_server_state(srv.app)
    old_boot = st.boot_agent
    old_poller = st.sensor_poller
    old_reg = st.chat_async_registry
    st.boot_agent = boot
    st.sensor_poller = None
    st.chat_async_registry = None
    try:
        yield srv.app.test_client(), calls, shared
    finally:
        st.boot_agent = old_boot
        st.sensor_poller = old_poller
        st.chat_async_registry = old_reg


def test_post_agent_environment_ok(bridge_client):
    client, calls, _shared = bridge_client
    body = {"scene": "lab", "tick": 1}
    r = client.post("/agent/environment", json=body)
    assert r.status_code == 200
    j = r.get_json() or {}
    assert j.get("status") == "ok"
    assert j.get("result", {}).get("ok") is True
    assert "agent_environment" in j
    assert calls["env"] == [body]


def test_post_agent_environment_requires_object(bridge_client):
    client, _, _ = bridge_client
    r = client.post("/agent/environment", json="not-an-object", content_type="application/json")
    assert r.status_code == 400
    assert (r.get_json() or {}).get("error") == "expected JSON object"


def test_post_robot_bridge_feedback_ok(bridge_client):
    client, calls, _ = bridge_client
    payload = {
        "bridge_schema_version": 1,
        "correlation_id": "corr-test",
        "results": [],
    }
    r = client.post("/robot/bridge/feedback", json=payload)
    assert r.status_code == 200
    j = r.get_json() or {}
    assert j.get("status") == "ok"
    assert j.get("result", {}).get("ok") is True
    assert "agent_environment" in j
    assert calls["feedback"] == [payload]


def test_post_robot_bridge_feedback_requires_object(bridge_client):
    client, _, _ = bridge_client
    r = client.post(
        "/robot/bridge/feedback",
        json=None,
        content_type="application/json",
    )
    # Flask silent JSON parse -> None body
    assert r.status_code == 400
    assert (r.get_json() or {}).get("error") == "expected JSON object"
