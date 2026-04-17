"""Flask routes for async /chat (mocked boot; no full stack boot)."""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

import pytest

from tests._import_guard import prepare_test_imports

prepare_test_imports(__file__)


@pytest.fixture
def srv_mock():
    import mind.elarion_server_v2 as srv
    from mind.chat_async_registry import ChatAsyncRegistry

    class Shared:
        class _Sig:
            def snapshot(self):
                return {
                    "http_chat_inflight": 0,
                    "http_chat_depth_stack": [],
                    "last_background_training_at": 0.0,
                    "last_background_training_had_effect": False,
                }

        def __init__(self):
            self.begins = 0
            self.ends = 0
            self.cycle_count = 42
            self.memory = SimpleNamespace(count_nodes=lambda: 0)
            self.organ_registry = SimpleNamespace(summary=lambda: {})
            self.symbolic_queue = SimpleNamespace(stats=lambda: {})
            self.fingerprint_engine = SimpleNamespace(stats=lambda: {})
            self.reconciliation = SimpleNamespace(stats=lambda: {})
            self.training_scheduler = None
            self.signals = Shared._Sig()

        @property
        def http_chat_inflight(self):
            return 0

        def http_chat_begin(self, depth=None):
            self.begins += 1

        def http_chat_end(self):
            self.ends += 1

        def set_agent_environment(self, _raw):
            return {"ok": True}

        def agent_environment_status(self):
            return {}

    shared = Shared()

    class MockInput:
        def __init__(self):
            self._slot_factory = None

        def set_slot_factory(self, fn):
            self._slot_factory = fn

        def push_text(self, *a, **k):
            assert self._slot_factory is not None
            return self._slot_factory()

        def log_response(self, _r):
            pass

    inp = MockInput()
    boot = SimpleNamespace(
        shared=shared,
        input_agent=inp,
        stats=lambda: {},
        persona_agents=[],
        bus=SimpleNamespace(stats=lambda: {}),
    )
    from mind.server_state import get_haroma_server_state

    st = get_haroma_server_state(srv.app)
    old_boot = st.boot_agent
    old_reg = st.chat_async_registry
    old_poller = st.sensor_poller
    st.boot_agent = boot
    st.sensor_poller = SimpleNamespace(stats=lambda: {})
    st.chat_async_registry = ChatAsyncRegistry(shared, ttl_sec=60.0)
    try:
        yield srv, shared, inp
    finally:
        st.boot_agent = old_boot
        st.chat_async_registry = old_reg
        st.sensor_poller = old_poller


def test_async_post_202_and_immediate_result(srv_mock):
    srv, shared, inp = srv_mock

    def make_ready_slot():
        ev = threading.Event()
        slot = {
            "event": ev,
            "result": {"response": "mock reply", "cycle": shared.cycle_count},
        }
        ev.set()
        return slot

    inp.set_slot_factory(make_ready_slot)
    client = srv.app.test_client()
    resp = client.post("/chat", json={"message": "hi", "async": True, "depth": "normal"})
    assert resp.status_code == 202
    data = resp.get_json() or {}
    assert data.get("status") == "pending"
    rid = data.get("request_id")
    assert rid
    assert shared.begins == 1
    assert shared.ends == 0

    r2 = client.get(f"/chat/result?id={rid}")
    assert r2.status_code == 200
    body = r2.get_json() or {}
    assert body.get("response") == "mock reply"
    assert body.get("cycle") == 42
    assert shared.ends == 1


def test_async_poll_pending_then_complete(srv_mock):
    srv, shared, inp = srv_mock
    ev = threading.Event()
    slot_holder: dict = {}

    def make_slow_slot():
        slot = {"event": ev, "result": None}
        slot_holder["s"] = slot
        return slot

    inp.set_slot_factory(make_slow_slot)
    client = srv.app.test_client()
    resp = client.post("/chat", json={"message": "x", "async": True, "depth": "normal"})
    assert resp.status_code == 202
    rid = (resp.get_json() or {}).get("request_id")
    assert rid

    p1 = client.get(f"/chat/result?id={rid}")
    assert p1.status_code == 200
    assert (p1.get_json() or {}).get("status") == "pending"

    slot_holder["s"]["result"] = {"response": "later", "cycle": 0}
    ev.set()

    p2 = client.get(f"/chat/result?id={rid}")
    assert p2.status_code == 200
    assert (p2.get_json() or {}).get("response") == "later"
    assert shared.ends == 1


def test_status_includes_chat_async_fields(srv_mock):
    srv, shared, inp = srv_mock

    def make_ready_slot():
        ev = threading.Event()
        slot = {"event": ev, "result": {"response": "x", "cycle": 0}}
        ev.set()
        return slot

    inp.set_slot_factory(make_ready_slot)
    client = srv.app.test_client()
    st0 = client.get("/status")
    assert st0.status_code == 200
    j0 = st0.get_json() or {}
    assert j0.get("chat_async_pending") == 0
    assert j0.get("chat_async_ttl_sec") == 60.0
    assert j0.get("http_chat_inflight") == 0
    assert j0.get("input_pipeline_busy") is False
    assert j0.get("bg_training_defer_enabled") is True
    assert j0.get("bg_training_deferred") is False

    pr = client.post("/chat", json={"message": "q", "async": True, "depth": "normal"})
    assert pr.status_code == 202
    rid = (pr.get_json() or {}).get("request_id")
    st1 = client.get("/status")
    j1 = st1.get_json() or {}
    assert j1.get("chat_async_pending") == 1
    client.get(f"/chat/result?id={rid}")
    st2 = client.get("/status")
    assert (st2.get_json() or {}).get("chat_async_pending") == 0


def test_default_async_from_env(srv_mock, monkeypatch):
    monkeypatch.setenv("HAROMA_CHAT_DEFAULT_ASYNC", "1")
    srv, shared, inp = srv_mock

    def make_ready_slot():
        ev = threading.Event()
        slot = {
            "event": ev,
            "result": {"response": "env async", "cycle": 0},
        }
        ev.set()
        return slot

    inp.set_slot_factory(make_ready_slot)
    client = srv.app.test_client()
    resp = client.post("/chat", json={"message": "hi", "depth": "normal"})
    assert resp.status_code == 202
    assert (resp.get_json() or {}).get("request_id")


def test_explicit_async_false_sync_when_env_default_async(srv_mock, monkeypatch):
    monkeypatch.setenv("HAROMA_CHAT_DEFAULT_ASYNC", "1")
    srv, shared, inp = srv_mock

    def make_ready_slot():
        ev = threading.Event()
        slot = {
            "event": ev,
            "result": {"response": "sync", "cycle": 0},
        }
        ev.set()
        return slot

    inp.set_slot_factory(make_ready_slot)
    client = srv.app.test_client()
    resp = client.post(
        "/chat",
        json={"message": "hi", "depth": "normal", "async": False},
    )
    assert resp.status_code == 200
    assert (resp.get_json() or {}).get("response") == "sync"
    assert shared.ends == 1


def test_delete_cancels_pending(srv_mock):
    srv, shared, inp = srv_mock
    ev = threading.Event()

    def make_slow_slot():
        return {"event": ev, "result": None}

    inp.set_slot_factory(make_slow_slot)
    client = srv.app.test_client()
    resp = client.post("/chat", json={"message": "x", "async": True, "depth": "normal"})
    rid = (resp.get_json() or {}).get("request_id")
    d = client.delete(f"/chat/result?id={rid}")
    assert d.status_code == 200
    assert (d.get_json() or {}).get("status") == "cancelled"
    assert shared.ends == 1
    assert client.get(f"/chat/result?id={rid}").status_code == 404


def test_delete_409_when_result_ready_not_yet_fetched(srv_mock):
    srv, shared, inp = srv_mock

    def make_ready_slot():
        ev = threading.Event()
        slot = {"event": ev, "result": {"response": "done", "cycle": 0}}
        ev.set()
        return slot

    inp.set_slot_factory(make_ready_slot)
    client = srv.app.test_client()
    resp = client.post("/chat", json={"message": "x", "async": True, "depth": "normal"})
    rid = (resp.get_json() or {}).get("request_id")
    d = client.delete(f"/chat/result?id={rid}")
    assert d.status_code == 409
    g = client.get(f"/chat/result?id={rid}")
    assert g.status_code == 200
    assert (g.get_json() or {}).get("response") == "done"


def test_sse_stream_contains_reply(srv_mock):
    srv, shared, inp = srv_mock

    def make_ready_slot():
        ev = threading.Event()
        slot = {"event": ev, "result": {"response": "streamed", "cycle": 0}}
        ev.set()
        return slot

    inp.set_slot_factory(make_ready_slot)
    client = srv.app.test_client()
    resp = client.post("/chat", json={"message": "x", "async": True, "depth": "normal"})
    rid = (resp.get_json() or {}).get("request_id")
    r = client.get(f"/chat/wait?id={rid}")
    assert r.status_code == 200
    assert "event-stream" in (r.headers.get("Content-Type") or "")
    assert "streamed" in r.get_data(as_text=True)


def test_rate_limit_returns_429_after_cap(srv_mock, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_HTTP_RATE_LIMIT_PER_MIN", "2")
    srv, shared, inp = srv_mock

    def make_ready_slot():
        ev = threading.Event()
        slot = {"event": ev, "result": {"response": "ok", "cycle": shared.cycle_count}}
        ev.set()
        return slot

    inp.set_slot_factory(make_ready_slot)
    client = srv.app.test_client()
    assert client.post("/chat", json={"message": "a", "async": False}).status_code == 200
    assert client.post("/chat", json={"message": "b", "async": False}).status_code == 200
    r3 = client.post("/chat", json={"message": "c", "async": False})
    assert r3.status_code == 429
    body = r3.get_json() or {}
    assert body.get("error") == "rate_limited"
    assert body.get("request_id")
    assert r3.headers.get("Retry-After") == "60"
    assert r3.headers.get("X-Haroma-Request-Id") == body.get("request_id")


def test_status_includes_request_id_header(srv_mock):
    r = srv_mock[0].app.test_client().get("/status")
    assert r.status_code == 200
    hdr = r.headers.get("X-Haroma-Request-Id")
    assert hdr and len(hdr) >= 32


def test_laws_get_snapshot_when_law_unset(srv_mock):
    srv, shared, _ = srv_mock
    assert getattr(shared, "law", None) is None
    r = srv.app.test_client().get("/laws")
    assert r.status_code == 200
    j = r.get_json() or {}
    assert j.get("available") is False


def test_research_snapshot_route(srv_mock):
    srv, shared, inp = srv_mock
    shared.run_manifest = {"manifest_version": 1, "test_marker": True}
    shared.lab_run_id = "lr-unit"
    client = srv.app.test_client()
    r = client.get("/research/snapshot")
    assert r.status_code == 200
    j = r.get_json() or {}
    assert j.get("lab_run_id") == "lr-unit"
    assert j.get("run_manifest", {}).get("test_marker") is True
    assert "lab_experiment_events" in j
    assert "agent_environment" in j
    assert "embodiment_readiness" in j
    assert isinstance(j.get("embodiment_readiness"), dict)
