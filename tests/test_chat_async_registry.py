"""Async submit→poll chat registry and server helpers."""

import threading
import unittest.mock as mock


def test_truthy_async_flag():
    from mind.chat_async_registry import truthy_async_flag

    assert truthy_async_flag(True) is True
    assert truthy_async_flag("true") is True
    assert truthy_async_flag("1") is True
    assert truthy_async_flag("on") is True
    assert truthy_async_flag(False) is False
    assert truthy_async_flag("false") is False
    assert truthy_async_flag(None) is False
    assert truthy_async_flag("") is False


def test_registry_poll_completes_and_manual_end():
    class Shared:
        def __init__(self):
            self.ends = 0

        def http_chat_end(self):
            self.ends += 1

    from mind.chat_async_registry import ChatAsyncRegistry

    shared = Shared()
    reg = ChatAsyncRegistry(shared, ttl_sec=60.0)
    ev = threading.Event()
    slot = {"event": ev, "result": None}

    rid = reg.register(slot, experiment_id="exp-a", lab_run_id="lr-1")
    assert reg.get(rid) is not None

    ent = reg.get(rid)
    assert ent["experiment_id"] == "exp-a"
    assert ent["lab_run_id"] == "lr-1"
    assert ent["slot"]["event"].is_set() is False

    slot["result"] = {"response": "hi", "cycle": 1}
    ev.set()

    ent2 = reg.get(rid)
    assert ent2["slot"]["event"].is_set() is True

    reg.pop(rid)
    shared.http_chat_end()
    assert shared.ends == 1
    assert reg.get(rid) is None


def test_registry_cancel_pending():
    from mind.chat_async_registry import ChatAsyncRegistry

    class Shared:
        def __init__(self):
            self.ends = 0

        def http_chat_end(self):
            self.ends += 1

    shared = Shared()
    reg = ChatAsyncRegistry(shared, ttl_sec=60.0)
    ev = threading.Event()
    slot = {"event": ev, "result": None}
    rid = reg.register(slot)
    assert reg.cancel(rid) == "ok"
    assert reg.get(rid) is None
    assert shared.ends == 1


def test_registry_cancel_not_pending_when_done():
    from mind.chat_async_registry import ChatAsyncRegistry

    class Shared:
        def http_chat_end(self) -> None:
            pass

    reg = ChatAsyncRegistry(Shared(), ttl_sec=60.0)
    ev = threading.Event()
    slot = {"event": ev, "result": {"response": "x", "cycle": 0}}
    ev.set()
    rid = reg.register(slot)
    assert reg.cancel(rid) == "not_pending"


def test_registry_pending_count():
    from mind.chat_async_registry import ChatAsyncRegistry

    class Shared:
        def http_chat_end(self) -> None:
            pass

    reg = ChatAsyncRegistry(Shared(), ttl_sec=60.0)
    ev = threading.Event()
    slot = {"event": ev, "result": None}
    rid = reg.register(slot)
    assert reg.pending_count() == 1
    reg.pop(rid)
    assert reg.pending_count() == 0


def test_registry_evict_stale_calls_http_end():
    class Shared:
        def __init__(self):
            self.ends = 0

        def http_chat_end(self):
            self.ends += 1

    from mind.chat_async_registry import ChatAsyncRegistry

    shared = Shared()
    reg = ChatAsyncRegistry(shared, ttl_sec=1.0)
    ev = threading.Event()
    slot = {"event": ev, "result": None}

    clock = [1000.0]

    def fake_time():
        return clock[0]

    with mock.patch("mind.chat_async_registry.time.time", fake_time):
        rid = reg.register(slot)
        assert rid
        clock[0] = 1002.5
        assert reg.get(rid) is None

    assert shared.ends == 1
