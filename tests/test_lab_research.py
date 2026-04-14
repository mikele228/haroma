"""mind.lab_research — manifests and experiment id parsing."""

from __future__ import annotations

import sys

import pytest

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mind import lab_research as lr


def test_parse_experiment_id_header():
    class H:
        def get(self, k, d=""):
            return {"X-Experiment-Id": "exp-a"}.get(k, d)

    assert lr.parse_experiment_id(headers=H(), body=None) == "exp-a"


def test_parse_experiment_id_json_wins_if_header_missing():
    assert (
        lr.parse_experiment_id(headers={}, body={"experiment_id": "from-json"})
        == "from-json"
    )


def test_parse_experiment_id_header_wins_over_body():
    class H:
        def get(self, k, d=""):
            return {"X-Experiment-Id": "hdr"}.get(k, d)

    assert (
        lr.parse_experiment_id(headers=H(), body={"experiment_id": "body"})
        == "hdr"
    )


def test_parse_experiment_id_empty():
    assert lr.parse_experiment_id(headers={}, body={}) is None


def test_build_run_manifest_shape(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_LAB_SEED", "42")
    class S:
        resource_config = type("RC", (), {"tier_name": "edge", "tier": 1})()
        llm_backend = type("LB", (), {"backend_type": "local", "model_name": "x", "available": True})()

    m = lr.build_run_manifest(shared=S())
    assert m["manifest_version"] >= 1
    assert m["haroma_lab_seed"] == "42"
    assert m["env_haroma_elarion"].get("HAROMA_LAB_SEED") == "42"
    assert m["llm_backend"]["model_name"] == "x"


def test_lab_events_ring():
    lr.record_lab_http_event("/chat", "e1")
    lr.record_lab_http_event("/agent/environment", "e2")
    snap = lr.lab_events_snapshot(10)
    assert len(snap) == 2
    assert snap[-1]["experiment_id"] == "e2"


def test_apply_lab_experiment_to_request_sets_g():
    class R:
        method = "POST"
        path = "/chat"
        headers = {"X-Experiment-Id": "flow-test"}

        def get_json(self, silent=True):
            return {"message": "hi"}

    class G:
        lab_experiment_id = None

    g = G()
    lr.apply_lab_experiment_to_request(R(), g)
    assert g.lab_experiment_id == "flow-test"
    snap = lr.lab_events_snapshot(3)
    assert snap and snap[-1]["path"] == "/chat"


def test_merge_lab_context():
    class G:
        lab_experiment_id = "exp-9"

    class S:
        lab_run_id = "run-1"

    m = lr.merge_lab_context({"ok": True}, S(), G())
    assert m["ok"] is True
    assert m["experiment_id"] == "exp-9"
    assert m["lab_run_id"] == "run-1"


def test_merge_lab_context_values_direct():
    m = lr.merge_lab_context_values(
        {"x": 1}, experiment_id="e", lab_run_id="r"
    )
    assert m["x"] == 1
    assert m["experiment_id"] == "e"
    assert m["lab_run_id"] == "r"


def test_merge_lab_context_accepts_experiment_id_zero_string():
    class G:
        lab_experiment_id = "0"

    class S:
        lab_run_id = ""

    m = lr.merge_lab_context({"a": 1}, S(), G())
    assert m["experiment_id"] == "0"


def test_haroma_lab_log_prints_when_enabled(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_LAB_LOG", "1")

    class R:
        method = "POST"
        path = "/chat"
        headers = {"X-Experiment-Id": "exp-log-line"}

        def get_json(self, silent=True):
            return {}

    class G:
        lab_experiment_id = None

    g = G()
    lr.apply_lab_experiment_to_request(R(), g)
    out = capsys.readouterr().out
    assert "[HAROMA lab]" in out
    assert "exp-log-line" in out


def test_apply_lab_skips_non_lab_paths():
    class R:
        method = "POST"
        path = "/teach"
        headers = {}

        def get_json(self, silent=True):
            return {}

    class G:
        lab_experiment_id = "should_clear"

    g = G()
    lr.apply_lab_experiment_to_request(R(), g)
    assert g.lab_experiment_id is None
