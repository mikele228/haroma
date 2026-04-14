"""Robot bridge feedback counters on SharedResources (GET /status via agent_environment)."""

from __future__ import annotations

import sys

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from agents.shared_resources import SharedResources


def test_merge_robot_bridge_feedback_increments_metrics():
    s = SharedResources()
    bad = s.merge_robot_bridge_feedback({"bridge_schema_version": 99, "results": []})
    assert bad.get("ok") is False
    m = s.robot_bridge_metrics_snapshot()
    assert m.get("feedback_posts_error", 0) == 1
    assert m.get("feedback_posts_ok", 0) == 0

    ok = s.merge_robot_bridge_feedback(
        {
            "bridge_schema_version": 1,
            "correlation_id": "c-test",
            "results": [
                {"command_id": "cmd_a", "status": "completed", "detail": "ok"},
                {"command_id": "cmd_b", "status": "failed", "detail": "x"},
            ],
        }
    )
    assert ok.get("ok") is True
    m2 = s.robot_bridge_metrics_snapshot()
    assert m2.get("feedback_posts_ok", 0) == 1
    assert m2.get("feedback_result_rows_accepted", 0) == 2

    st = s.agent_environment_status()
    assert st["robot_bridge_metrics"]["feedback_posts_ok"] == 1
    assert st["robot_bridge"]["has_bridge"] is True


def test_agent_environment_status_includes_empty_bridge_metrics():
    s = SharedResources()
    st = s.agent_environment_status()
    assert st["robot_bridge_metrics"] == {}
    assert st["robot_bridge"]["has_bridge"] is False
    assert st["lab_experiment_events"] == []
    assert st.get("lab_run_id") in (None, "")
