"""mind.robot_readiness — embodiment heuristic summary."""

from __future__ import annotations

import sys


_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mind.robot_readiness import embodiment_readiness_summary


def test_embodiment_summary_empty_environment():
    class S:
        def agent_environment_status(self):
            return {"has_environment": False, "robot_bridge": {}, "robot_bridge_metrics": {}}

    out = embodiment_readiness_summary(S())
    assert out.get("ok") is True
    assert "overall" in out
    assert "no_agent_environment_posted" in (out.get("notes") or [])


def test_embodiment_summary_with_bridge_metrics():
    class S:
        def agent_environment_status(self):
            return {
                "has_environment": True,
                "robot_body_defined": True,
                "robot_bridge": {"has_bridge": True, "results_count": 2},
                "robot_bridge_metrics": {"feedback_posts_ok": 3, "feedback_posts_error": 1},
            }

    out = embodiment_readiness_summary(S())
    assert out.get("ok") is True
    assert out.get("overall", 0) > 0.3
    assert "scores" in out
