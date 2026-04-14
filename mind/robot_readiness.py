"""
Heuristic embodiment readiness summary for operators and ``GET /status``.

Not a safety certification — informational scores in ``[0, 1]`` derived from
``agent_environment_status()``-shaped data.
"""

from __future__ import annotations

from typing import Any, Dict


def embodiment_readiness_summary(shared: Any) -> Dict[str, Any]:
    """Return compact scores + notes from :meth:`agents.shared_resources.SharedResources.agent_environment_status`."""
    try:
        st = shared.agent_environment_status()
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}

    scores: Dict[str, float] = {}
    notes: list[str] = []

    if st.get("has_environment"):
        scores["environment_bound"] = 1.0
    else:
        scores["environment_bound"] = 0.0
        notes.append("no_agent_environment_posted")

    rb = st.get("robot_bridge") if isinstance(st.get("robot_bridge"), dict) else {}
    if rb.get("has_bridge"):
        rc = int(rb.get("results_count") or 0)
        scores["bridge_feedback"] = min(1.0, 0.25 + 0.15 * min(rc, 5))
        if rc == 0:
            notes.append("robot_bridge_empty_results")
    else:
        scores["bridge_feedback"] = 0.2
        notes.append("no_robot_bridge_data")

    rbm = st.get("robot_bridge_metrics") if isinstance(st.get("robot_bridge_metrics"), dict) else {}
    ok_p = int(rbm.get("feedback_posts_ok") or 0)
    err_p = int(rbm.get("feedback_posts_error") or 0)
    if ok_p + err_p > 0:
        scores["bridge_post_reliability"] = ok_p / max(1, ok_p + err_p)
    else:
        scores["bridge_post_reliability"] = 0.5
        notes.append("no_bridge_http_feedback_yet")

    rbod = st.get("robot_body_defined")
    if rbod is True:
        scores["robot_body_defined"] = 1.0
    elif rbod is False:
        scores["robot_body_defined"] = 0.3
        notes.append("robot_body_undefined")
    else:
        scores["robot_body_defined"] = 0.5

    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    overall = round(sum(vals) / max(1, len(vals)), 3) if vals else 0.0

    return {
        "ok": True,
        "overall": overall,
        "scores": scores,
        "notes": notes[:8],
    }
