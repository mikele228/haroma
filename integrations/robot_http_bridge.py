"""
HTTP-oriented **robot bridge** helpers.

Typical flow:

1. Cognition produces ``body_actions`` (see :mod:`engine.LLMContextReasoner`).
2. :func:`mind.robot_execution_contract.build_executor_command_batch` builds a batch.
3. Your bridge POSTs that batch to the robot controller HTTP API.
4. The robot POSTs feedback to Haroma (e.g. ``POST /agent/environment``) with
   ``extensions.robot_bridge`` populated via
   :func:`merge_feedback_into_agent_environment`.

ROS 2: implement the same JSON in a ``rosbridge`` or custom node; keep
``command_id`` / ``correlation_id`` for tracing.

See ``docs/robot-cognitive-control-split.md`` for how this bridge fits under
real-time control and safety layers on the robot.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from mind.robot_execution_contract import merge_robot_bridge_history, normalize_feedback_payload


def append_robot_bridge_to_extensions(
    extensions: Dict[str, Any],
    robot_bridge_block: Dict[str, Any],
) -> Dict[str, Any]:
    """Return *extensions* with ``robot_bridge`` merged (rolling results, deduped by ``command_id``)."""
    out = dict(extensions) if extensions else {}
    rb = dict(robot_bridge_block) if robot_bridge_block else {}
    prev = out.get("robot_bridge")
    out["robot_bridge"] = merge_robot_bridge_history(
        prev if isinstance(prev, dict) else None,
        rb,
    )
    return out


def merge_feedback_into_agent_environment(
    agent_environment: Dict[str, Any],
    feedback_raw: Any,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Immutably merge robot feedback into ``agent_environment.extensions.robot_bridge``.

    Returns ``(new_agent_environment, error_or_none)``.
    """
    block, err = normalize_feedback_payload(feedback_raw)
    if err:
        return deepcopy(agent_environment) if agent_environment else {}, err
    base = deepcopy(agent_environment) if agent_environment else {}
    ext = base.get("extensions")
    if not isinstance(ext, dict):
        ext = {}
    base["extensions"] = append_robot_bridge_to_extensions(ext, block)
    return base, None
