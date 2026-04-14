"""Integrations — bridges to hosts, robots, and simulators."""

from integrations.robot_http_bridge import (
    append_robot_bridge_to_extensions,
    merge_feedback_into_agent_environment,
)
from integrations.sim import (
    SimulationBackend,
    create_backend,
    load_backend_from_env,
    merge_simulation_into_extensions,
    register_backend,
    simulation_summary_for_prompt,
)

__all__ = [
    "SimulationBackend",
    "append_robot_bridge_to_extensions",
    "create_backend",
    "load_backend_from_env",
    "merge_feedback_into_agent_environment",
    "merge_simulation_into_extensions",
    "register_backend",
    "simulation_summary_for_prompt",
]
