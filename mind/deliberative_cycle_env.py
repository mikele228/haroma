"""Shared env parsing for multi-goal deliberative action (controller + PersonaAgent).

Implementation lives in :mod:`mind.haroma_settings`; this module re-exports for
backward-compatible imports (``from mind.deliberative_cycle_env import …``).
"""

from __future__ import annotations

from mind.haroma_settings import MultiGoalDeliberativeEnv, read_multi_goal_deliberative_env

__all__ = ["MultiGoalDeliberativeEnv", "read_multi_goal_deliberative_env"]
