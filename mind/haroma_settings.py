"""Central Haroma runtime settings from ``os.environ`` (``HAROMA_*``, ``ELARION_*``).

Prefer functions and datatypes here instead of ad-hoc ``os.environ.get`` so defaults
and truthy sets stay aligned. Low-level parsers: :mod:`mind.config_env`. HTTP bind
and ``.env`` loading: :mod:`mind.deploy_config`.

Settings used at **module import time** in other modules (e.g. ``cycle_flow``) call
the functions here once at import — same pattern as before centralization.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from mind.config_env import env_int, env_truthy


def synthetic_llm_dummy_reply_env() -> bool:
    """``HAROMA_LLM_DUMMY_REPLY`` — skip native ``generate_chat`` (dummy / probe)."""
    return env_truthy("HAROMA_LLM_DUMMY_REPLY", default=False)


def packed_llm_dummy_reply_raw() -> str:
    """Raw ``HAROMA_LLM_DUMMY_REPLY`` string (logging)."""
    return str(os.environ.get("HAROMA_LLM_DUMMY_REPLY", "") or "")


def haroma_controller_packed_llm_enabled() -> bool:
    """``HAROMA_CONTROLLER_PACKED_LLM`` — run packed LLM inside ``ElarionController.run_cycle``."""
    return env_truthy("HAROMA_CONTROLLER_PACKED_LLM", default=False)


def haroma_chat_llm_primary_enabled() -> bool:
    """``HAROMA_CHAT_LLM_PRIMARY`` — conversant fast path uses packed LLM as primary voice (default on)."""
    raw = str(os.environ.get("HAROMA_CHAT_LLM_PRIMARY", "1") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def haroma_llm_centric_env_enabled() -> bool:
    """Raw ``HAROMA_LLM_CENTRIC`` switch (not ``0`` / ``false`` / empty)."""
    return os.environ.get("HAROMA_LLM_CENTRIC", "0") not in ("0", "false", "")


def elarion_cycle_flow_debug() -> bool:
    """``ELARION_CYCLE_FLOW_DEBUG`` — extra logging in :mod:`mind.cycle_flow`."""
    return str(os.environ.get("ELARION_CYCLE_FLOW_DEBUG", "0") or "0").strip() == "1"


def haroma_state_prompt_max_chars() -> int:
    """``HAROMA_STATE_PROMPT_MAX_CHARS`` — cap for state JSON in prompts (clamped)."""
    return max(512, min(100_000, env_int("HAROMA_STATE_PROMPT_MAX_CHARS", 3000)))


@dataclass(frozen=True)
class MultiGoalDeliberativeEnv:
    """Flags and caps for :func:`mind.cycle_flow.run_multi_goal_deliberative_actions`."""

    enabled: bool
    max_cycle_goals: int
    max_actions_per_goal: int


def read_multi_goal_deliberative_env() -> MultiGoalDeliberativeEnv:
    """``HAROMA_MULTI_GOAL_PER_CYCLE`` and per-goal caps."""
    enabled = env_truthy("HAROMA_MULTI_GOAL_PER_CYCLE", default=False)
    try:
        max_cycle_goals = max(1, int(os.environ.get("HAROMA_MAX_CYCLE_GOALS", "3") or 3))
    except (TypeError, ValueError):
        max_cycle_goals = 3
    try:
        max_actions_per_goal = max(1, int(os.environ.get("HAROMA_MAX_ACTIONS_PER_GOAL", "2") or 2))
    except (TypeError, ValueError):
        max_actions_per_goal = 2
    return MultiGoalDeliberativeEnv(
        enabled=enabled,
        max_cycle_goals=max_cycle_goals,
        max_actions_per_goal=max_actions_per_goal,
    )
