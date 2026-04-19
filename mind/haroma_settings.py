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

# Upper bound for ``HAROMA_DIALOGUE_PHASE`` (see :mod:`mind.dialogue_phases`).
HAROMA_DIALOGUE_PHASE_MAX = 9


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


def haroma_dialogue_phase() -> int:
    """``HAROMA_DIALOGUE_PHASE`` — dialogue roadmap tier (1–9). See :mod:`mind.dialogue_phases`."""
    try:
        p = int(str(os.environ.get("HAROMA_DIALOGUE_PHASE", "1") or "1").strip())
    except (TypeError, ValueError):
        p = 1
    return max(1, min(HAROMA_DIALOGUE_PHASE_MAX, p))


def haroma_dialogue_eval_log_enabled() -> bool:
    """``HAROMA_DIALOGUE_EVAL_LOG`` — stderr one line per enriched discourse when phase >= 3."""
    return env_truthy("HAROMA_DIALOGUE_EVAL_LOG", default=False)


def haroma_memory_recall_intensity() -> int:
    """``HAROMA_MEMORY_RECALL_INTENSITY`` — semantic recall strength (0–10).

    * **0** — skip recall entirely (no retrieved memories this cycle).
    * **10** — use the full computed recall limit (default; same as unset).
    * **1–9** — scale the effective ``recall_limit`` down: ``ceil(limit * intensity / 10)``.
    """
    try:
        v = int(str(os.environ.get("HAROMA_MEMORY_RECALL_INTENSITY", "10") or "10").strip())
    except (TypeError, ValueError):
        v = 10
    return max(0, min(10, v))


def haroma_recall_cmem_only() -> bool:
    """``HAROMA_RECALL_CMEM_ONLY`` — prefer semantic recall from ``cmem`` tree only."""
    return env_truthy("HAROMA_RECALL_CMEM_ONLY", default=False)


def haroma_cmem_recall_fallback_forest() -> bool:
    """``HAROMA_CMEM_RECALL_FALLBACK_FOREST`` — if cmem-only recall finds nothing, run full-forest dense recall.

    Default **off**: an empty or tiny ``cmem`` tree should not trigger an expensive
    scan of the entire semantic index (can dominate latency on large forests).
    Set to ``1`` to restore the previous “never empty recall” behavior.
    """
    return env_truthy("HAROMA_CMEM_RECALL_FALLBACK_FOREST", default=False)


def haroma_cmem_build_enabled() -> bool:
    """``HAROMA_CMEM_BUILD_ENABLED`` — BackgroundAgent writes consolidated nodes into ``cmem`` (default on)."""
    return env_truthy("HAROMA_CMEM_BUILD_ENABLED", default=True)


def haroma_cmem_recall_max_probe() -> int:
    """``HAROMA_CMEM_RECALL_MAX_PROBE`` — max dense index neighbors to scan when filtering by tree (default 2000)."""
    return max(64, min(500_000, env_int("HAROMA_CMEM_RECALL_MAX_PROBE", 2000)))


def haroma_cmem_merge_prime() -> bool:
    """``HAROMA_CMEM_MERGE_PRIME`` — when cmem-only recall, still prepend nexus ``prime`` nodes (default off)."""
    return env_truthy("HAROMA_CMEM_MERGE_PRIME", default=False)


def haroma_cmem_build_every_n_ticks() -> int:
    """``HAROMA_CMEM_BUILD_EVERY_N_TICKS`` — run cmem builder every N BackgroundAgent ticks (default 4)."""
    try:
        v = int(str(os.environ.get("HAROMA_CMEM_BUILD_EVERY_N_TICKS", "4") or "4").strip())
    except (TypeError, ValueError):
        v = 4
    return max(1, min(100, v))


def haroma_cmem_build_max_nodes_per_tick() -> int:
    """``HAROMA_CMEM_BUILD_MAX_NODES_PER_TICK`` — max new ``cmem`` nodes per builder run (default 12)."""
    return max(1, min(500, env_int("HAROMA_CMEM_BUILD_MAX_NODES_PER_TICK", 12)))


def haroma_cmem_bootstrap_max_nodes() -> int:
    """``HAROMA_CMEM_BOOTSTRAP_MAX_NODES`` — one-time boot fill when ``cmem`` is empty (default 128)."""
    return max(32, min(2000, env_int("HAROMA_CMEM_BOOTSTRAP_MAX_NODES", 128)))


def haroma_cmem_max_total_nodes() -> int:
    """``HAROMA_CMEM_MAX_TOTAL_NODES`` — max ``cmem`` nodes across all branches; 0 = unlimited (default)."""
    v = env_int("HAROMA_CMEM_MAX_TOTAL_NODES", 0)
    return max(0, min(2_000_000, v))


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
