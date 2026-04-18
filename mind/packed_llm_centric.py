"""``HAROMA_LLM_CENTRIC`` gate for PersonaAgent (board mandate, internal, TrueSelf)."""

from __future__ import annotations

from typing import Any, Optional

from mind.haroma_settings import haroma_llm_centric_env_enabled


def llm_centric_enabled_for_persona_cycle(
    *,
    is_internal: bool,
    trueself_agent: bool,
    user_or_traced_turn: bool,
    goal_board: Optional[Any] = None,
) -> bool:
    """True when packed LLM may drive the reply voice (see PersonaAgent module doc).

    Disabled on internal cycles, when the goal board has an active mandate, or when
    ``HAROMA_LLM_CENTRIC`` is off. For TrueSelf-hosted personas, requires a user or
    traced turn.
    """
    cfg = haroma_llm_centric_env_enabled()
    mandate_active = (
        goal_board.has_active_mandate() if goal_board is not None else False
    )
    return (
        cfg
        and not is_internal
        and not mandate_active
        and (not trueself_agent or user_or_traced_turn)
    )
