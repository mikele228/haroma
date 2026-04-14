"""Typing protocols for reward / composite scoring (optional, structural)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SupportsCompositeReward(Protocol):
    """Minimum surface for :func:`mind.training.vw_rl_bridge.composite_trained_scores`."""

    reward_model: Any

    def effective_env_summary_for_vw_scoring(self) -> str:
        """Cached environment text for VW ``|e``, with TTL applied (may be ``""``)."""
        ...
