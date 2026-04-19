"""
ActionAgent — CEO role: advances execution of the board-ratified mandate.

The president (TrueSelf) sets a mandate on ``SharedResources.goal_board`` when
persona proposals reach consensus. This agent ticks independently, records CEO
progress in memory and goal mission log, and completes the mandate after a
configured number of ticks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.base import BaseAgent
from agents.message_bus import MessageBus

if TYPE_CHECKING:
    from agents.shared_resources import SharedResources


class ActionAgent(BaseAgent):
    AGENT_TYPE = "action"

    def __init__(
        self,
        shared: SharedResources,
        bus: MessageBus,
        tick_interval: float = 1.0,
    ):
        super().__init__(
            agent_id="action_agent",
            shared=shared,
            bus=bus,
            tick_interval=tick_interval,
        )

    def _tick(self):
        if not self._running:
            return
        gb = getattr(self.shared, "goal_board", None)
        if gb is None or not gb.has_active_mandate():
            return
        try:
            gb.tick_ceo_execution(self.shared, self.agent_id)
        except Exception as exc:
            print(f"[ActionAgent/CEO] tick error: {exc}", flush=True)
