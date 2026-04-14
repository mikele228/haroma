"""
Elarion Multi-Agent Architecture (v6 -- TrueSelf).

Five agent types cooperate to form Elarion's mind:
  - BootAgent:       loads soul + data, spawns all other agents, supervises
  - InputAgent:      collects peripheral inputs (sensors, text), lightweight NLU,
                     forwards everything to TrueSelf
  - TrueSelfAgent:   executive consciousness -- fast-path or delegates to personas
  - BackgroundAgent: subconscious processing -- dreams, reconciliation, training,
                     persistence, goal synthesis, dynamic persona spawning
  - PersonaAgent:    specialist inner voice -- receives delegated work from
                     TrueSelf, sends results back, relays to siblings
"""

from agents.base import BaseAgent
from agents.message_bus import MessageBus, Message, MessageRouter
from agents.shared_resources import SharedResources

__all__ = [
    "BaseAgent",
    "MessageBus",
    "Message",
    "MessageRouter",
    "SharedResources",
]
