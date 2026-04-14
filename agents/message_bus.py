"""
MessageBus -- inter-agent communication hub for Elarion.

Supports:
  - Channel-based pub/sub  (publish -> all subscribers)
  - Direct point-to-point  (send_direct -> one agent)
  - Competitive claiming   (claim -> first-wins atomic lock)
  - Pluggable routing      (BroadcastClaim, Affinity, RoundRobin, PrimaryRelay)
  - Dead-letter queue      (explicit push when routing cannot deliver)
"""

from __future__ import annotations

import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


# =====================================================================
# Message
# =====================================================================


@dataclass
class Message:
    sender_id: str
    channel: str
    content: Any
    message_type: str = "generic"
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def prior_processors(self) -> List[str]:
        """Agents that have already processed this message (relay tracking)."""
        return self.metadata.get("prior_processors", [])

    def mark_processed_by(self, agent_id: str):
        self.metadata.setdefault("prior_processors", []).append(agent_id)

    @property
    def response_slot(self) -> Optional[Dict]:
        """Event-based slot for HTTP callers waiting on a reply."""
        return self.metadata.get("_response_slot")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "channel": self.channel,
            "message_type": self.message_type,
            "timestamp": self.timestamp,
            "content_preview": str(self.content)[:120],
            "metadata_keys": list(self.metadata.keys()),
            "prior_processors": self.prior_processors,
        }


# =====================================================================
# MessageBus
# =====================================================================


class MessageBus:
    """Thread-safe message routing hub."""

    def __init__(self, dead_letter_timeout_ms: float = 2000.0):
        # dead_letter_timeout_ms is stored for API/backward compatibility only.
        # Dead letters are pushed explicitly (e.g. routers with no targets), not
        # by a timed "unclaimed message" sweep.
        self._lock = threading.Lock()
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self._mailboxes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=256))

        self._claim_lock = threading.Lock()
        self._claims: Dict[str, str] = {}

        self._dead_letter: deque = deque(maxlen=128)
        self._dead_letter_timeout = dead_letter_timeout_ms / 1000.0

        self._total_published = 0
        self._total_claimed = 0
        self._total_dead = 0

    # -- subscriptions -------------------------------------------------

    def subscribe(self, channel: str, agent_id: str):
        with self._lock:
            self._subscriptions[channel].add(agent_id)

    def unsubscribe(self, channel: str, agent_id: str):
        with self._lock:
            self._subscriptions[channel].discard(agent_id)

    def unsubscribe_all(self, agent_id: str):
        """Remove agent from every channel and delete its mailbox."""
        with self._lock:
            for subs in self._subscriptions.values():
                subs.discard(agent_id)
            self._mailboxes.pop(agent_id, None)

    # -- publish (channel broadcast) -----------------------------------

    def publish(self, message: Message):
        """Deliver message to all agents subscribed to its channel."""
        with self._lock:
            subscribers = set(self._subscriptions.get(message.channel, set()))
            for agent_id in subscribers:
                self._mailboxes[agent_id].append(message)
            self._total_published += 1

    # -- direct send ---------------------------------------------------

    def send_direct(self, target_id: str, message: Message):
        """Deliver message to exactly one agent's mailbox."""
        with self._lock:
            self._mailboxes[target_id].append(message)

    # -- poll (agent pulls messages) -----------------------------------

    def poll(self, agent_id: str, timeout: float = 0.0) -> List[Message]:
        """Return and clear all pending messages for *agent_id*."""
        with self._lock:
            box = self._mailboxes.get(agent_id)
            if not box:
                return []
            messages = list(box)
            box.clear()
        return messages

    # -- competitive claiming ------------------------------------------

    def claim(self, message_id: str, agent_id: str) -> bool:
        """Atomically claim a message. Returns True if this agent wins.

        .. deprecated::
            Not used by any current routing strategy. Retained for
            potential future broadcast-claim routing.
        """
        with self._claim_lock:
            if message_id in self._claims:
                return False
            self._claims[message_id] = agent_id
            self._total_claimed += 1
            return True

    def is_claimed(self, message_id: str) -> bool:
        with self._claim_lock:
            return message_id in self._claims

    def get_claimer(self, message_id: str) -> Optional[str]:
        with self._claim_lock:
            return self._claims.get(message_id)

    # -- dead-letter queue ---------------------------------------------

    def push_dead_letter(self, message: Message):
        """Enqueue a message for BackgroundAgent (e.g. unroutable input). Not time-based."""
        with self._lock:
            self._dead_letter.append(message)
            self._total_dead += 1

    def drain_dead_letters(self) -> List[Message]:
        with self._lock:
            letters = list(self._dead_letter)
            self._dead_letter.clear()
        return letters

    # -- claim garbage collection --------------------------------------

    def gc_claims(self, max_entries: int = 10000):
        """Evict oldest claim records when the table grows too large."""
        with self._claim_lock:
            if len(self._claims) > max_entries:
                to_remove = list(self._claims.keys())[: max_entries // 2]
                for k in to_remove:
                    del self._claims[k]

    # -- introspection -------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            mailbox_sizes = {aid: len(box) for aid, box in self._mailboxes.items()}
            subscriptions = {ch: list(subs) for ch, subs in self._subscriptions.items()}
            with self._claim_lock:
                active_claims = len(self._claims)
                total_claimed = self._total_claimed
            return {
                "subscriptions": subscriptions,
                "mailbox_sizes": mailbox_sizes,
                "total_published": self._total_published,
                "total_claimed": total_claimed,
                "dead_letter_count": len(self._dead_letter),
                "total_dead": self._total_dead,
                "active_claims": active_claims,
            }


# =====================================================================
# Routing strategies (pluggable)
# =====================================================================


class MessageRouter(ABC):
    """Pluggable strategy deciding how input messages reach persona agents."""

    @abstractmethod
    def route(
        self,
        message: Message,
        bus: MessageBus,
        persona_ids: List[str],
    ) -> None: ...


class BroadcastClaimRouter(MessageRouter):
    """Default: broadcast to all personas; first to claim() wins.

    If only one persona exists, skips the claim dance entirely.
    """

    def __init__(self, claim_timeout_ms: float = 500.0):
        self.claim_timeout = claim_timeout_ms / 1000.0

    def route(
        self,
        message: Message,
        bus: MessageBus,
        persona_ids: List[str],
    ) -> None:
        if not persona_ids:
            bus.push_dead_letter(message)
            return
        if len(persona_ids) == 1:
            bus.send_direct(persona_ids[0], message)
            return
        for pid in persona_ids:
            bus.send_direct(pid, message)


class AffinityRouter(MessageRouter):
    """Score each persona's affinity to the message, route to highest."""

    def __init__(self, affinity_scorer: Optional[Callable] = None):
        self._scorer = affinity_scorer or self._default_score

    @staticmethod
    def _default_score(message: Message, persona_id: str) -> float:
        return 0.5

    def set_scorer(self, scorer: Callable):
        self._scorer = scorer

    def route(
        self,
        message: Message,
        bus: MessageBus,
        persona_ids: List[str],
    ) -> None:
        if not persona_ids:
            bus.push_dead_letter(message)
            return
        scored = [(self._scorer(message, pid), pid) for pid in persona_ids]
        scored.sort(key=lambda x: x[0], reverse=True)
        bus.send_direct(scored[0][1], message)


class RoundRobinRouter(MessageRouter):
    """Simple rotation across persona agents."""

    def __init__(self):
        self._index = 0
        self._lock = threading.Lock()

    def route(
        self,
        message: Message,
        bus: MessageBus,
        persona_ids: List[str],
    ) -> None:
        if not persona_ids:
            bus.push_dead_letter(message)
            return
        with self._lock:
            idx = self._index % len(persona_ids)
            self._index += 1
        bus.send_direct(persona_ids[idx], message)


class PrimaryRelayRouter(MessageRouter):
    """Always send to the first (primary/default) persona."""

    def route(
        self,
        message: Message,
        bus: MessageBus,
        persona_ids: List[str],
    ) -> None:
        if not persona_ids:
            bus.push_dead_letter(message)
            return
        bus.send_direct(persona_ids[0], message)


# -- router factory ----------------------------------------------------

ROUTER_REGISTRY: Dict[str, type] = {
    "broadcast_claim": BroadcastClaimRouter,
    "affinity": AffinityRouter,
    "round_robin": RoundRobinRouter,
    "primary_relay": PrimaryRelayRouter,
}


def create_router(strategy: str = "broadcast_claim", **kwargs) -> MessageRouter:
    """Instantiate a router by strategy name."""
    cls = ROUTER_REGISTRY.get(strategy)
    if cls is None:
        print(
            f"[MessageRouter] Unknown strategy '{strategy}', falling back to broadcast_claim",
            flush=True,
        )
        cls = BroadcastClaimRouter
    return cls(**kwargs)
