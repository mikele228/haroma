"""
BaseAgent -- abstract foundation for all Elarion agent types.

Every agent runs in its own daemon thread with a configurable tick interval.
Lifecycle: start() -> _tick() loop -> stop().  Agents communicate exclusively
through the MessageBus; shared cognitive state lives in SharedResources.
"""

from __future__ import annotations

import threading
import time
import traceback
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from agents.message_bus import MessageBus, Message
    from agents.shared_resources import SharedResources


class BaseAgent(ABC):
    """Abstract base for Boot, Input, Background, and Persona agents."""

    AGENT_TYPE: str = "base"

    def __init__(
        self,
        agent_id: str,
        shared: Optional[SharedResources] = None,
        bus: Optional[MessageBus] = None,
        tick_interval: float = 1.0,
    ):
        self.agent_id = agent_id
        self.agent_type = self.AGENT_TYPE
        self.shared = shared
        self.bus = bus
        self.tick_interval = tick_interval

        self._running = False
        self._lifecycle_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._tick_count = 0
        self._error_count = 0
        self._last_tick_time: float = 0.0
        self._start_time: float = 0.0
        self._stop_event = threading.Event()

    # -- lifecycle -----------------------------------------------------

    def start(self):
        """Spin up the agent's daemon thread."""
        with self._lifecycle_lock:
            if self._running:
                return
            self._running = True
            self._start_time = time.time()
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name=f"Agent-{self.agent_type}-{self.agent_id}",
            )
            self._thread.start()
        print(
            f"[{self.agent_type}:{self.agent_id}] started  (interval={self.tick_interval}s)",
            flush=True,
        )

    def stop(self):
        """Signal the agent to stop and wait for thread exit."""
        with self._lifecycle_lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.tick_interval * 3, 5.0))
        print(f"[{self.agent_type}:{self.agent_id}] stopped", flush=True)

    def is_alive(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    # -- main loop -----------------------------------------------------

    def _run_loop(self):
        """Tick repeatedly until stopped."""
        me = threading.current_thread()
        try:
            while self._running:
                t0 = time.time()
                try:
                    self._tick()
                    self._tick_count += 1
                except Exception as exc:
                    self._error_count += 1
                    print(
                        f"[{self.agent_type}:{self.agent_id}] tick error "
                        f"#{self._error_count}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    traceback.print_exc()
                self._last_tick_time = time.time() - t0
                self._stop_event.wait(timeout=self.tick_interval)
                if self._stop_event.is_set() and self._running:
                    self._stop_event.clear()
        finally:
            with self._lifecycle_lock:
                if self._thread is me:
                    self._running = False

    @abstractmethod
    def _tick(self):
        """One heartbeat -- subclasses implement their logic here."""
        ...

    # -- wake (interrupt sleep early) ----------------------------------

    def wake(self):
        """Interrupt the current tick-interval sleep so the agent acts immediately."""
        self._stop_event.set()

    # -- messaging helpers ---------------------------------------------

    def send(self, channel: str, content: Any, **metadata):
        """Publish a message through the bus to a channel."""
        if self.bus is None:
            return
        from agents.message_bus import Message

        msg = Message(
            sender_id=self.agent_id,
            channel=channel,
            content=content,
            message_type=self.agent_type,
            metadata=metadata,
        )
        self.bus.publish(msg)

    def send_direct(self, target_id: str, content: Any, **metadata):
        """Send a point-to-point message to another agent."""
        if self.bus is None:
            return
        from agents.message_bus import Message

        msg = Message(
            sender_id=self.agent_id,
            channel="direct",
            content=content,
            message_type=self.agent_type,
            metadata={"target": target_id, **metadata},
        )
        self.bus.send_direct(target_id, msg)

    def poll(self, timeout: float = 0.0) -> List:
        """Pull pending messages for this agent from the bus."""
        if self.bus is None:
            return []
        return self.bus.poll(self.agent_id, timeout=timeout)

    # -- introspection -------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "alive": self.is_alive(),
            "tick_count": self._tick_count,
            "error_count": self._error_count,
            "last_tick_ms": round(self._last_tick_time * 1000, 1),
            "uptime_s": round(time.time() - self._start_time, 1) if self._start_time else 0,
            "tick_interval": self.tick_interval,
        }

    def __repr__(self):
        alive = "alive" if self.is_alive() else "stopped"
        return f"<{self.agent_type}:{self.agent_id} {alive} ticks={self._tick_count}>"
