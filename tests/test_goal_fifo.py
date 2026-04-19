"""Tests for the FIFO input-goal queue on GoalEngine and InputAgent unification."""

import os
import sys
import threading
import time
from typing import List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests._import_guard import torch_loads_in_subprocess

from core.Goal import GoalEngine, reset_shared_goal_engine_for_tests


# ---------------------------------------------------------------------------
# GoalEngine FIFO basics
# ---------------------------------------------------------------------------


class TestGoalEngineFIFO:
    def _make_engine(self) -> GoalEngine:
        with patch.object(GoalEngine, "load_state", lambda self: setattr(self, "state", {})):
            return GoalEngine()

    def test_register_and_queue_order(self):
        eng = self._make_engine()
        eng.register_input_goal("g1", "first")
        eng.register_input_goal("g2", "second")
        eng.register_input_goal("g3", "third")
        assert eng.input_goal_queue == ["g1", "g2", "g3"]
        assert not eng.goals["g1"]["completed"]

    def test_prioritize_fifo_order(self):
        eng = self._make_engine()
        eng.register_input_goal("g1", "first", priority=0.1)
        eng.register_input_goal("g2", "second", priority=0.9)
        # Non-input goal with highest priority
        eng.register_goal("sys1", "system goal", priority=1.0, source="system")
        order = eng.prioritize()
        assert order[:2] == ["g1", "g2"], "FIFO input goals must come first"
        assert "sys1" in order

    def test_complete_pops_head(self):
        eng = self._make_engine()
        eng.register_input_goal("g1", "first")
        eng.register_input_goal("g2", "second")
        assert eng.complete_input_goal("g1")
        assert eng.goals["g1"]["completed"]
        # g1 should be drained from queue head
        assert "g1" not in eng.input_goal_queue
        assert eng.input_goal_queue == ["g2"]

    def test_complete_out_of_order(self):
        """Completing a non-head goal marks it completed but doesn't pop it
        until the head is also completed."""
        eng = self._make_engine()
        eng.register_input_goal("g1", "first")
        eng.register_input_goal("g2", "second")
        eng.register_input_goal("g3", "third")
        # Complete g2 (not head)
        assert eng.complete_input_goal("g2")
        assert eng.goals["g2"]["completed"]
        # g1 still heads the queue
        assert eng.input_goal_queue[0] == "g1"
        # Now complete g1 — both g1 and g2 should drain from head
        assert eng.complete_input_goal("g1")
        assert eng.input_goal_queue == ["g3"]

    def test_orphan_queue_head_skipped_on_prioritize(self):
        eng = self._make_engine()
        eng.register_input_goal("g1", "first")
        # Orphan id on queue (no goals[ghost]) must not appear in priorities
        eng.input_goal_queue.insert(0, "ghost")
        order = eng.prioritize()
        assert order == ["g1"], "ghost id must not block FIFO"

    def test_orphan_head_popped_when_completing_next(self):
        eng = self._make_engine()
        eng.input_goal_queue = ["ghost", "g1"]
        eng.goals["g1"] = {
            "description": "x",
            "priority": 0.5,
            "source": "input",
            "timestamp": 0.0,
            "completed": False,
            "input_meta": {},
        }
        assert eng.complete_input_goal("g1")
        assert "ghost" not in eng.input_goal_queue
        assert eng.input_goal_queue == []

    def test_current_input_goal(self):
        eng = self._make_engine()
        assert eng.current_input_goal() is None
        eng.register_input_goal("g1", "first")
        assert eng.current_input_goal() == "g1"
        eng.complete_input_goal("g1")
        assert eng.current_input_goal() is None

    def test_complete_idempotent(self):
        eng = self._make_engine()
        eng.register_input_goal("g1", "x")
        assert eng.complete_input_goal("g1")
        assert not eng.complete_input_goal("g1"), "second complete returns False"

    def test_complete_unknown_goal(self):
        eng = self._make_engine()
        assert not eng.complete_input_goal("nonexistent")

    def test_no_duplicate_enqueue(self):
        eng = self._make_engine()
        eng.register_input_goal("g1", "x")
        eng.register_input_goal("g1", "x again")
        assert eng.input_goal_queue.count("g1") == 1

    def test_prioritize_skips_completed(self):
        eng = self._make_engine()
        eng.register_input_goal("g1", "first")
        eng.register_input_goal("g2", "second")
        eng.complete_input_goal("g1")
        order = eng.prioritize()
        assert "g1" not in order
        assert order[0] == "g2"

    def test_thread_safety(self):
        eng = self._make_engine()
        errors: List[str] = []

        def register_batch(prefix: str, n: int):
            for i in range(n):
                try:
                    eng.register_input_goal(f"{prefix}_{i}", f"desc {prefix}_{i}")
                except Exception as e:
                    errors.append(str(e))

        threads = [
            threading.Thread(target=register_batch, args=("a", 50)),
            threading.Thread(target=register_batch, args=("b", 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(eng.input_goal_queue) == 100


# ---------------------------------------------------------------------------
# InputAgent goal-id dispatch (with mocked shared/bus)
# ---------------------------------------------------------------------------


class TestInputAgentGoalDispatch:
    @staticmethod
    def _make_input_agent():
        """Build a minimal InputAgent with mocked dependencies."""
        # Patch env so FIFO is enabled
        with patch.dict(
            os.environ,
            {
                "HAROMA_GOAL_FIFO_INPUT": "1",
                "HAROMA_GOAL_UNIFY_SENSOR": "1",
            },
        ):
            from agents.input_agent import InputAgent

            shared = MagicMock()
            shared.cycle_count = 1
            shared.encoder = None
            shared.goal = MagicMock()
            shared.neural_sync = MagicMock(
                return_value=MagicMock(
                    __enter__=MagicMock(),
                    __exit__=MagicMock(),
                )
            )
            bus = MagicMock()
            agent = InputAgent(shared=shared, bus=bus, tick_interval=1.0)
            agent._boot_agent = None
            return agent, shared, bus

    def test_dispatch_registers_goal(self):
        agent, shared, bus = self._make_input_agent()
        item = {"content": "hello", "source": "user", "tags": [], "depth": "normal"}
        agent._dispatch_text_item(item)
        shared.goal.register_input_goal.assert_called_once()
        call_kwargs = shared.goal.register_input_goal.call_args
        assert call_kwargs[1]["source"] == "input" or call_kwargs[0][3] == "input"

    def test_dispatch_attaches_goal_id_to_message(self):
        agent, shared, bus = self._make_input_agent()
        item = {"content": "hello", "source": "user", "tags": [], "depth": "normal"}
        agent._dispatch_text_item(item)
        sent_msg = bus.send_direct.call_args[0][1]
        assert sent_msg.content.get("input_goal_id") is not None

    def test_tick_unify_sensor_with_text(self):
        """When unify is on and both text+sensor exist, only one dispatch happens
        and the sensor data is merged into that message."""
        agent, shared, bus = self._make_input_agent()

        with agent._lock:
            agent._text_queue.append(
                {
                    "content": "hi",
                    "source": "user",
                    "tags": [],
                    "depth": "normal",
                    "channel": "chat",
                }
            )
            agent._text_queue.append(
                {
                    "content": "bye",
                    "source": "user",
                    "tags": [],
                    "depth": "normal",
                    "channel": "chat",
                }
            )
            agent._sensor_queue.append(
                {"channel": "cam", "data": {"frame": 1}, "timestamp": time.time()}
            )

        agent._tick()

        # Only one send_direct call (the unified message)
        assert bus.send_direct.call_count == 1
        sent_msg = bus.send_direct.call_args[0][1]
        assert sent_msg.content["text"] == "hi"
        assert "cam" in sent_msg.content["sensor_data"]
        assert "chat" in sent_msg.content["sensor_data"]

        # "bye" should be re-queued (priority queue for user + normal depth)
        assert len(agent._text_queue_priority) == 1
        assert agent._text_queue_priority[0]["content"] == "bye"

    def test_tick_sensor_only_registers_goal(self):
        agent, shared, bus = self._make_input_agent()

        with agent._lock:
            agent._sensor_queue.append(
                {"channel": "mic", "data": {"vol": 0.5}, "timestamp": time.time()}
            )

        agent._tick()
        assert bus.send_direct.call_count == 1
        shared.goal.register_input_goal.assert_called_once()


# ---------------------------------------------------------------------------
# GoalManager wrapper delegation
# ---------------------------------------------------------------------------


class TestGoalManagerWrappers:
    @pytest.mark.torch
    @pytest.mark.skipif(
        not torch_loads_in_subprocess(),
        reason="torch not loadable (mind.manager pulls torch-dependent engines)",
    )
    def test_register_and_complete(self):
        from mind.manager import GoalManager

        reset_shared_goal_engine_for_tests()
        try:
            gm = GoalManager()
            gm.register_input_goal("g1", "desc")
            assert gm.current_input_goal() == "g1"
            assert gm.complete_input_goal("g1")
            assert gm.current_input_goal() is None
        finally:
            reset_shared_goal_engine_for_tests()

    @pytest.mark.torch
    @pytest.mark.skipif(
        not torch_loads_in_subprocess(),
        reason="torch not loadable (mind.manager pulls torch-dependent engines)",
    )
    def test_goal_managers_share_one_engine(self):
        from mind.manager import GoalManager

        reset_shared_goal_engine_for_tests()
        try:
            a = GoalManager()
            b = GoalManager()
            assert a.engine is b.engine
        finally:
            reset_shared_goal_engine_for_tests()
