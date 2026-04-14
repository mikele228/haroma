"""
Optional ROS 2 adapter sketch (does **not** require ``rclpy`` at import time).

Subclass ``rclpy.node.Node``, subscribe to your command topic (JSON or custom),
map payloads to the same shape as ``build_executor_command_batch``, run your
real controller, then POST feedback via :func:`bridge.haroma_client.post_robot_bridge_feedback`.

This module only provides :func:`try_import_rclpy` and a small :class:`StubExecutorNode`
that does not create an ``rclpy`` node — use it as a logic helper inside *your* node.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from bridge.stub_executor import feedback_block_from_results, simulate_command_results


def try_import_rclpy():
    """Return ``(rclpy, Node)`` or ``(None, None)`` if ROS 2 Python is not installed."""
    try:
        import rclpy
        from rclpy.node import Node

        return rclpy, Node
    except ImportError:
        return None, None


class StubExecutorNode:
    """Stand-in executor logic (no ROS). Wire this from your ``rclpy`` node callbacks."""

    def __init__(self, haroma_base_url: str):
        self.haroma_base_url = haroma_base_url.rstrip("/")

    def on_batch_json(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        results = simulate_command_results(batch)
        return feedback_block_from_results(batch, results)

    def send_feedback_to_haroma(self, feedback_block: Dict[str, Any]) -> Tuple[Any, int]:
        from bridge.haroma_client import post_robot_bridge_feedback

        return post_robot_bridge_feedback(self.haroma_base_url, feedback_block)
