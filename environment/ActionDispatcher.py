"""
ActionDispatcher — Closes the perception-action-outcome loop (Sentience Upgrade 6).

Routes cognitive action decisions to the appropriate environment or
actuator, collects observations, and feeds them back into the sensor
pipeline so the next cognitive cycle perceives consequences of its
own actions.

Action types:
  dialogue   — text response (existing path, pass-through)
  explore    — move to a new location and observe
  observe    — detailed look at a specific target
  manipulate — interact with an object (take, use, put)
  navigate   — pathfind to a named location
  adjust_sensor — reconfigure a sensor parameter

For TextEnvironment actions are translated into env commands.
For hardware actuators, commands are dispatched via HTTP POST to
configurable endpoints.
"""

from __future__ import annotations

import time
import requests
from typing import Dict, Any, List, Optional


PHYSICAL_ACTION_TYPES = frozenset(
    {
        "explore",
        "observe",
        "manipulate",
        "navigate",
        "adjust_sensor",
    }
)


class ActionDispatcher:
    """Routes action decisions to environments and actuators."""

    def __init__(self, actuator_endpoints: Optional[Dict[str, str]] = None):
        self._actuator_endpoints = actuator_endpoints or {}
        self._dispatch_count = 0
        self._error_count = 0
        self._last_observation: Optional[Dict[str, Any]] = None

    @staticmethod
    def is_physical_action(action: Dict[str, Any]) -> bool:
        return action.get("action_type", "") in PHYSICAL_ACTION_TYPES

    def dispatch(
        self,
        action: Dict[str, Any],
        environment=None,
    ) -> Dict[str, Any]:
        """Route an action to the right target and return an observation.

        Parameters
        ----------
        action : dict
            Must contain ``action_type``.  May contain ``target``,
            ``params``, ``text``, etc.
        environment : TextEnvironment or None
            The simulated world to act upon.

        Returns
        -------
        dict with keys: success, description, state_changes, timestamp
        """
        self._dispatch_count += 1
        action_type = action.get("action_type", "dialogue")

        if action_type == "dialogue":
            return self._passthrough(action)

        if environment is not None and hasattr(environment, "execute_action"):
            obs = self._dispatch_to_env(action_type, action, environment)
        elif action_type in self._actuator_endpoints:
            obs = self._dispatch_to_hardware(action_type, action)
        else:
            obs = {
                "success": False,
                "description": f"No handler for action_type={action_type}",
                "state_changes": [],
            }
            self._error_count += 1

        obs["timestamp"] = time.time()
        obs["action_type"] = action_type
        self._last_observation = obs
        return obs

    def _passthrough(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "description": action.get("text", ""),
            "state_changes": [],
            "action_type": "dialogue",
            "timestamp": time.time(),
        }

    def _dispatch_to_env(
        self,
        action_type: str,
        action: Dict[str, Any],
        environment,
    ) -> Dict[str, Any]:
        target = action.get("target", "")
        params = action.get("params", {})
        try:
            return environment.execute_action(action_type, target, params)
        except Exception as exc:
            self._error_count += 1
            return {
                "success": False,
                "description": f"Environment error: {exc}",
                "state_changes": [],
            }

    def _dispatch_to_hardware(
        self,
        action_type: str,
        action: Dict[str, Any],
    ) -> Dict[str, Any]:
        url = self._actuator_endpoints[action_type]
        payload = {
            "action_type": action_type,
            "target": action.get("target", ""),
            "params": action.get("params", {}),
        }
        try:
            resp = requests.post(url, json=payload, timeout=5)
            data = resp.json()
            return {
                "success": data.get("success", resp.ok),
                "description": data.get("description", resp.text[:200]),
                "state_changes": data.get("state_changes", []),
            }
        except Exception as exc:
            self._error_count += 1
            return {
                "success": False,
                "description": f"Actuator error: {exc}",
                "state_changes": [],
            }

    def stats(self) -> Dict[str, Any]:
        return {
            "dispatch_count": self._dispatch_count,
            "error_count": self._error_count,
            "actuator_endpoints": list(self._actuator_endpoints.keys()),
            "last_observation": self._last_observation,
        }
