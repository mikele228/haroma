"""Gymnasium environment: one step = one synchronous ``POST /chat``.

Requires ``pip install gymnasium`` (see requirements-rl.txt).

Intended for low-step contextual bandit experiments, not high-throughput RL.
Use a single trainer thread against one Haroma process.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from bridge.haroma_client import post_chat

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError("HaromaBanditChatEnv requires gymnasium: pip install gymnasium") from e

RewardFn = Callable[[int, str], float]
MessageFn = Callable[[int, int], str]


class HaromaBanditChatEnv(gym.Env):
    """Discrete message actions, one-hot task observation, one chat turn per step.

    *tasks* — list of task strings (prompt prefixes). Rotation order or random.
    *candidate_messages* — action ``i`` sends text from ``build_message(task_idx, i)``.
    *reward_fn* — ``(task_index, assistant_response_text) -> reward`` in [0, 1] typically.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_url: str,
        candidate_messages: List[str],
        tasks: Optional[List[str]] = None,
        *,
        reward_fn: Optional[RewardFn] = None,
        build_message: Optional[MessageFn] = None,
        depth: str = "normal",
        chat_timeout_sec: float = 600.0,
        shuffle_tasks: bool = False,
    ) -> None:
        super().__init__()
        if not candidate_messages:
            raise ValueError("candidate_messages must be non-empty")
        self._base = base_url.rstrip("/")
        self._candidates = list(candidate_messages)
        self._tasks = list(tasks) if tasks else [""]
        self._reward_fn = reward_fn or self._default_reward
        self._build: MessageFn = build_message or self._default_build_message
        # Server maps chat depth=fast to normal; align client to avoid confusion.
        d = str(depth or "normal").lower()
        self._depth = "normal" if d == "fast" else d
        self._chat_timeout = float(chat_timeout_sec)
        self._shuffle = bool(shuffle_tasks)
        self._task_idx = 0

        n_tasks = max(1, len(self._tasks))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_tasks,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self._candidates))

    def _default_build_message(self, task_idx: int, action_idx: int) -> str:
        prefix = (self._tasks[task_idx % len(self._tasks)] or "").strip()
        body = (self._candidates[action_idx] or "").strip()
        if prefix and body:
            return f"{prefix} {body}"
        return prefix or body

    @staticmethod
    def _default_reward(task_idx: int, response: str) -> float:
        text = (response or "").lower()
        return 1.0 if "yes" in text or "ok" in text else 0.0

    def _observation(self) -> np.ndarray:
        n = self.observation_space.shape[0]
        obs = np.zeros((n,), dtype=np.float32)
        obs[self._task_idx % n] = 1.0
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if self._shuffle:
            assert self.np_random is not None
            self._task_idx = int(self.np_random.integers(0, len(self._tasks)))
        else:
            self._task_idx = (self._task_idx + 1) % len(self._tasks)
        return self._observation(), {}

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        idx = int(action)
        if idx < 0 or idx >= len(self._candidates):
            raise ValueError(f"action {idx} out of range for Discrete({len(self._candidates)})")

        message = self._build(self._task_idx, idx)
        data, code = post_chat(
            self._base,
            message,
            depth=self._depth,
            async_=False,
            timeout=self._chat_timeout,
        )
        response_text = ""
        if isinstance(data, dict):
            response_text = str(data.get("response") or "")
        info: Dict[str, Any] = {
            "http_status": code,
            "raw": data if isinstance(data, dict) else {},
        }
        if code != 200:
            reward = 0.0
            info["error"] = data.get("error") if isinstance(data, dict) else "non_200"
        else:
            try:
                reward = float(self._reward_fn(self._task_idx, response_text))
            except Exception as exc:
                reward = 0.0
                info["reward_error"] = str(exc)
            reward = max(0.0, min(1.0, reward))

        obs = self._observation()
        terminated = True
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        return None


__all__ = ["HaromaBanditChatEnv"]
