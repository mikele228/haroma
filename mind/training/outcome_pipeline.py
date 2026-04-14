"""Fan-out for (prompt, response, reward) training signals — keeps LLMBackend thinner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from engine.LLMBackend import LLMBackend

from mind.environment_prompt_budgets import RECORD_OUTCOME_ENV_SUMMARY_MAX_CHARS


class OutcomeRecordingPipeline:
    """Writes torch reward, finetune collector, optional VW, optional RLlib JSONL."""

    def __init__(self, backend: "LLMBackend") -> None:
        self._b = backend

    def record(
        self,
        prompt: str,
        response: str,
        outcome_score: float,
        *,
        alignment_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        reward = max(0.0, min(1.0, outcome_score))
        env_summary = ""
        meta = dict(alignment_metadata) if alignment_metadata else None
        if meta is not None:
            meta.setdefault("alignment_training", True)
            ae = meta.get("agent_environment")
            if isinstance(ae, dict) and ae:
                try:
                    from mind.environment_context import environment_summary_for_prompt

                    env_summary = environment_summary_for_prompt(
                        ae,
                        max_chars=RECORD_OUTCOME_ENV_SUMMARY_MAX_CHARS,
                    )
                except Exception:
                    env_summary = ""
            if not env_summary and meta.get("environment_summary"):
                cap = RECORD_OUTCOME_ENV_SUMMARY_MAX_CHARS
                env_summary = str(meta["environment_summary"])[:cap]

        lock = getattr(self._b, "_env_context_lock", None)
        if lock is not None:
            with lock:
                if env_summary:
                    self._b._last_env_summary = env_summary
                    self._b._last_env_summary_ts = time.time()
        else:
            if env_summary:
                self._b._last_env_summary = env_summary
                self._b._last_env_summary_ts = time.time()

        self._b.reward_model.record(prompt, response, reward)
        self._b.finetune_collector.record(prompt, response, reward, metadata=meta)
        vw = getattr(self._b, "_vw_trainer", None)
        if vw is not None:
            vw.record(
                prompt,
                response,
                reward,
                environment_summary=env_summary,
            )
        rllib = getattr(self._b, "_rllib_logger", None)
        if rllib is not None:
            rllib.record(prompt, response, reward, metadata=meta)
