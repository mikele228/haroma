"""Optional external training bridges (Vowpal Wabbit, RLlib-oriented logging).

Gymnasium / offline JSONL helpers live in :mod:`mind.training.gymnasium_offline_bridge`
and :mod:`mind.training.haroma_gym_env` (requires ``pip install -r requirements-rl.txt``).
"""

from mind.training import rllib_jsonl_offline
from mind.training.vw_rl_bridge import (
    RLlibTransitionLogger,
    VowpalWabbitRewardTrainer,
    composite_trained_scores,
    load_rllib_score_callable,
)

__all__ = [
    "RLlibTransitionLogger",
    "VowpalWabbitRewardTrainer",
    "composite_trained_scores",
    "load_rllib_score_callable",
    "rllib_jsonl_offline",
]
