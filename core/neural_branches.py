"""Neural RW branch ids — align with :func:`core.training_surface.build_background_train_map` order.

Each branch has its own :class:`core.concurrency.NeuralRWLock` when
``HAROMA_NEURAL_LOCK_MODE=per_branch`` (default): inference holds **read** on the
branches it touches; background training holds **write** on one module at a time.
That allows e.g. encoder inference while backbone trains, when weights are disjoint.

Set ``HAROMA_NEURAL_LOCK_MODE=global`` for legacy single-lock behavior.
"""

from __future__ import annotations

import os
from typing import Sequence, Tuple

# Must match the keys in build_background_train_map (same order helps diagnostics).
NEURAL_TRAIN_BRANCH_NAMES: Tuple[str, ...] = (
    "encoder",
    "backbone",
    "attention",
    "process_gate",
    "self_model",
    "appraisal",
    "modulation",
    "goal_synth",
    "imagination",
    "metacog",
    "composer",
    "generative",
    "counterfactual",
    "grounder",
    "mental_sim",
    "arch_search",
    "llm_reward",
)


def normalize_neural_branches(branches: Sequence[str]) -> list[str]:
    """Return sorted unique branch names (acquire order for multi-branch read locks)."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in branches:
        b = str(raw).strip()
        if not b or b in seen:
            continue
        seen.add(b)
        out.append(b)
    out.sort()
    return out


def neural_lock_mode() -> str:
    """``global`` (single lock) or ``per_branch`` (default)."""
    raw = (os.environ.get("HAROMA_NEURAL_LOCK_MODE") or "").strip().lower()
    if raw in ("global", "legacy", "single"):
        return "global"
    return "per_branch"
