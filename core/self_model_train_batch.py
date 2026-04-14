"""Foreground snapshot consumed by background self-model training (multi-agent path)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class SelfModelTrainBatch:
    """Same triple as ``mind.control`` uses for ``SelfModel.train_step``."""

    embedding: Any
    prev_state: Dict[str, Any]
    actual_state: Dict[str, Any]
