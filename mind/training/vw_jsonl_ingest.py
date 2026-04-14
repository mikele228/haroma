"""Optional: feed bandit JSONL into :class:`VowpalWabbitRewardTrainer` during background training.

Use when an external process appends ``bandit_step`` lines to a file and you want
VW online learning without duplicating ``record_outcome`` (use a **dedicated** ingest
file, not the same stream you also log from RLlibTransitionLogger, to avoid double
counting).

Env:

* ``HAROMA_VW_BANDIT_INGEST_PATH`` — path to JSONL (``type: bandit_step`` lines).
* ``HAROMA_VW_BANDIT_INGEST_MAX_LINES`` — max **bandit** rows per call (default ``64``).
* ``HAROMA_VW_BANDIT_INGEST_OFFSET_PATH`` — stores one integer: next **file line index** (0-based) to read (default ``data/rllib/vw_ingest_line.offset`` under project root).
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_OFFSET = os.path.join(_PROJECT_ROOT, "data", "rllib", "vw_ingest_line.offset")


def _read_offset(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return max(0, int(f.read().strip() or "0"))
    except (OSError, ValueError):
        return 0


def _write_offset(path: str, line_idx: int) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(line_idx))
    os.replace(tmp, path)


def ingest_bandit_jsonl_into_vw(vw_trainer: Any, *, max_lines: Optional[int] = None) -> int:
    """Append up to *max_lines* new bandit rows into *vw_trainer*'s pending queue.

    Returns number of bandit rows ingested (0 if disabled or empty).
    """
    if vw_trainer is None or not getattr(vw_trainer, "available", False):
        return 0
    path = str(os.environ.get("HAROMA_VW_BANDIT_INGEST_PATH", "") or "").strip()
    if not path or not os.path.isfile(path):
        return 0
    cap = max_lines
    if cap is None:
        try:
            cap = max(1, int(os.environ.get("HAROMA_VW_BANDIT_INGEST_MAX_LINES", "64")))
        except (TypeError, ValueError):
            cap = 64
    offset_path = str(
        os.environ.get("HAROMA_VW_BANDIT_INGEST_OFFSET_PATH", "") or _DEFAULT_OFFSET
    )
    start = _read_offset(offset_path)
    ingested = 0
    next_start = start

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for idx, raw in enumerate(f):
                if idx < start:
                    continue
                line = raw.strip()
                if not line:
                    next_start = idx + 1
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    next_start = idx + 1
                    continue
                if not isinstance(rec, dict) or rec.get("type") != "bandit_step":
                    next_start = idx + 1
                    continue
                if ingested >= cap:
                    next_start = idx
                    break
                obs = str(rec.get("obs") or "")
                action = str(rec.get("action") or "")
                try:
                    rw = float(rec.get("reward", 0.5))
                except (TypeError, ValueError):
                    rw = 0.5
                info = rec.get("info") if isinstance(rec.get("info"), dict) else {}
                env_summary = str(info.get("environment_summary") or "") if info else ""
                vw_trainer.record(
                    obs,
                    action,
                    rw,
                    environment_summary=env_summary,
                )
                ingested += 1
                next_start = idx + 1
        _write_offset(offset_path, next_start)
    except OSError as exc:
        print(f"[vw_jsonl_ingest] read failed: {exc}", flush=True)
        return 0
    return ingested


__all__ = ["ingest_bandit_jsonl_into_vw"]
