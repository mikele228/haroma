"""Bandit JSONL ingestion into VW pending queue (background training hook)."""

from __future__ import annotations

import json
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def test_ingest_bandit_jsonl_advances_offset(tmp_path, monkeypatch):
    from mind.training.vw_jsonl_ingest import ingest_bandit_jsonl_into_vw

    jl = tmp_path / "b.jsonl"
    off = tmp_path / "off.txt"
    jl.write_text(
        "\n".join(
            [
                json.dumps({"type": "bandit_step", "obs": "a", "action": "b", "reward": 0.3}),
                json.dumps({"type": "bandit_step", "obs": "c", "action": "d", "reward": 0.8}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HAROMA_VW_BANDIT_INGEST_PATH", str(jl))
    monkeypatch.setenv("HAROMA_VW_BANDIT_INGEST_OFFSET_PATH", str(off))
    monkeypatch.setenv("HAROMA_VW_BANDIT_INGEST_MAX_LINES", "10")

    recorded = []

    class _VW:
        available = True

        def record(self, p, r, rw, *, environment_summary=""):
            recorded.append((p, r, rw, environment_summary))

    n = ingest_bandit_jsonl_into_vw(_VW())
    assert n == 2
    assert len(recorded) == 2
    assert recorded[0][0] == "a"
    assert recorded[1][1] == "d"
    assert off.is_file()
    # Second pass: nothing new
    n2 = ingest_bandit_jsonl_into_vw(_VW())
    assert n2 == 0


def test_ingest_skips_when_no_path(monkeypatch):
    from mind.training.vw_jsonl_ingest import ingest_bandit_jsonl_into_vw

    monkeypatch.delenv("HAROMA_VW_BANDIT_INGEST_PATH", raising=False)
    assert ingest_bandit_jsonl_into_vw(None) == 0
