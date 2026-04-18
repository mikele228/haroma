"""Smoke: architecture doc module is importable and describes dual entry points."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mind.cognitive_entrypoints as ce


def test_module_documents_controller_and_persona():
    doc = (ce.__doc__ or "").lower()
    assert "elarioncontroller" in doc
    assert "persona" in doc
    assert "run_llm_context_reasoning_phase" in doc
