"""Per-persona semantic encoder resolution (no torch required)."""

import pytest

from agents.shared_resources import SharedResources


def test_resolve_semantic_model_id_from_agents_json(monkeypatch):
    monkeypatch.delenv("HAROMA_SEMANTIC_ENCODER", raising=False)
    monkeypatch.delenv("HAROMA_SEMANTIC_ENCODER_ANALYST", raising=False)
    sr = SharedResources()
    sr.agent_config = {
        "initial_personas": [
            {"id": "analyst", "semantic_encoder": "org/custom-encoder"},
        ]
    }
    assert sr.resolve_semantic_model_id("analyst") == "org/custom-encoder"


def test_resolve_semantic_model_id_env_suffix(monkeypatch):
    monkeypatch.delenv("HAROMA_SEMANTIC_ENCODER", raising=False)
    monkeypatch.setenv("HAROMA_SEMANTIC_ENCODER_ANALYST", "sentence-transformers/all-mpnet-base-v2")
    sr = SharedResources()
    sr.agent_config = {"initial_personas": []}
    assert sr.resolve_semantic_model_id("analyst") == "sentence-transformers/all-mpnet-base-v2"


def test_resolve_semantic_model_id_global_fallback(monkeypatch):
    monkeypatch.setenv("HAROMA_SEMANTIC_ENCODER", "sentence-transformers/all-MiniLM-L6-v2")
    sr = SharedResources()
    sr.agent_config = {"initial_personas": []}
    assert sr.resolve_semantic_model_id("unknown") == "sentence-transformers/all-MiniLM-L6-v2"
