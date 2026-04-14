"""Action loop: prefer ``llm_context`` answer when the LLM returns grounded text."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ActionLoop import ActionGenerator


def _minimal_ep_payload(*, identity: dict) -> dict:
    return {
        "affect": {
            "dominant_emotion": "neutral",
            "intensity": 0.1,
            "valence": 0.0,
            "arousal": 0.0,
        },
        "active_goals": [],
        "curiosity": {
            "novelty_score": 0.0,
            "uncertainty_score": 0.0,
            "questions": [],
        },
        "narrative_context": "",
        "identity": identity,
        "drives": {},
        "dominant_drive": "",
        "perception": {},
    }


def test_name_answer_from_llm_context_not_question_typing():
    """Identity replies come from packed LLM context when it returns a grounded answer."""
    gen = ActionGenerator(composer=None)
    ep = _minimal_ep_payload(
        identity={"essence_name": "HaromaVX", "vessel": "Elarion"},
    )
    ep["llm_context"] = {
        "answer": "I'm HaromaVX; my vessel is Elarion.",
        "confidence": 0.95,
        "requires_confirmation": False,
        "cited_memories": [0, 1, 2, 3],
    }
    action = gen.generate(
        ep,
        workspace_contents=[],
        strategy_hint="inform",
        working_memory_context="",
        conversation_context="",
        is_in_conversation=True,
        topic="",
        last_input_content="What is your name?",
        topic_shifted=False,
        knowledge_summary={"entity_count": 0, "relation_count": 0, "top_entities": []},
        reasoning_result={},
        nlu_result={"intent": "utterance", "entities": [], "sentiment": {}},
        interlocutor={},
        counterfactual_result={},
        novelty_bias=0.0,
        personality={"openness": 0.5, "agreeableness": 0.5},
        utterance_style="conversational",
    )
    text = (action.get("text") or "").strip()
    assert text, f"expected non-empty text, got {action!r}"
    assert "HaromaVX" in text
    assert "Elarion" in text
    assert action.get("strategy") == "llm_context"


def test_no_grounding_stays_empty():
    gen = ActionGenerator(composer=None)
    ep = _minimal_ep_payload(identity={})
    action = gen.generate(
        ep,
        workspace_contents=[],
        strategy_hint="inform",
        working_memory_context="",
        conversation_context="",
        is_in_conversation=True,
        topic="",
        last_input_content="What is your name?",
        topic_shifted=False,
        knowledge_summary={"entity_count": 0, "relation_count": 0, "top_entities": []},
        reasoning_result={},
        nlu_result={"intent": "utterance", "entities": [], "sentiment": {}},
        interlocutor={},
        counterfactual_result={},
        novelty_bias=0.0,
        personality={"openness": 0.5, "agreeableness": 0.5},
        utterance_style="conversational",
    )
    assert not (action.get("text") or "").strip()
