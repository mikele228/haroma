"""Suppress conversational replies that only repeat the user's utterance."""

from core.ActionLoop import (
    ActionCandidate,
    ActionGenerator,
    _is_echo_of_last_input,
)


def test_is_echo_exact_and_near_duplicate():
    assert _is_echo_of_last_input("How are you doing?", "how are you doing?")
    assert _is_echo_of_last_input("why repeat?", "Why repeat?")
    assert not _is_echo_of_last_input("E", "A,B,C,D what letter is next?")
    long_q = "A,B,C,D what letter is next?"
    assert _is_echo_of_last_input(long_q, long_q)


def test_assemble_conversational_skips_echo():
    gen = ActionGenerator(composer=None)
    winner = ActionCandidate(
        "advance_goal",
        ["How are you doing?", "A real step forward."],
    )
    ctx = {
        "utterance_style": "conversational",
        "last_input": "How are you doing?",
    }
    out = gen._assemble_text_conversational(winner, ctx)
    assert out == "A real step forward."


def test_winner_has_substance_false_when_only_echo():
    winner = ActionCandidate("advance_goal", ["why repeat?"])
    ctx = {
        "utterance_style": "conversational",
        "last_input": "why repeat?",
    }
    assert not ActionGenerator._winner_has_substance(winner, ctx)


def test_advance_goal_omits_goal_label_when_echo_of_user():
    gen = ActionGenerator(composer=None)
    ctx = {
        "goals": [{"description": "how are you doing?", "goal_id": "g1"}],
        "reasoning": {},
        "counterfactual": {},
        "utterance_style": "conversational",
        "last_input": "How are you doing?",
    }
    c = gen._candidate_advance_goal(ctx)
    assert "how are you doing" not in " ".join(c.content_elements).lower()


def test_inquire_skips_question_echo():
    gen = ActionGenerator(composer=None)
    ctx = {
        "curiosity": {"questions": ["Why repeat?"]},
        "knowledge": {"top_entities": ["fallback_entity"]},
        "interlocutor": {"inferred_beliefs": []},
        "utterance_style": "conversational",
        "last_input": "why repeat?",
    }
    c = gen._candidate_inquire(ctx)
    assert "Why repeat?" not in c.content_elements
    assert "fallback_entity" not in c.content_elements


def test_inquire_conversational_no_top_entity_fallback():
    gen = ActionGenerator(composer=None)
    ctx = {
        "curiosity": {"questions": []},
        "knowledge": {"top_entities": ["The Clearing"]},
        "interlocutor": {},
        "utterance_style": "conversational",
        "last_input": "What is physics?",
        "nlu": {"entities": []},
    }
    c = gen._candidate_inquire(ctx)
    assert c.content_elements == []


def test_inform_conversational_inference_requires_grounding():
    gen = ActionGenerator(composer=None)
    inf = [{"subject": "The Clearing", "predicate": "near", "object": "Forest"}]
    ctx_unrelated = {
        "utterance_style": "conversational",
        "last_input": "Hello",
        "interlocutor": {},
        "nlu": {"entities": []},
        "knowledge": {"top_entities": []},
        "reasoning": {"inferences": inf},
    }
    c1 = gen._candidate_inform(ctx_unrelated)
    assert not any("Clearing" in e for e in c1.content_elements)
    ctx_related = {
        **ctx_unrelated,
        "last_input": "Is the clearing near the forest?",
    }
    c2 = gen._candidate_inform(ctx_related)
    assert any("Clearing" in e for e in c2.content_elements)
