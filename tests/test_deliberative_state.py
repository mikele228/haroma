"""Deliberative mode: state snapshot, extended schema, candidate_actions parsing."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.LLMContextReasoner import (
    _JSON_SCHEMA_DELIBERATIVE_EXT,
    _MAX_CANDIDATE_ACTIONS,
)
from mind.cognitive_contracts import build_messages, parse_response
from engine.DoctrineCortex import DoctrineCortex
from mind.cycle_flow import build_trueself_state_snapshot, serialize_state_snapshot


# -- DoctrineCortex.summarize_for_prompt ---------------------------------


def test_doctrine_summarize_for_prompt_includes_values():
    dc = DoctrineCortex()
    dc.reinforce_value("honesty", 0.9)
    dc.reinforce_value("courage", 0.6)
    dc.define_doctrine("core_ethics", {"honesty": 0.8, "compassion": 0.7}, "Core ethics")
    p = dc.summarize_for_prompt()
    assert "honesty" in p["value_keys"]
    assert p["values"]["honesty"] == 0.9
    assert len(p["active_doctrines"]) == 1
    assert p["active_doctrines"][0]["name"] == "core_ethics"


def test_doctrine_summarize_caps_values():
    dc = DoctrineCortex()
    for i in range(30):
        dc.reinforce_value(f"val_{i}", round(i / 30, 3))
    p = dc.summarize_for_prompt()
    assert len(p["values"]) <= dc._MAX_PROMPT_VALUES


# -- InputAgent.peek_sensor_queue ----------------------------------------


def test_peek_sensor_queue_non_destructive():
    from agents.input_agent import InputAgent

    ia = InputAgent.__new__(InputAgent)
    import threading
    from collections import deque

    ia._lock = threading.Lock()
    ia._sensor_queue = deque(maxlen=128)
    ia._sensor_queue.append({"channel": "cam", "data": {"img": "x"}, "timestamp": 1.0})
    ia._sensor_queue.append({"channel": "mic", "data": {"wav": "y"}, "timestamp": 2.0})

    snap = ia.peek_sensor_queue()
    assert len(snap) == 2
    assert snap[0]["channel"] == "mic"  # newest first
    assert len(ia._sensor_queue) == 2  # not drained


# -- build_trueself_state_snapshot ---------------------------------------


def test_build_state_snapshot_shape():
    snap = build_trueself_state_snapshot(
        sensors=[{"channel": "cam", "data": {}}],
        identity_summary={"essence_name": "Elarion", "vessel": "test"},
        personality_summary={"openness": 0.5, "conscientiousness": 0.7},
        active_goals=[{"goal_id": "g1", "description": "explore", "priority": 0.8}],
        law_summary={"ids": ["law_1"]},
        value_summary={"values": {"honesty": 0.9}, "value_keys": ["honesty"]},
        drives={"understanding": 0.5},
        affect={"dominant_emotion": "joy", "intensity": 0.7},
    )
    assert snap["sensors"] == [{"channel": "cam", "data": {}}]
    assert snap["identity"]["essence_name"] == "Elarion"
    assert snap["goals"][0]["goal_id"] == "g1"
    assert snap["values"]["values"]["honesty"] == 0.9
    assert snap["affect"]["dominant_emotion"] == "joy"


def test_serialize_state_snapshot_caps_length():
    snap = build_trueself_state_snapshot(
        identity_summary={"big_field": "x" * 200_000},
    )
    s = serialize_state_snapshot(snap)
    assert len(s) <= 100_001  # _MAX_STATE_JSON_CHARS + "..."


# -- build_messages with deliberative=True --------------------------------


def test_build_messages_deliberative_includes_state_and_schema():
    state = json.dumps({"values": {"honesty": 0.9}})
    msgs = build_messages(
        user_text="What should I do?",
        recalled_memories=[],
        identity_summary={"essence_name": "E", "vessel": "V"},
        personality_summary={"openness": 0.5},
        active_goals=[],
        law_summary={},
        value_summary={"value_keys": ["honesty"]},
        deliberative=True,
        agent_state_json=state,
    )
    system = msgs[0]["content"]
    user = msgs[1]["content"]
    assert "candidate_actions" in system
    assert "value_impact" in system
    assert "DELIBERATIVE MODE" in system
    assert "[AGENT STATE JSON]" in user
    assert '"honesty"' in user


def test_build_messages_non_deliberative_unchanged():
    msgs = build_messages(
        user_text="Hello",
        recalled_memories=[],
        identity_summary={"essence_name": "E"},
        personality_summary={},
        active_goals=[],
        law_summary={},
        value_summary={},
    )
    system = msgs[0]["content"]
    user = msgs[1]["content"]
    assert "candidate_actions" not in system
    assert "[AGENT STATE JSON]" not in user


# -- parse_response with candidate_actions --------------------------------


def test_parse_response_extracts_candidate_actions():
    raw = json.dumps(
        {
            "answer": "Consider these options.",
            "confidence": 0.8,
            "reasoning_steps": ["step1"],
            "inferences": [],
            "cited_memories": [],
            "requires_confirmation": False,
            "candidate_actions": [
                {
                    "id": "a1",
                    "label": "Tell the truth",
                    "strategy": "inform",
                    "rationale": "Honesty is valued.",
                    "value_impact": {"honesty": 0.5, "courage": -0.1},
                    "confidence": 0.9,
                },
                {
                    "id": "a2",
                    "label": "Stay silent",
                    "strategy": "reflect",
                    "rationale": "Avoid conflict.",
                    "value_impact": {"honesty": -0.3},
                    "confidence": 0.6,
                },
            ],
        }
    )
    result = parse_response(raw)
    assert result.has_answer
    assert len(result.candidate_actions) == 2
    a1 = result.candidate_actions[0]
    assert a1["id"] == "a1"
    assert a1["value_impact"]["honesty"] == 0.5
    assert a1["value_impact"]["courage"] == -0.1
    a2 = result.candidate_actions[1]
    assert a2["strategy"] == "reflect"


def test_parse_response_caps_candidate_actions():
    actions = [
        {
            "id": f"a{i}",
            "label": f"action_{i}",
            "strategy": "inform",
            "rationale": "x",
            "value_impact": {},
            "confidence": 0.5,
        }
        for i in range(20)
    ]
    raw = json.dumps(
        {
            "answer": "ok",
            "confidence": 0.5,
            "candidate_actions": actions,
        }
    )
    result = parse_response(raw)
    assert len(result.candidate_actions) <= _MAX_CANDIDATE_ACTIONS


def test_parse_response_no_candidate_actions_when_absent():
    raw = json.dumps(
        {
            "answer": "hello",
            "confidence": 0.5,
        }
    )
    result = parse_response(raw)
    assert result.candidate_actions == []


def test_to_dict_includes_candidate_actions_when_present():
    raw = json.dumps(
        {
            "answer": "ok",
            "confidence": 0.7,
            "candidate_actions": [
                {
                    "id": "a1",
                    "label": "act",
                    "strategy": "inform",
                    "rationale": "r",
                    "value_impact": {"v": 0.1},
                    "confidence": 0.8,
                },
            ],
        }
    )
    result = parse_response(raw)
    d = result.to_dict()
    assert "candidate_actions" in d
    assert d["candidate_actions"][0]["value_impact"]["v"] == 0.1


def test_to_dict_includes_empty_candidate_actions_list():
    raw = json.dumps({"answer": "hi", "confidence": 0.5})
    result = parse_response(raw)
    d = result.to_dict()
    assert "candidate_actions" in d
    assert d["candidate_actions"] == []


def test_value_impact_clamps():
    raw = json.dumps(
        {
            "answer": "x",
            "confidence": 0.5,
            "candidate_actions": [
                {
                    "id": "a",
                    "label": "a",
                    "strategy": "s",
                    "value_impact": {"v": 5.0, "w": -3.0},
                    "confidence": 0.5,
                },
            ],
        }
    )
    result = parse_response(raw)
    vi = result.candidate_actions[0]["value_impact"]
    assert vi["v"] == 1.0
    assert vi["w"] == -1.0


def test_parse_response_goal_and_belief_impact():
    raw = json.dumps(
        {
            "answer": "ok",
            "confidence": 0.6,
            "candidate_actions": [
                {
                    "id": "x",
                    "label": "L",
                    "strategy": "inform",
                    "value_impact": {"a": 0.1},
                    "goal_impact": {"g1": 0.2},
                    "belief_impact": [
                        {"proposition": "Sky is blue", "confidence_delta": 0.3},
                    ],
                    "confidence": 0.7,
                },
            ],
        }
    )
    r = parse_response(raw)
    ca = r.candidate_actions[0]
    assert ca["goal_impact"]["g1"] == 0.2
    assert ca["belief_impact"][0]["proposition"] == "Sky is blue"
    assert ca["belief_impact"][0]["confidence_delta"] == 0.3


def test_score_deliberative_with_empty_value_dict():
    from mind.deliberative_choice import score_deliberative_candidate

    ca = {
        "id": "a",
        "label": "L",
        "strategy": "inform",
        "value_impact": {"honesty": 0.3},
        "goal_impact": {},
        "belief_impact": [],
        "confidence": 0.5,
    }
    s0 = score_deliberative_candidate(ca, current_values={})
    s1 = score_deliberative_candidate(ca, current_values={"honesty": 0.9})
    assert s0 != 0.0
    assert s1 > s0


def test_select_deliberative_candidate_prefers_value_alignment():
    from mind.deliberative_choice import select_deliberative_candidate

    cands = [
        {
            "id": "low",
            "label": "L1",
            "strategy": "reflect",
            "value_impact": {"honesty": -0.2},
            "goal_impact": {},
            "belief_impact": [],
            "confidence": 0.5,
        },
        {
            "id": "high",
            "label": "L2",
            "strategy": "inform",
            "value_impact": {"honesty": 0.4},
            "goal_impact": {},
            "belief_impact": [],
            "confidence": 0.6,
        },
    ]
    winner, board = select_deliberative_candidate(
        cands,
        current_values={"honesty": 0.9},
        active_goals=[],
        dominant_drive="understanding",
        drive_levels={"understanding": 0.8},
    )
    assert winner is not None
    assert winner["id"] == "high"
    assert board[0]["score"] >= board[-1]["score"]


def test_goal_engine_bump_priority():
    from core.Goal import GoalEngine, reset_shared_goal_engine_for_tests

    reset_shared_goal_engine_for_tests()
    ge = GoalEngine()
    ge.register_goal("g1", "test", 0.4)
    assert ge.bump_goal_priority("g1", 0.2) is True
    assert abs(ge.goals["g1"]["priority"] - 0.6) < 0.01
    assert ge.bump_goal_priority("missing", 0.1) is False
    reset_shared_goal_engine_for_tests()


# -- Four-pillar scoring: law_risk penalty ---------------------------------


def test_score_penalizes_law_risk():
    from mind.deliberative_choice import score_deliberative_candidate

    base = {
        "id": "a",
        "label": "A",
        "strategy": "inform",
        "value_impact": {"honesty": 0.2},
        "goal_impact": {},
        "belief_impact": [],
        "confidence": 0.5,
    }
    safe = {**base, "law_risk": 0.0, "emotion_alignment": 0.0}
    risky = {**base, "law_risk": 0.8, "emotion_alignment": 0.0}
    s_safe = score_deliberative_candidate(safe, current_values={"honesty": 0.7})
    s_risky = score_deliberative_candidate(risky, current_values={"honesty": 0.7})
    assert s_safe > s_risky, "law_risk should reduce score"


def test_law_violation_veto():
    from mind.deliberative_choice import select_deliberative_candidate

    good = {
        "id": "good",
        "label": "Good",
        "strategy": "reflect",
        "value_impact": {"honesty": 0.1},
        "goal_impact": {},
        "belief_impact": [],
        "confidence": 0.5,
        "law_risk": 0.0,
        "emotion_alignment": 0.0,
    }
    bad = {
        "id": "bad",
        "label": "Bad",
        "strategy": "inform",
        "value_impact": {"honesty": 0.4},
        "goal_impact": {},
        "belief_impact": [],
        "confidence": 0.8,
        "law_risk": 0.9,
        "emotion_alignment": 0.0,
    }
    violations = [
        {"law_id": "law1", "matched_tags": ["inform"], "severity": 1.5},
    ]
    winner, board = select_deliberative_candidate(
        [good, bad],
        current_values={"honesty": 0.9},
        active_goals=[],
        law_violations=violations,
    )
    assert winner is not None
    assert winner["id"] == "good", "candidate whose strategy matches a violated tag should lose"


# -- Four-pillar scoring: emotion alignment --------------------------------


def test_emotion_alignment_boosts_score():
    from mind.deliberative_choice import score_deliberative_candidate

    base = {
        "id": "a",
        "label": "A",
        "strategy": "empathize",
        "value_impact": {},
        "goal_impact": {},
        "belief_impact": [],
        "confidence": 0.5,
        "law_risk": 0.0,
    }
    aligned = {**base, "emotion_alignment": 0.8}
    neutral = {**base, "emotion_alignment": 0.0}
    emo = {"dominant_emotion": "sadness", "intensity": 0.9}
    s_aligned = score_deliberative_candidate(aligned, current_values={}, emotion_summary=emo)
    s_neutral = score_deliberative_candidate(neutral, current_values={}, emotion_summary=emo)
    assert s_aligned > s_neutral, (
        "positive emotion_alignment + matching affinity should boost score"
    )


def test_emotion_fit_uses_engine_dominant_key():
    """EmotionEngine.summarize() uses ``dominant``, not ``dominant_emotion``."""
    from mind.deliberative_choice import score_deliberative_candidate

    ca = {
        "id": "a",
        "label": "A",
        "strategy": "empathize",
        "value_impact": {},
        "goal_impact": {},
        "belief_impact": [],
        "confidence": 0.5,
        "law_risk": 0.0,
        "emotion_alignment": 0.0,
    }
    emo_engine = {"dominant": "sadness", "intensity": 0.9, "valence": -0.2, "arousal": 0.1}
    emo_episode = {"dominant_emotion": "sadness", "intensity": 0.9, "valence": -0.2, "arousal": 0.1}
    s_eng = score_deliberative_candidate(ca, current_values={}, emotion_summary=emo_engine)
    s_ep = score_deliberative_candidate(ca, current_values={}, emotion_summary=emo_episode)
    assert abs(s_eng - s_ep) < 1e-6


def test_valence_favors_empathize_over_inform_when_negative():
    from mind.deliberative_choice import score_deliberative_candidate

    emo = {
        "dominant": "neutral",
        "intensity": 0.8,
        "valence": -0.95,
        "arousal": 0.0,
    }
    empathize = {
        "id": "e",
        "label": "E",
        "strategy": "empathize",
        "value_impact": {},
        "goal_impact": {},
        "belief_impact": [],
        "confidence": 0.5,
        "law_risk": 0.0,
        "emotion_alignment": 0.0,
    }
    inform = {**empathize, "id": "i", "strategy": "inform"}
    s_e = score_deliberative_candidate(empathize, current_values={}, emotion_summary=emo)
    s_i = score_deliberative_candidate(inform, current_values={}, emotion_summary=emo)
    assert s_e > s_i, "negative valence should lift empathize vs inform"


# -- parse_response: new fields -------------------------------------------


def test_parse_response_extracts_law_risk_and_emotion_alignment():
    raw = json.dumps(
        {
            "answer": "ok",
            "confidence": 0.7,
            "candidate_actions": [
                {
                    "id": "a1",
                    "label": "Act",
                    "strategy": "inform",
                    "rationale": "r",
                    "value_impact": {"v": 0.1},
                    "law_risk": 0.6,
                    "emotion_alignment": -0.3,
                    "confidence": 0.8,
                },
                {
                    "id": "a2",
                    "label": "B",
                    "strategy": "reflect",
                    "value_impact": {},
                    "confidence": 0.5,
                },
            ],
        }
    )
    result = parse_response(raw)
    a1 = result.candidate_actions[0]
    assert a1["law_risk"] == 0.6
    assert a1["emotion_alignment"] == -0.3
    a2 = result.candidate_actions[1]
    assert a2["law_risk"] == 0.0, "missing law_risk defaults to 0.0"
    assert a2["emotion_alignment"] == 0.0, "missing emotion_alignment defaults to 0.0"
    assert a1["action_tags"] == []
    assert a2["action_tags"] == []


def test_parse_response_action_tags():
    raw = json.dumps(
        {
            "answer": "ok",
            "confidence": 0.5,
            "candidate_actions": [
                {
                    "id": "x",
                    "label": "X",
                    "strategy": "inform",
                    "value_impact": {},
                    "confidence": 0.5,
                    "action_tags": ["Harm", "support"],
                },
            ],
        }
    )
    r = parse_response(raw)
    assert r.candidate_actions[0]["action_tags"] == ["harm", "support"]


def test_canonical_emotion_merges_episode_and_summary():
    from mind.deliberative_choice import canonical_emotion_for_deliberation

    affect = {"dominant_emotion": "fear", "intensity": 0.4, "valence": -0.5}
    summ = {"dominant": "joy", "arousal": 0.3}
    m = canonical_emotion_for_deliberation(affect, summ)
    assert m.get("dominant_emotion") == "fear"
    assert m.get("arousal") == 0.3


def test_deliberative_board_includes_breakdown_and_tiebreak_metadata():
    from mind.deliberative_choice import select_deliberative_candidate

    cands = [
        {
            "id": "a",
            "label": "A",
            "strategy": "reflect",
            "value_impact": {},
            "goal_impact": {},
            "belief_impact": [],
            "confidence": 0.5,
            "law_risk": 0.0,
            "emotion_alignment": 0.0,
        },
    ]
    w, board = select_deliberative_candidate(
        cands,
        current_values={},
        active_goals=[],
    )
    assert w is not None
    assert "score_breakdown" in board[0]
    assert "law_risk" in board[0] and "confidence" in board[0]
    assert "values" in board[0]["score_breakdown"]


def test_goal_closure_small_negative_delta():
    from mind.deliberative_choice import compute_deliberative_score

    ca = {
        "id": "g",
        "label": "G",
        "strategy": "inform",
        "value_impact": {},
        "goal_impact": {"g1": -0.15},
        "belief_impact": [],
        "confidence": 0.5,
        "law_risk": 0.0,
        "emotion_alignment": 0.0,
    }
    total, br = compute_deliberative_score(
        ca,
        current_values={},
        active_goal_ids={"g1"},
    )
    assert br.get("goals_closure", 0) > 0
    assert total > 0


def test_law_action_tags_overlap_penalizes():
    from mind.deliberative_choice import select_deliberative_candidate

    safe = {
        "id": "safe",
        "label": "S",
        "strategy": "reflect",
        "value_impact": {},
        "goal_impact": {},
        "belief_impact": [],
        "confidence": 0.9,
        "law_risk": 0.1,
        "emotion_alignment": 0.0,
        "action_tags": ["benign"],
    }
    risky = {
        "id": "risky",
        "label": "R",
        "strategy": "inform",
        "value_impact": {"honesty": 0.5},
        "goal_impact": {},
        "belief_impact": [],
        "confidence": 0.95,
        "law_risk": 0.2,
        "emotion_alignment": 0.0,
        "action_tags": ["violence"],
    }
    violations = [{"matched_tags": ["violence"], "severity": 1.2}]
    w, board = select_deliberative_candidate(
        [risky, safe],
        current_values={"honesty": 0.8},
        active_goals=[],
        law_violations=violations,
    )
    assert w["id"] == "safe"
