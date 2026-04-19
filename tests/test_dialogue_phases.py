"""mind.dialogue_phases — tiered dialogue roadmap."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.packed_llm_discourse import discourse_context_for_packed_llm


def test_haroma_dialogue_phase_clamped(monkeypatch):
    from mind.haroma_settings import haroma_dialogue_phase

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "99")
    assert haroma_dialogue_phase() == 9
    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "0")
    assert haroma_dialogue_phase() == 1


def test_session_line_phase1(monkeypatch):
    from mind.dialogue_phases import session_discourse_line

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "1")
    s = session_discourse_line(user_id="u1", display_name="Ada", cycle_id=5)
    assert "client_id=u1" in s
    assert "display=Ada" in s
    assert "cycle=5" in s


def test_session_line_empty_without_ids(monkeypatch):
    from mind.dialogue_phases import session_discourse_line

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "1")
    assert session_discourse_line(user_id=None, display_name=None, cycle_id=1) == ""


def test_discourse_phase2_correction_hint(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "2")
    conv = MagicMock()
    conv.is_in_conversation.return_value = True
    conv.get_context_summary.return_value = "ctx"
    out = discourse_context_for_packed_llm(
        conv,
        cycle_id=2,
        speaker_key="k",
        session_uid=True,
        first_encounter_asks_name=False,
        user_text="Actually I meant Tuesday.",
    )
    assert "ctx" in out
    assert "Actually" in out or "correcting" in out.lower()


def test_eval_line_phase3(monkeypatch):
    from mind.dialogue_phases import eval_discourse_line

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "2")
    assert eval_discourse_line(cycle_id=7, trace_id="tid") == ""
    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "3")
    s = eval_discourse_line(cycle_id=7, trace_id="abc-123")
    assert "[Eval]" in s
    assert "cycle=7" in s
    assert "trace=abc-123" in s


def test_voice_line_phase4(monkeypatch):
    from mind.dialogue_phases import voice_discourse_line

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "3")
    assert voice_discourse_line(persona_name="Core") == ""
    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "4")
    assert "persona=Core" in voice_discourse_line(persona_name="Core")


def test_discourse_phase1_appends_session(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "1")
    conv = MagicMock()
    conv.is_in_conversation.return_value = True
    conv.get_context_summary.return_value = "hi"
    out = discourse_context_for_packed_llm(
        conv,
        cycle_id=3,
        speaker_key="k",
        session_uid=True,
        first_encounter_asks_name=False,
        user_id="client_x",
        display_name=None,
        user_text="hello",
    )
    assert out.startswith("hi | [Session]")
    assert "client_id=client_x" in out


def test_discourse_phase3_includes_eval(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "3")
    conv = MagicMock()
    conv.is_in_conversation.return_value = False
    out = discourse_context_for_packed_llm(
        conv,
        cycle_id=9,
        speaker_key="k",
        session_uid=False,
        first_encounter_asks_name=False,
        user_text="hi",
        trace_id="trace-z",
    )
    assert "[Eval]" in out
    assert "cycle=9" in out
    assert "trace=trace-z" in out


def test_discourse_phase4_voice(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "4")
    conv = MagicMock()
    conv.is_in_conversation.return_value = False
    out = discourse_context_for_packed_llm(
        conv,
        cycle_id=1,
        speaker_key="k",
        session_uid=False,
        first_encounter_asks_name=False,
        persona_name="Analyst",
        essence_name="Logic",
    )
    assert "[Voice]" in out
    assert "persona=Analyst" in out
    assert "essence=Logic" in out
    assert "[Rel]" not in out
    assert "[Cog]" not in out
    assert "[Turn]" not in out
    assert "[Role]" not in out


def test_discourse_phase5_rel_session(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "5")
    conv = MagicMock()
    conv.is_in_conversation.return_value = False
    out = discourse_context_for_packed_llm(
        conv,
        cycle_id=2,
        speaker_key="k",
        session_uid=True,
        first_encounter_asks_name=True,
        user_id="u",
    )
    assert "[Rel]" in out
    assert "encounter=first" in out


def test_cog_line_phase6(monkeypatch):
    from mind.dialogue_phases import cog_discourse_line

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "5")
    assert cog_discourse_line(deliberative_flag=True) == ""
    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "6")
    assert cog_discourse_line(deliberative_flag=False) == "[Cog] deliberative=0"
    assert cog_discourse_line(deliberative_flag=True) == "[Cog] deliberative=1"


def test_discourse_phase6_cog_deliberative(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "6")
    conv = MagicMock()
    conv.is_in_conversation.return_value = False
    out = discourse_context_for_packed_llm(
        conv,
        cycle_id=1,
        speaker_key="k",
        session_uid=False,
        first_encounter_asks_name=False,
        deliberative_flag=True,
    )
    assert "[Cog]" in out
    assert "deliberative=1" in out
    assert "[Turn]" not in out
    assert "[Pack]" not in out


def test_turn_line_phase7(monkeypatch):
    from mind.dialogue_phases import turn_discourse_line

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "6")
    assert turn_discourse_line(user_text="  hello  ") == ""
    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "7")
    assert turn_discourse_line(user_text="  hello  ") == "[Turn] chars=5"
    assert turn_discourse_line(user_text="") == "[Turn] chars=0"


def test_discourse_phase7_turn_chars(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "7")
    conv = MagicMock()
    conv.is_in_conversation.return_value = False
    out = discourse_context_for_packed_llm(
        conv,
        cycle_id=1,
        speaker_key="k",
        session_uid=False,
        first_encounter_asks_name=False,
        user_text="  abc  ",
    )
    assert "[Turn]" in out
    assert "chars=3" in out
    assert "[Pack]" not in out
    assert "[Role]" not in out


def test_pack_line_phase8(monkeypatch):
    from mind.dialogue_phases import pack_discourse_line

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "7")
    assert pack_discourse_line(llm_ctx_enabled=True) == ""
    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "8")
    assert pack_discourse_line(llm_ctx_enabled=False) == "[Pack] llm_ctx=0"
    assert pack_discourse_line(llm_ctx_enabled=True) == "[Pack] llm_ctx=1"


def test_discourse_phase8_pack_llm_ctx(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "8")
    conv = MagicMock()
    conv.is_in_conversation.return_value = False
    out = discourse_context_for_packed_llm(
        conv,
        cycle_id=1,
        speaker_key="k",
        session_uid=False,
        first_encounter_asks_name=False,
        llm_ctx_enabled=True,
    )
    assert "[Pack]" in out
    assert "llm_ctx=1" in out
    assert "[Role]" not in out


def test_cycle_role_line_phase9(monkeypatch):
    from mind.dialogue_phases import cycle_role_discourse_line

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "8")
    assert cycle_role_discourse_line(role="conversant") == ""
    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "9")
    assert cycle_role_discourse_line(role="") == ""
    assert cycle_role_discourse_line(role="conversant") == "[Role] role=conversant"
    assert "a_b" in cycle_role_discourse_line(role="a|b")


def test_discourse_phase9_cycle_role(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "9")
    conv = MagicMock()
    conv.is_in_conversation.return_value = False
    out = discourse_context_for_packed_llm(
        conv,
        cycle_id=1,
        speaker_key="k",
        session_uid=False,
        first_encounter_asks_name=False,
        role="conversant",
    )
    assert "[Role]" in out
    assert "role=conversant" in out
