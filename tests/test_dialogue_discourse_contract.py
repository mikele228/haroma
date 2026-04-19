"""Golden / contract tests for :func:`mind.dialogue_phases.enrich_discourse_for_dialogue_phases`.

See ``docs/adr/0002-dialogue-discourse-phase-ordering.md``. If enrichment order changes,
update the expected strings here.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_enrich_golden_full_stack_phase9(monkeypatch):
    from mind.dialogue_phases import enrich_discourse_for_dialogue_phases

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "9")
    out = enrich_discourse_for_dialogue_phases(
        "",
        cycle_id=7,
        user_id="u1",
        display_name="Ada",
        user_text="ping",
        trace_id="trace-contract",
        persona_name="P",
        essence_name="E",
        session_uid=True,
        first_encounter_asks_name=True,
        deliberative_flag=True,
        llm_ctx_enabled=True,
        role="conversant",
    )
    assert out == (
        "[Session] client_id=u1 display=Ada cycle=7 | "
        "[Role] role=conversant | "
        "[Rel] encounter=first | "
        "[Turn] chars=4 | "
        "[Eval] cycle=7 trace=trace-contract | "
        "[Cog] deliberative=1 | "
        "[Pack] llm_ctx=1 | "
        "[Voice] persona=P essence=E"
    )


def test_enrich_golden_correction_between_turn_and_eval(monkeypatch):
    """Correction discourse note appears after [Turn] and before [Eval]."""
    from mind.dialogue_phases import enrich_discourse_for_dialogue_phases

    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "9")
    out = enrich_discourse_for_dialogue_phases(
        "ctx",
        cycle_id=1,
        user_id=None,
        display_name=None,
        user_text="Actually I meant Tuesday.",
        trace_id=None,
        session_uid=False,
        role="conversant",
    )
    assert out.startswith("ctx | ")
    assert "[Turn]" in out
    assert "[Eval]" in out
    turn_pos = out.index("[Turn]")
    corr_pos = out.index("Discourse note: User may be correcting")
    eval_pos = out.index("[Eval]")
    assert turn_pos < corr_pos < eval_pos
