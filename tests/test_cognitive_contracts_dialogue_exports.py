"""``mind.cognitive_contracts`` dialogue symbols stay in ``__all__`` and match ``dialogue_phases``."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DIALOGUE_BARREL_NAMES = (
    "PHASE_SESSION",
    "PHASE_CORRECTION",
    "PHASE_MULTI_TURN_EVAL",
    "PHASE_PERSONA_RICHNESS",
    "PHASE_RELATIONSHIP",
    "PHASE_COGNITIVE_MODE",
    "PHASE_TURN_SHAPE",
    "PHASE_PACKED_PATH",
    "PHASE_CYCLE_ROLE",
    "dialogue_phase_at_least",
    "session_discourse_line",
    "eval_discourse_line",
    "voice_discourse_line",
    "rel_discourse_line",
    "cog_discourse_line",
    "turn_discourse_line",
    "pack_discourse_line",
    "cycle_role_discourse_line",
    "enrich_discourse_for_dialogue_phases",
    "haroma_dialogue_phase",
    "HAROMA_DIALOGUE_PHASE_MAX",
    "haroma_dialogue_eval_log_enabled",
    "haroma_memory_recall_intensity",
)


def test_dialogue_exports_in_barrel_all_and_match_dialogue_phases():
    import mind.cognitive_contracts as cc
    import mind.dialogue_phases as dp
    import mind.haroma_settings as hs

    for name in _DIALOGUE_BARREL_NAMES:
        assert name in cc.__all__, f"missing {name!r} in cognitive_contracts.__all__"
        if name.startswith("PHASE_") or name in (
            "dialogue_phase_at_least",
            "session_discourse_line",
            "eval_discourse_line",
            "voice_discourse_line",
            "rel_discourse_line",
            "cog_discourse_line",
            "turn_discourse_line",
            "pack_discourse_line",
            "cycle_role_discourse_line",
            "enrich_discourse_for_dialogue_phases",
        ):
            assert getattr(cc, name) is getattr(dp, name), name
        else:
            assert getattr(cc, name) is getattr(hs, name), name
