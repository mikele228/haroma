"""Phased dialogue roadmap (dialogue-only Haroma).

Tier ``HAROMA_DIALOGUE_PHASE`` (1–9) gates incremental behavior. Higher tiers include
lower-tier features.

**Phase 1 — Session & memory discipline:** optional ``[Session]`` line in packed-LLM
discourse when ``user_id`` / ``display_name`` are known (see
:func:`mind.packed_llm_discourse.discourse_context_for_packed_llm`).

**Phase 2 — Contradiction & correction:** when user text looks like a correction, append a
short discourse hint so the model reconciles with memory.

**Phase 3 — Multi-turn eval:** append ``[Eval]`` line with ``cycle`` and optional ``trace``
for scripted regression / log correlation.

**Phase 4 — Persona / prompt richness:** ``[Voice] persona=…`` and optional ``essence=…``
(soul essence from identity). Set ``HAROMA_DIALOGUE_EVAL_LOG=1`` for INFO logs when
discourse is enriched (phase >= 3).

**Phase 5 — Encounter / relationship (compact):** when ``session_uid`` is true, append
``[Rel] encounter=first`` or ``encounter=returning`` (first-exchange name prompt vs later
turns). Omitted when there is no session identity (e.g. controller bridge without user).

**Phase 6 — Cognitive mode:** append ``[Cog] deliberative=0|1`` so packed discourse matches
the deliberative multi-goal path (same flag as agent state JSON). Callers pass the cycle’s
``deliberative_flag``.

**Phase 7 — Turn shape:** append ``[Turn] chars=N`` (stripped user message length).

**Phase 8 — Packed path:** append ``[Pack] llm_ctx=0|1`` from the same gate as
``PackedLlmPathState.llm_ctx_enabled`` (structured packed LLM context on this cycle).

**Phase 9 — Cycle role:** append ``[Role] role=…`` for the cognitive cycle role string
(e.g. ``conversant`` vs internal). Callers pass *role* from the packed-LLM entry point.

Implementation notes: keep helpers pure and testable; wire through discourse only unless
a later phase needs memory graph changes.
"""

from __future__ import annotations

import logging
from typing import Optional

from mind.haroma_settings import (
    HAROMA_DIALOGUE_PHASE_MAX,
    haroma_dialogue_eval_log_enabled,
    haroma_dialogue_phase,
)

_log = logging.getLogger(__name__)

PHASE_SESSION = 1
PHASE_CORRECTION = 2
PHASE_MULTI_TURN_EVAL = 3
PHASE_PERSONA_RICHNESS = 4
PHASE_RELATIONSHIP = 5
PHASE_COGNITIVE_MODE = 6
PHASE_TURN_SHAPE = 7
PHASE_PACKED_PATH = 8
PHASE_CYCLE_ROLE = 9

_TURN_CHARS_CAP = 999_999
_CYCLE_ROLE_MAX_LEN = 64


def dialogue_phase_at_least(n: int) -> bool:
    """True when :func:`~mind.haroma_settings.haroma_dialogue_phase` is >= *n* (clamped 1–max)."""
    return haroma_dialogue_phase() >= max(1, min(HAROMA_DIALOGUE_PHASE_MAX, int(n)))


_CORRECTION_HINT = (
    "[Discourse note: User may be correcting earlier information; "
    "reconcile with recalled context if needed.]"
)


def _join_discourse(base: str, segment: str) -> str:
    """Append *segment* with ``" | "`` if *base* is non-empty; otherwise return *segment*."""
    if not segment:
        return base
    return f"{base} | {segment}" if base else segment


def _correction_markers_in_text(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    if len(t) < 4:
        return False
    markers = (
        "actually ",
        "actually,",
        "i meant",
        "correction:",
        "not what i",
        "that's wrong",
        "that was wrong",
        "you're wrong",
        "wrong about",
    )
    return any(m in t for m in markers)


def session_discourse_line(
    *,
    user_id: Optional[str],
    display_name: Optional[str],
    cycle_id: int,
) -> str:
    """Compact session line for packed discourse (Phase 1). Empty if nothing to add."""
    if not dialogue_phase_at_least(PHASE_SESSION):
        return ""
    parts: list[str] = []
    uid = str(user_id).strip() if user_id else ""
    if uid:
        parts.append(f"client_id={uid}")
    dn = str(display_name).strip() if display_name else ""
    if dn:
        parts.append(f"display={dn[:120]}")
    if not parts:
        return ""
    parts.append(f"cycle={cycle_id}")
    return "[Session] " + " ".join(parts)


def eval_discourse_line(*, cycle_id: int, trace_id: Optional[str]) -> str:
    """Phase 3 — stable tokens for multi-turn / CI matching."""
    if not dialogue_phase_at_least(PHASE_MULTI_TURN_EVAL):
        return ""
    parts = [f"cycle={cycle_id}"]
    tid = str(trace_id).strip() if trace_id else ""
    if tid:
        parts.append(f"trace={tid[:64]}")
    return "[Eval] " + " ".join(parts)


def voice_discourse_line(*, persona_name: str, essence_name: str = "") -> str:
    """Phase 4 — persona display name and optional soul essence for the packed prompt."""
    if not dialogue_phase_at_least(PHASE_PERSONA_RICHNESS):
        return ""
    name = str(persona_name or "").strip()
    ess = str(essence_name or "").strip()
    if not name and not ess:
        return ""
    parts: list[str] = []
    if name:
        parts.append(f"persona={name[:120]}")
    if ess:
        parts.append(f"essence={ess[:120]}")
    return "[Voice] " + " ".join(parts)


def rel_discourse_line(*, session_uid: bool, first_encounter_asks_name: bool) -> str:
    """Phase 5 — compact encounter tag when a session client identity is active."""
    if not dialogue_phase_at_least(PHASE_RELATIONSHIP):
        return ""
    if not session_uid:
        return ""
    if first_encounter_asks_name:
        return "[Rel] encounter=first"
    return "[Rel] encounter=returning"


def cog_discourse_line(*, deliberative_flag: bool) -> str:
    """Phase 6 — deliberative multi-goal path flag (aligned with packed agent state)."""
    if not dialogue_phase_at_least(PHASE_COGNITIVE_MODE):
        return ""
    return "[Cog] deliberative=1" if deliberative_flag else "[Cog] deliberative=0"


def turn_discourse_line(*, user_text: str) -> str:
    """Phase 7 — stripped user message length (compact, grep-friendly)."""
    if not dialogue_phase_at_least(PHASE_TURN_SHAPE):
        return ""
    n = len((user_text or "").strip())
    if n > _TURN_CHARS_CAP:
        n = _TURN_CHARS_CAP
    return f"[Turn] chars={n}"


def pack_discourse_line(*, llm_ctx_enabled: bool) -> str:
    """Phase 8 — packed LLM context path (matches :class:`~mind.packed_llm_paths.PackedLlmPathState`)."""
    if not dialogue_phase_at_least(PHASE_PACKED_PATH):
        return ""
    return "[Pack] llm_ctx=1" if llm_ctx_enabled else "[Pack] llm_ctx=0"


def cycle_role_discourse_line(*, role: str) -> str:
    """Phase 9 — cognitive cycle role (e.g. conversant) for grep/eval."""
    if not dialogue_phase_at_least(PHASE_CYCLE_ROLE):
        return ""
    r = str(role or "").strip()
    if not r:
        return ""
    r = r.replace("|", "_").replace("\n", " ")[:_CYCLE_ROLE_MAX_LEN]
    return f"[Role] role={r}"


def _log_discourse_enriched_if_configured(
    *,
    out: str,
    cycle_id: int,
    trace_id: Optional[str],
    deliberative_flag: bool,
) -> None:
    if not (
        haroma_dialogue_eval_log_enabled()
        and dialogue_phase_at_least(PHASE_MULTI_TURN_EVAL)
        and out
    ):
        return
    tid = (str(trace_id).strip()[:32] if trace_id else "")
    if dialogue_phase_at_least(PHASE_COGNITIVE_MODE):
        _log.info(
            "dialogue_discourse_enriched cycle=%s trace=%s len=%s deliberative=%s",
            cycle_id,
            tid,
            len(out),
            1 if deliberative_flag else 0,
        )
    else:
        _log.info(
            "dialogue_discourse_enriched cycle=%s trace=%s len=%s",
            cycle_id,
            tid,
            len(out),
        )


def enrich_discourse_for_dialogue_phases(
    base: str,
    *,
    cycle_id: int,
    user_id: Optional[str],
    display_name: Optional[str],
    user_text: str,
    trace_id: Optional[str] = None,
    persona_name: str = "",
    essence_name: str = "",
    session_uid: bool = False,
    first_encounter_asks_name: bool = False,
    deliberative_flag: bool = False,
    llm_ctx_enabled: bool = False,
    role: str = "",
) -> str:
    """Append Phase 1–9 lines to *base* (session, role, rel, turn, correction, eval, cog, pack, voice)."""
    out = base
    out = _join_discourse(
        out,
        session_discourse_line(
            user_id=user_id, display_name=display_name, cycle_id=cycle_id
        ),
    )
    out = _join_discourse(out, cycle_role_discourse_line(role=role))
    out = _join_discourse(
        out,
        rel_discourse_line(
            session_uid=session_uid,
            first_encounter_asks_name=first_encounter_asks_name,
        ),
    )
    out = _join_discourse(out, turn_discourse_line(user_text=user_text))
    if dialogue_phase_at_least(PHASE_CORRECTION) and _correction_markers_in_text(user_text):
        out = _join_discourse(out, _CORRECTION_HINT)
    out = _join_discourse(out, eval_discourse_line(cycle_id=cycle_id, trace_id=trace_id))
    out = _join_discourse(out, cog_discourse_line(deliberative_flag=deliberative_flag))
    out = _join_discourse(out, pack_discourse_line(llm_ctx_enabled=llm_ctx_enabled))
    out = _join_discourse(
        out, voice_discourse_line(persona_name=persona_name, essence_name=essence_name)
    )
    _log_discourse_enriched_if_configured(
        out=out,
        cycle_id=cycle_id,
        trace_id=trace_id,
        deliberative_flag=deliberative_flag,
    )
    return out
