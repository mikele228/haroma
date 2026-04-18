"""Debug detail strings for packed-LLM pipeline logging (PersonaAgent).

Keeps ``log_input_pipeline`` payloads consistent; env probes live in
:mod:`mind.packed_llm_dummy_env` (no ``agents`` import).
"""

from __future__ import annotations

from mind.packed_llm_dummy_env import packed_llm_dummy_probe_active, packed_llm_dummy_reply_raw


def packed_llm_before_llm_log_detail(
    *,
    agent_id: str,
    role: str,
    llm_ctx_enabled: bool,
) -> str:
    """Detail line for ``persona.before_packed_llm`` (dummy / probe visibility)."""
    _dum_raw = packed_llm_dummy_reply_raw()
    return (
        f"agent={agent_id} role={role} llm_enabled={llm_ctx_enabled} "
        f"dummy_env={packed_llm_dummy_probe_active()} "
        f"HAROMA_LLM_DUMMY_REPLY={_dum_raw!r}"
    )
