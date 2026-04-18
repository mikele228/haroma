"""Knowledge-graph triple selection for packed-context LLM prompts.

When :mod:`engine.LLMContextReasoner` runs the full ``build_messages`` path, NLU entity
names and ``knowledge_summary`` feed :meth:`engine.LanguageComposer.select_relevant_triples`.
Skipped when chat-only or dummy-without-full-pack (same gate as :func:`mind.packed_llm_inputs.should_skip_full_pack_messages_for_llm`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def kg_triples_for_packed_llm_prompt(
    *,
    llm_ctx_enabled: bool,
    skip_full_pack_messages: bool,
    knowledge_summary: Dict[str, Any],
    knowledge_graph: Any,
    nlu_result: Optional[Dict[str, Any]],
) -> Optional[List[Any]]:
    """Return triples for the packed prompt, or ``None`` when selection is skipped or fails.

    *llm_ctx_enabled* is true when any packed LLM path (centric, primary, or classic) is on.
    """
    if not llm_ctx_enabled or skip_full_pack_messages:
        return None
    try:
        from engine.LanguageComposer import LanguageComposer

        nlu_names = [
            e.get("text", "")
            for e in (nlu_result.get("entities", []) if nlu_result else [])
        ]
        return LanguageComposer.select_relevant_triples(
            knowledge_summary,
            knowledge_graph,
            nlu_names,
        )
    except Exception:
        return None
