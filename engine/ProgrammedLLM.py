"""
ProgrammedLLM — minimal stand-in for GGUF/API inference.

Use when you want deterministic JSON without loading weights.
Enable with ``soul/agents.json`` → ``"llm": { "engine": "programmed" }``
or env ``HAROMA_LLM_ENGINE=programmed``.

``generate_chat`` returns **only** JSON text matching ``LLMContextReasoner`` schema.

There is **no** pattern-based routing (name/math/greeting, etc.); the stub defers
to real models for content. Post-parse handling lives in the reasoner / action loop.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple


def _json_reply(
    answer: str,
    *,
    confidence: float = 0.82,
    reasoning_steps: Optional[List[str]] = None,
    cited_memories: Optional[List[int]] = None,
    requires_confirmation: bool = False,
) -> str:
    payload = {
        "answer": answer,
        "confidence": max(0.0, min(1.0, confidence)),
        "reasoning_steps": reasoning_steps or ["Programmed stub (no intent routing)."],
        "inferences": [],
        "cited_memories": cited_memories or [],
        "requires_confirmation": bool(requires_confirmation),
    }
    return json.dumps(payload, ensure_ascii=False)


def _persona_name_from_context(flat: str) -> str:
    m = re.search(r"You are (\w+)", flat, re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"\[PERSONA\].*?You are ([^\n,.\(]+)", flat, re.I | re.S)
    if m:
        return m.group(1).strip()
    return "Elarion"


def _collect_messages(messages: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Return (combined_upper_case_block, last_user_text)."""
    parts: List[str] = []
    last_user = ""
    for m in messages:
        role = str(m.get("role", "user")).strip().upper()
        content = str(m.get("content", "") or "").strip()
        if not content:
            continue
        parts.append(f"{role}: {content}")
        if role == "USER":
            last_user = content
    return "\n".join(parts), last_user


def _decide_answer(persona: str, user_block: str) -> Tuple[str, float, bool]:
    """Single non-classifying reply — no keyword buckets for question types."""
    del user_block  # prompt carries context; stub does not interpret utterances.
    return (
        f"I am the programmed LLM stub ({persona}). "
        "Use a neural or API model to answer from the messages above.",
        0.28,
        True,
    )


class ProgrammedLLMResponder:
    """Stateless programmed backend compatible with ``LLMBackend.generate*``."""

    model_tag = "programmed:haroma-v1"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """Plain-prompt path: return short text (not JSON)."""
        flat, last = _collect_messages([{"role": "user", "content": prompt}])
        persona = _persona_name_from_context(prompt)
        ans, conf, _ = _decide_answer(persona, last or prompt)
        if temperature > 0.85 and random.random() < 0.2:
            return f"{ans} (lightweight stub.)"
        return ans

    def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> Optional[str]:
        if not messages:
            return None
        flat, last_user = _collect_messages(messages)
        persona = _persona_name_from_context(flat)
        answer, confidence, req_conf = _decide_answer(persona, last_user or flat)
        steps = [
            "ProgrammedLLM: stub only — no intent classification.",
            f"Persona label: {persona}.",
        ]
        return _json_reply(
            answer,
            confidence=confidence,
            reasoning_steps=steps,
            requires_confirmation=req_conf,
        )
