"""
LLM context-reasoning module: packs user turn, recalled memories, and
stable persona context into a structured LLM prompt, then parses the
model's JSON reply into an ``LLMContextResult``.

The LLM is treated as a **proposal engine** — its outputs are advisory
and must be scored / grounded before committing to memory or surfacing
to the user.

Design choices
--------------
* Pure-function ``build_messages`` so it is testable without a live LLM.
* ``generate_chat``-compatible message list (system + user).
* JSON schema enforced via prompt instruction (works with any backend).
* Confidence, citations, ``requires_confirmation``, and ``body_actions``
  (embodiment aligned to active goals) let downstream modules decide trust level.

Env (development / probe — not production modes)
------------------------------------------------
The following flags exist for tests and local profiling. They may be removed or
superseded; **do not** treat them as the supported way to tune production ``/chat``
latency (that is :class:`~agents.persona_agent.PersonaAgent` HTTP-trace recall,
phasing, and budgets).

* ``HAROMA_LLM_LOG_PACKED_STATS`` (``1``/``true``): after ``build_messages``,
  log one line with UTF-8 byte size, character count, rough token estimate
  (chars/4), and per-role character counts.
* ``HAROMA_LLM_LOG_PAYLOAD`` (``1``/``true``): before/after each real ``generate_chat``,
  print JSON lines ``PAYLOAD_IN`` / ``PAYLOAD_OUT`` (messages list and raw model text).
  Truncation: ``HAROMA_LLM_LOG_PAYLOAD_PER_MESSAGE_CHARS`` (default ``16000``, hard max ``128000``) per message
  body, ``HAROMA_LLM_LOG_PAYLOAD_OUT_CHARS`` (default ``32000``, hard max ``256000``) for the assistant raw string.
  **Secrets:** prompts may contain recalled text or API-adjacent content — use only on trusted logs.
* ``HAROMA_LLM_DUMMY_REPLY`` (``1``/``true``): skip ``generate_chat`` and
  return immediately (default user-visible text ``Testing reply``). Chat JSON
  includes ``latency_trace`` automatically (see :mod:`agents.chat_latency`);
  set ``HAROMA_LLM_DUMMY_NO_LATENCY_TRACE=1`` to omit it. If the env
  is unset but the backend is missing or not ``available``, the same synthetic
  reply is used (``source=dummy_probe``) — there is no empty ``llm_unavailable``
  result from this module.
  For packed (non-``HAROMA_LLM_CHAT_ONLY``) prompts the reply is built as **JSON**
  in the same shape the model is asked for, then parsed with ``parse_response``
  so ``raw_json`` / fields match a real LLM pass. ``HAROMA_LLM_CHAT_ONLY`` keeps
  a plain-text ``answer`` like the real chat-only path (no JSON layer).
* ``HAROMA_LLM_DUMMY_REPLY_TEXT``: override dummy answer (default ``Testing reply``).
* ``HAROMA_LLM_DUMMY_REPLY_VERBOSE`` (``1``/``true``): put full packed-stats probe
  string in the JSON ``answer`` field (legacy long probe text) instead of the
  short dummy string.
* ``HAROMA_LLM_DUMMY_REPLY_JSON``: optional raw JSON string (single object). When
  set, it is parsed with ``parse_response`` instead of the built-in synthetic
  object (still tagged ``dummy_probe``). Invalid JSON falls back to the built-in
  synthetic JSON.
* ``HAROMA_LLM_DUMMY_FULL_PACK`` (``0``/``false``): with ``HAROMA_LLM_DUMMY_REPLY``,
  use a tiny placeholder prompt instead of ``build_messages`` (legacy fast probe).
  **Default when dummy is set:** full ``build_messages`` (honest latency vs production
  minus decode). Set ``HAROMA_LLM_DUMMY_FAST_PACK=1`` for the same minimal prompt
  without setting ``FULL_PACK=0``.
  **No backend** (dummy env unset): unchanged — full pack only if
  ``HAROMA_LLM_DUMMY_FULL_PACK`` is truthy.
* ``HAROMA_LLM_CHAT_ONLY`` (``1``/``true``): send only the user line as a
  single user message (no soul, recall, KG, or JSON schema). The model reply
  is treated as plain text (not JSON). For JSON-style packed chat, leave
  this unset.
* ``HAROMA_LLM_PROMPT_INFO`` (``1``/``true``): include ``prompt_info`` on the
  result (prompt sizes + ``n_ctx_allocated``). Also enabled automatically
  when ``HAROMA_LLM_CHAT_ONLY`` is on.
* ``HAROMA_LLM_JSON_STRICT`` (``1``/``true``): reject model output that is not
  parseable as a single JSON object (no plain-text fallback). Strips ```json
  fences before parsing.
* ``HAROMA_LLM_JSON_ANSWER_KEY`` (default ``answer``): JSON field used as the
  user-visible reply text after parsing.
* ``HAROMA_LLM_FALLBACK_WALL_SEC`` (default ``300``): if both the context timeout
  and ``HAROMA_LLM_MAX_GENERATE_SEC`` are set to unlimited, this ceiling is applied
  so native ``llama-cpp`` is never invoked without a wall-clock bound (which would
  freeze the chat thread indefinitely).
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Dict, List, Optional

from mind.packed_llm_context import host_environment_sections_for_prompt
from mind.haroma_settings import synthetic_llm_dummy_reply_env
from mind.packed_llm_inputs import synthetic_uses_placeholder_prompt
from mind.response_text import clean_packed_json_answer_text, sanitize_llm_plain_answer


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def packed_messages_stats(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Measure packed chat prompt size (for logging / dummy probe).

    ``est_tokens_approx`` is a rough lower bound (chars/4); real tokenizer
    counts differ by model.
    """
    by_role: Dict[str, int] = {}
    total_chars = 0
    total_bytes = 0
    for m in messages:
        role = str(m.get("role", "?"))
        content = str(m.get("content", "") or "")
        n = len(content)
        by_role[role] = by_role.get(role, 0) + n
        total_chars += n
        total_bytes += len(content.encode("utf-8"))
    est = max(1, (total_chars + 3) // 4)
    return {
        "message_count": len(messages),
        "total_chars": total_chars,
        "total_utf8_bytes": total_bytes,
        "est_tokens_approx": est,
        "chars_by_role": dict(by_role),
    }


def _log_packed_stats(
    stats: Dict[str, Any],
    *,
    n_ctx: Optional[int] = None,
) -> None:
    _extra = f" | n_ctx_allocated={n_ctx}" if n_ctx is not None else ""
    print(
        "[LLMContextReasoner] packed_prompt stats: "
        f"{stats['total_chars']} chars, {stats['total_utf8_bytes']} UTF-8 bytes, "
        f"~{stats['est_tokens_approx']} est. tokens (chars/4 rule), "
        f"{stats['message_count']} messages, chars_by_role={stats['chars_by_role']}"
        f"{_extra}",
        flush=True,
    )


def _payload_log_per_message_chars() -> int:
    raw = str(os.environ.get("HAROMA_LLM_LOG_PAYLOAD_PER_MESSAGE_CHARS", "16000") or "").strip()
    try:
        return max(256, min(int(raw), 128_000))
    except (TypeError, ValueError):
        return 16000


def _payload_log_out_chars() -> int:
    raw = str(os.environ.get("HAROMA_LLM_LOG_PAYLOAD_OUT_CHARS", "32000") or "").strip()
    try:
        return max(512, min(int(raw), 256_000))
    except (TypeError, ValueError):
        return 32000


def _json_line_for_payload_log(obj: Dict[str, Any]) -> str:
    """Serialize log payload; fall back to ASCII escapes if values are not JSON-safe."""
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(obj, ensure_ascii=True, default=str)


def _truncate_payload_log_text(s: str, limit: int) -> str:
    t = str(s)
    if len(t) <= limit:
        return t
    return t[: max(0, limit - 24)] + "\n… [truncated] …\n"


def _log_llm_payload_in(messages: List[Dict[str, Any]]) -> None:
    """Log packed chat messages sent to ``generate_chat`` (opt-in via ``HAROMA_LLM_LOG_PAYLOAD``)."""
    if not _env_truthy("HAROMA_LLM_LOG_PAYLOAD"):
        return
    cap = _payload_log_per_message_chars()
    try:
        slim: List[Dict[str, Any]] = []
        for m in messages:
            role = str(m.get("role", "?"))
            content = _truncate_payload_log_text(str(m.get("content", "") or ""), cap)
            slim.append({"role": role, "content": content})
        line = _json_line_for_payload_log({"kind": "PAYLOAD_IN", "messages": slim})
        print(f"[LLMContextReasoner] {line}", flush=True)
    except Exception as exc:
        print(f"[LLMContextReasoner] PAYLOAD_IN log error: {exc}", flush=True)


def _log_llm_payload_out(
    raw: Optional[str],
    *,
    error: Optional[str] = None,
) -> None:
    """Log raw model text (or timeout/error) after ``generate_chat``."""
    if not _env_truthy("HAROMA_LLM_LOG_PAYLOAD"):
        return
    cap = _payload_log_out_chars()
    try:
        payload: Dict[str, Any] = {"kind": "PAYLOAD_OUT"}
        if error:
            payload["error"] = error[:2000]
        if raw is None:
            payload["raw"] = None
        else:
            payload["raw"] = _truncate_payload_log_text(str(raw), cap)
            payload["raw_chars"] = len(str(raw))
        line = _json_line_for_payload_log(payload)
        print(f"[LLMContextReasoner] {line}", flush=True)
    except Exception as exc:
        print(f"[LLMContextReasoner] PAYLOAD_OUT log error: {exc}", flush=True)


def _prompt_info_payload(
    pack_stats: Dict[str, Any],
    llm_backend: Any,
    *,
    chat_only: bool,
) -> Dict[str, Any]:
    """Sizes sent to the model + KV context slot (``n_ctx``) from the backend."""
    n_ctx = getattr(llm_backend, "_n_ctx", None)
    return {
        "prompt_chars": pack_stats["total_chars"],
        "prompt_utf8_bytes": pack_stats["total_utf8_bytes"],
        "est_prompt_tokens_approx": pack_stats["est_tokens_approx"],
        "message_count": pack_stats["message_count"],
        "chars_by_role": pack_stats["chars_by_role"],
        "n_ctx_allocated": n_ctx,
        "chat_only": chat_only,
    }


def _should_include_prompt_info(chat_only: bool, synthetic: bool) -> bool:
    """*synthetic* = env dummy or no-backend fallback (same short path as dummy)."""
    return _env_truthy("HAROMA_LLM_PROMPT_INFO") or chat_only or synthetic


def _llm_context_timeout_seconds() -> Optional[float]:
    """Max wall time for one packed ``generate_chat`` call (local GGUF can stall).

    ``HAROMA_LLM_CONTEXT_TIMEOUT_SEC`` (default ``600``): cap wait so the HTTP
    path can finish. Large GGUF first load + first reply on CPU often exceed
    several minutes.     Use ``0`` / ``off`` / ``none`` for *unbounded* scheduling; the actual native
    ``generate_chat`` call is still limited by ``HAROMA_LLM_MAX_GENERATE_SEC``
    (default 7200s) unless that is set to ``0``/``off`` (can hang forever).
    Values below ``0.25`` are clamped to ``0.25``.
    """
    raw = str(os.environ.get("HAROMA_LLM_CONTEXT_TIMEOUT_SEC", "600") or "").strip().lower()
    if raw in ("", "0", "off", "none", "false", "no", "inf", "infinite"):
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 600.0
    if v <= 0:
        return None
    return min(max(v, 0.25), 7200.0)


def _unbounded_generate_cap_seconds() -> Optional[float]:
    """When ``HAROMA_LLM_CONTEXT_TIMEOUT_SEC`` is unlimited (``0``/off), still
    bound ``generate_chat`` so the HTTP thread cannot hang forever on native code.

    ``HAROMA_LLM_MAX_GENERATE_SEC`` (default ``7200``): max seconds for one call.
    Set ``0`` / ``off`` / ``none`` for a truly synchronous, uncapped call (can hang).
    """
    raw = str(os.environ.get("HAROMA_LLM_MAX_GENERATE_SEC", "7200") or "").strip().lower()
    if raw in ("0", "off", "none", "inf", "infinite"):
        return None
    try:
        v = float(raw) if raw else 7200.0
    except (TypeError, ValueError):
        return 7200.0
    if v <= 0:
        return None
    return min(v, 86400.0)


_JSON_SCHEMA_DESCRIPTION = """\
{
  "answer": "<string or null — proposed reply text; null when unsure>",
  "confidence": <float 0.0–1.0>,
  "reasoning_steps": ["<step 1>", "<step 2>", ...],
  "inferences": [
    {"subject": "<str>", "predicate": "<str>", "object": "<str>", "confidence": <float 0.0–1.0>}
  ],
  "cited_memories": [<int indices into RECALLED MEMORIES>],
  "requires_confirmation": <true if answer is speculative and needs KG/memory grounding>,
  "env_updates": {
    "emotion": {"label": "<str from joy|wonder|curiosity|fear|sadness|anger|resolve|peace|surprise|neutral>", "intensity": <float 0.0–1.0>},
    "goals": [{"goal_id": "<str>", "description": "<str>", "priority": <float 0.0–1.0>, "child_goal_ids": ["<str>", ...] (optional — sub-goals that must be satisfied first), "action_items": ["<step>", ...] or [{"id":"a0","description":"<str>","done":false}] (optional checklist), "parent_goal_id": "<str>" (optional if this row is a child of another goal_id)}],
    "personality_nudges": [{"trait": "<openness|conscientiousness|extraversion|agreeableness|neuroticism|resilience|assertiveness>", "delta": <float -0.01..0.01>}],
    "kg_triples": [{"subject": "<str>", "predicate": "<str>", "object": "<str>", "confidence": <float>}],
    "wm_notes": [{"content": "<str>", "salience": <float 0.0–1.0>}],
    "memory_notes": [{"tree": "<tree_name>", "content": "<str>", "tags": ["<str>"]}]
  },
  "body_actions": [
    {
      "label": "<short name for this embodiment / actuator intent>",
      "layer": "<hardware|proprioception|localization|scene|control|safety|perception|speech|other>",
      "command": "<machine-oriented hint e.g. move_base, set_pose, look_at, gesture, torque, speak>",
      "parameters": {},
      "supports_goal_id": "<goal_id from Active goals in [PERSONA] — empty if none applies>",
      "rationale": "<how this motion advances that goal and respects [ROBOT BODY STATE]>",
      "priority": <float 0.0–1.0 relative urgency vs other body_actions this turn>,
      "resource": "<base|arm|head|hands|full_body|speech|none — actuator resource to schedule>",
      "cancel_current": <bool — true if this should preempt conflicting motion on that resource>,
      "safety_class": "<safe|caution|restricted|human_present|estop_required|no_motion>",
      "duration_hint_sec": <float optional — expected motion or hold duration>,
      "coordinate_frame": "<str optional — must match localization.frame_id when moving in map>",
      "preconditions": ["<sensor/state checks before executing, e.g. pose_valid>"],
      "confidence": <float 0.0–1.0>
    }
  ]
}"""

_JSON_SCHEMA_DELIBERATIVE_EXT = """\
  "candidate_actions": [
    {
      "id": "<short unique action identifier>",
      "label": "<human-readable action name>",
      "strategy": "<one of: inform, inquire, empathize, reflect, advance_goal, explore, observe>",
      "rationale": "<1-2 sentence explanation of why this action is appropriate>",
      "value_impact": {
        "<value_key>": <float -1.0 to 1.0 — estimated directional effect on this value>
      },
      "goal_impact": {
        "<goal_id>": <float ~-0.5..0.5 — estimated change to that goal's priority / urgency>
      },
      "belief_impact": [
        {"proposition": "<short belief statement>", "confidence_delta": <float -1..1>}
      ],
      "law_risk": <float 0.0–1.0 — estimated probability this action violates an active law from [AGENT STATE JSON].laws>,
      "emotion_alignment": <float -1.0 to 1.0 — how well this action fits the agent's current affect in [AGENT STATE JSON].affect>,
      "action_tags": ["<optional content tags for law overlap e.g. same vocabulary as law forbidden tags>"],
      "confidence": <float 0.0–1.0>
    }
  ]"""

_MAX_CANDIDATE_ACTIONS = 8
_MAX_ACTION_TAGS_PER_CANDIDATE = 8
_MAX_BODY_ACTIONS = 8

_MAX_RECALL_CHARS = 800
_MAX_KG_TRIPLES = 30
_MAX_GOAL_LINES = 8
_MAX_LAW_LINES = 12
_MAX_VALUE_KEYS = 10
_MAX_SOUL_JSON_CHARS = max(
    512,
    min(100_000, int(os.environ.get("HAROMA_SOUL_PROMPT_MAX_CHARS", "2000"))),
)


def _truncate(text: str, limit: int = 200) -> str:
    return text[:limit] + ("…" if len(text) > limit else "")


# ------------------------------------------------------------------
# Prompt builder (pure function)
# ------------------------------------------------------------------


def build_messages(
    *,
    user_text: str,
    recalled_memories: List[Dict[str, Any]],
    identity_summary: Dict[str, Any],
    personality_summary: Dict[str, float],
    active_goals: List[Dict[str, Any]],
    law_summary: Dict[str, Any],
    value_summary: Dict[str, Any],
    knowledge_triples: Optional[List[Any]] = None,
    discourse_context: str = "",
    nlu_result: Optional[Dict[str, Any]] = None,
    memory_forest_seed: str = "",
    llm_centric: bool = False,
    deliberative: bool = False,
    agent_state_json: str = "",
    agent_environment: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Build an OpenAI-style message list for the context-reasoning pass.

    When *deliberative* is True, the schema is extended with
    ``candidate_actions`` (each with ``value_impact``) and the agent's full
    state snapshot is included as ``[AGENT STATE JSON]`` in the user message.

    Returns ``[{"role": "system", ...}, {"role": "user", ...}]``.
    """
    # --- System message: persona + rules ---------------------------------
    persona_lines: List[str] = []
    name = identity_summary.get("essence_name") or "Agent"
    vessel = identity_summary.get("vessel", "")
    role = identity_summary.get("current_role", "observer")
    phase = identity_summary.get("current_phase", "stable")
    persona_lines.append(
        f"You are {name}"
        + (f" ({vessel})" if vessel else "")
        + f", currently in role={role}, phase={phase}."
    )

    _soul = identity_summary.get("soul")
    if isinstance(_soul, dict) and _soul:
        try:
            soul_txt = json.dumps(_soul, ensure_ascii=False, default=str)
            if len(soul_txt) > _MAX_SOUL_JSON_CHARS:
                soul_txt = soul_txt[:_MAX_SOUL_JSON_CHARS] + "…"
            persona_lines.append(
                "Bound soul snapshot (authoritative JSON from soul/*.json):\n" + soul_txt
            )
        except (TypeError, ValueError):
            pass
    else:
        _birth = str(identity_summary.get("birth") or "").strip()
        if _birth:
            persona_lines.append(
                f"Soul-record origin timestamp (for temporal / duration reasoning): {_birth}."
            )

    if personality_summary:
        traits = ", ".join(f"{k}={v:.2f}" for k, v in sorted(personality_summary.items()))
        persona_lines.append(f"Personality traits: {traits}.")

    # Goals
    if active_goals:
        goal_strs = []
        for g in active_goals[:_MAX_GOAL_LINES]:
            desc = g.get("description", g.get("goal_id", str(g)))
            pri = g.get("priority", "?")
            gid = g.get("goal_id", "")
            bits = [f"priority {pri}"]
            if gid:
                bits.insert(0, f"id={gid}")
            cg = g.get("child_goal_ids")
            if isinstance(cg, list) and cg:
                bits.append("children=" + ",".join(str(x) for x in cg[:6]))
            ai = g.get("action_items")
            if isinstance(ai, list) and ai:
                n = len(ai)
                done = sum(
                    1
                    for x in ai
                    if isinstance(x, dict) and x.get("done")
                )
                bits.append(f"actions {done}/{n}")
            goal_strs.append(f"  - {desc} ({'; '.join(bits)})")
        persona_lines.append(
            "Active goals (parents complete only after children + action items):\n"
            + "\n".join(goal_strs)
        )

    # Laws
    law_ids = law_summary.get("ids", [])
    if law_ids:
        persona_lines.append(
            "Applicable laws/constraints: "
            + ", ".join(str(lid) for lid in law_ids[:_MAX_LAW_LINES])
            + "."
        )

    # Values
    vkeys = value_summary.get("value_keys", [])
    if vkeys:
        persona_lines.append(
            "Core values: " + ", ".join(str(v) for v in vkeys[:_MAX_VALUE_KEYS]) + "."
        )

    _env_rules = ""
    if llm_centric:
        _env_rules = (
            "- You are the sole response generator. ALWAYS populate env_updates.\n"
            "- env_updates.emotion: set the emotion you believe the agent should "
            "feel after this interaction. Use one of: joy, wonder, curiosity, "
            "fear, sadness, anger, resolve, peace, surprise, neutral.\n"
            "- env_updates.goals: register new goals or reinforce existing ones "
            "when the conversation reveals them (max 3 per turn). You may add "
            "child_goal_ids (ordered sub-goals) and action_items (checklist); "
            "the engine will not mark a parent complete until children and "
            "checklist steps are done.\n"
            "- env_updates.personality_nudges: tiny trait adjustments (-0.01 to "
            "0.01) only when the interaction clearly warrants a shift.\n"
            "- env_updates.kg_triples: extract factual relationships from the "
            "conversation (max 5 per turn).\n"
            "- env_updates.wm_notes: short notes to remember for immediate "
            "context (max 3 per turn).\n"
            "- env_updates.memory_notes: important facts to store long-term "
            "in specific memory trees (max 3 per turn).\n"
        )

    _schema_block = _JSON_SCHEMA_DESCRIPTION
    _deliberative_rules = ""
    if deliberative:
        _schema_block = (
            _JSON_SCHEMA_DESCRIPTION.rstrip().rstrip("}")
            + ",\n"
            + _JSON_SCHEMA_DELIBERATIVE_EXT
            + "\n}"
        )
        _deliberative_rules = (
            "- DELIBERATIVE MODE: You MUST also populate candidate_actions "
            f"(1–{_MAX_CANDIDATE_ACTIONS} actions). For each action, estimate "
            "how executing it would shift each value key listed in "
            "[AGENT STATE JSON].values (positive = reinforces, negative = harms). "
            "value_impact keys MUST match value keys from the state.\n"
            "- Also provide goal_impact for active goal_ids when relevant "
            "(priority delta), and belief_impact as propositions whose credence "
            "would shift if the action were taken.\n"
            "- For each candidate, set law_risk (0.0–1.0) based on whether "
            "the action could violate any law listed in [AGENT STATE JSON].laws "
            "(0 = clearly safe, 1 = certain violation).\n"
            "- For each candidate, set emotion_alignment (-1.0 to 1.0) based on "
            "how congruent the action is with the agent's current affect in "
            "[AGENT STATE JSON].affect (positive = harmonious, negative = discordant).\n"
            "- Optional action_tags: short lowercase tokens describing the action's "
            "content (for law overlap); use tags that could match forbidden law tags.\n"
            "- Example candidate fragment: "
            '{"id":"c1","label":"Listen","strategy":"empathize","law_risk":0.1,'
            '"emotion_alignment":0.5,"action_tags":["support"],"confidence":0.7}'
            "\n"
            "- Rank candidate_actions from most to least recommended.\n"
        )

    system_text = (
        "You are a reasoning assistant embedded inside a cognitive agent. "
        "Given the context below, reason step-by-step about the user's "
        "message and produce a JSON response conforming to the schema.\n\n"
        "[PERSONA]\n" + "\n".join(persona_lines) + "\n\n"
        "[OUTPUT JSON SCHEMA]\n" + _schema_block + "\n\n"
        "Rules:\n"
        "- ALWAYS prefer grounding your answer in [MEMORY FOREST SEED], "
        "[RECALLED MEMORIES], and [KNOWLEDGE GRAPH] when they contain relevant "
        "facts.  Synthesize the information in your own words — do not copy "
        "recalled text verbatim.\n"
        "- Cite every recalled memory you use by its 0-based index in "
        "cited_memories so downstream modules can verify provenance.\n"
        "- When [RECALLED MEMORIES] or [KNOWLEDGE GRAPH] are insufficient but "
        "[PERSONA] or the bound soul snapshot contains the needed facts, "
        "answer concisely with requires_confirmation=false and confidence "
        "at least 0.8; cited_memories may be [].\n"
        "- When the answer follows only from correct reasoning on the user "
        "message (e.g. stated arithmetic), use requires_confirmation=false, "
        "confidence at least 0.9, cited_memories [].\n"
        "- If no source provides adequate support, set answer=null and "
        "requires_confirmation=true.  Do NOT hallucinate facts.\n"
        "- Confidence must reflect how well-supported the answer is by the "
        "provided context — not your general knowledge.\n"
        "- Keep reasoning_steps concise (max 5 steps).\n"
        "- When [ROBOT BODY STATE] is present, treat it as the agent's embodiment "
        "(hardware, proprioception, localization, scene). Prefer physically plausible "
        "body_actions consistent with those readings.\n"
        "- When [ROBOT BRIDGE FEEDBACK] is present, it lists recent on-robot command "
        "results (command_id, status), possibly across multiple correlation batches. "
        "Align new body_actions with completed/failed work; do not repeat successful "
        "moves unless the user asks. If a command_id shows failed/rejected, prefer a "
        "different approach or parameters rather than blindly re-issuing the same intent.\n"
        "- Populate body_actions (0–8) with embodiment / actuation intents that serve "
        "the user message and **Active goals** in [PERSONA]: set supports_goal_id to a "
        "goal_id listed there when an action materially advances that goal; use \"\" "
        "if none applies. If [ROBOT BODY STATE] shows ESTOP or safety.human_in_zone, "
        "prefer safety_class caution/restricted/no_motion until cleared.\n"
        + _env_rules
        + _deliberative_rules
        + "- Output ONLY valid JSON, no markdown fences or extra text.\n"
        + (
            "\nSTRICT JSON (HAROMA_LLM_JSON_STRICT): reply must be a single JSON object, "
            "first character `{`, last `}`. No prose or code fences before/after.\n"
            if _env_json_strict()
            else ""
        )
    )

    # --- User message + context ------------------------------------------
    user_parts: List[str] = []

    # Memory forest seed (per-tree context snapshot)
    if memory_forest_seed:
        user_parts.append(memory_forest_seed)

    # Recalled memories
    if recalled_memories:
        mem_lines: List[str] = []
        total_chars = 0
        for idx, mem in enumerate(recalled_memories):
            content = str(
                mem.get("content", "")
                if isinstance(mem, dict)
                else getattr(mem, "content", str(mem))
            ).strip()
            if not content:
                continue
            snippet = _truncate(content, 200)
            if total_chars + len(snippet) > _MAX_RECALL_CHARS:
                break
            mem_lines.append(f"  [{idx}] {snippet}")
            total_chars += len(snippet)
        if mem_lines:
            user_parts.append("[RECALLED MEMORIES]\n" + "\n".join(mem_lines))

    # Knowledge graph triples
    if knowledge_triples:
        triple_strs: List[str] = []
        for t in knowledge_triples[:_MAX_KG_TRIPLES]:
            if isinstance(t, str):
                triple_strs.append(f"  {t}")
            elif isinstance(t, dict):
                s = t.get("subject", "?")
                p = t.get("predicate", "?")
                o = t.get("object", "?")
                triple_strs.append(f"  {s} --[{p}]--> {o}")
        if triple_strs:
            user_parts.append("[KNOWLEDGE GRAPH]\n" + "\n".join(triple_strs))

    # Discourse / conversation
    if discourse_context:
        user_parts.append("[CONVERSATION CONTEXT]\n  " + _truncate(discourse_context, 400))

    # NLU
    if nlu_result:
        intent = nlu_result.get("intent", "")
        entities = nlu_result.get("entities", [])
        if intent or entities:
            nlu_str = f"intent={intent}"
            if entities:
                ent_strs = [
                    e.get("text", str(e))[:60]
                    for e in (entities[:6] if isinstance(entities, list) else [])
                ]
                nlu_str += ", entities=[" + ", ".join(ent_strs) + "]"
            user_parts.append(f"[NLU] {nlu_str}")

    if deliberative and agent_state_json:
        user_parts.append(f"[AGENT STATE JSON]\n{agent_state_json}")

    # Host world + embodiment (see ``mind.packed_llm_context`` for pipeline notes).
    user_parts.extend(host_environment_sections_for_prompt(agent_environment))

    user_parts.append(f"[USER MESSAGE]\n{user_text}")

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


# ------------------------------------------------------------------
# Response parser
# ------------------------------------------------------------------


def _env_json_strict() -> bool:
    return _env_truthy("HAROMA_LLM_JSON_STRICT")


def _json_answer_key() -> str:
    k = (os.environ.get("HAROMA_LLM_JSON_ANSWER_KEY") or "answer").strip()
    return k if k else "answer"


def _preprocess_raw_for_json(raw: str) -> str:
    """Strip markdown fences so brace-based JSON extraction can succeed."""
    s = (raw or "").strip()
    if not s or "```" not in s:
        return s
    i = s.find("```")
    rest = s[i + 3 :].lstrip()
    if len(rest) >= 4 and rest[:4].lower() == "json":
        rest = rest[4:].lstrip()
    end = rest.find("```")
    if end != -1:
        inner = rest[:end].strip()
        if inner:
            return inner
    return s


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of the first JSON object from LLM output."""
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n", 1)
        raw = lines[1] if len(lines) > 1 else ""
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    # Try direct parse first
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Find first { ... } block via brace counting; validate with json.loads
    # to avoid false positives from braces inside string literals.
    m = re.search(r"\{", raw)
    if not m:
        return None
    depth = 0
    start = m.start()
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(raw[start : i + 1])
                    if isinstance(obj, dict):
                        return obj
                except (json.JSONDecodeError, ValueError):
                    pass
                # Brace mismatch due to string contents; keep scanning
                break
    return None


def _parse_candidate_actions_from_obj(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize deliberative ``candidate_actions`` from model JSON."""
    candidate_actions: List[Dict[str, Any]] = []
    for ca in obj.get("candidate_actions") or []:
        if not isinstance(ca, dict):
            continue
        label = str(ca.get("label") or ca.get("id") or "").strip()
        if not label:
            continue
        vi = ca.get("value_impact")
        safe_vi: Dict[str, float] = {}
        if isinstance(vi, dict):
            for vk, vv in vi.items():
                try:
                    safe_vi[str(vk)] = max(-1.0, min(1.0, float(vv)))
                except (TypeError, ValueError):
                    pass
        ca_conf = 0.5
        try:
            ca_conf = max(0.0, min(1.0, float(ca.get("confidence", 0.5))))
        except (TypeError, ValueError):
            pass

        gi_raw = ca.get("goal_impact")
        safe_gi: Dict[str, float] = {}
        if isinstance(gi_raw, dict):
            for gk, gv in gi_raw.items():
                try:
                    safe_gi[str(gk)[:80]] = max(-0.5, min(0.5, float(gv)))
                except (TypeError, ValueError):
                    pass

        belief_rows: List[Dict[str, Any]] = []
        bi_raw = ca.get("belief_impact")
        if isinstance(bi_raw, list):
            for row in bi_raw[:6]:
                if not isinstance(row, dict):
                    continue
                prop = str(row.get("proposition") or row.get("belief") or "").strip()
                if not prop:
                    continue
                try:
                    cd = max(-1.0, min(1.0, float(row.get("confidence_delta", 0.0))))
                except (TypeError, ValueError):
                    cd = 0.0
                belief_rows.append(
                    {
                        "proposition": prop[:400],
                        "confidence_delta": cd,
                    }
                )

        ca_law_risk = 0.0
        try:
            ca_law_risk = max(0.0, min(1.0, float(ca.get("law_risk", 0.0))))
        except (TypeError, ValueError):
            ca_law_risk = 0.0

        ca_emotion_align = 0.0
        try:
            ca_emotion_align = max(-1.0, min(1.0, float(ca.get("emotion_alignment", 0.0))))
        except (TypeError, ValueError):
            ca_emotion_align = 0.0

        action_tags: List[str] = []
        at_raw = ca.get("action_tags")
        if isinstance(at_raw, (list, tuple)):
            for t in at_raw[:_MAX_ACTION_TAGS_PER_CANDIDATE]:
                if t is None:
                    continue
                s = str(t).strip().lower()[:40]
                if s:
                    action_tags.append(s)

        candidate_actions.append(
            {
                "id": str(ca.get("id") or label)[:60],
                "label": label[:120],
                "strategy": str(ca.get("strategy") or "")[:30],
                "rationale": str(ca.get("rationale") or "")[:300],
                "value_impact": safe_vi,
                "goal_impact": safe_gi,
                "belief_impact": belief_rows,
                "law_risk": ca_law_risk,
                "emotion_alignment": ca_emotion_align,
                "action_tags": action_tags,
                "confidence": ca_conf,
            }
        )
    return candidate_actions[:_MAX_CANDIDATE_ACTIONS]


def _parse_body_actions_from_obj(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize embodiment ``body_actions`` from model JSON."""
    body_actions: List[Dict[str, Any]] = []
    for ba in obj.get("body_actions") or []:
        if not isinstance(ba, dict):
            continue
        label = str(ba.get("label") or "").strip()
        if not label:
            continue
        params: Dict[str, Any] = {}
        pr = ba.get("parameters")
        if isinstance(pr, dict):
            params = pr
        ba_conf = 0.5
        try:
            ba_conf = max(0.0, min(1.0, float(ba.get("confidence", 0.5))))
        except (TypeError, ValueError):
            pass
        dur: Optional[float] = None
        try:
            if ba.get("duration_hint_sec") is not None:
                dur = max(0.0, min(86400.0, float(ba["duration_hint_sec"])))
        except (TypeError, ValueError):
            dur = None
        sc = str(ba.get("safety_class") or "").strip().lower()[:48]
        pre: List[str] = []
        prq = ba.get("preconditions")
        if isinstance(prq, (list, tuple)):
            for p in prq[:8]:
                s = str(p).strip()[:120]
                if s:
                    pre.append(s)
        pri = 0.5
        try:
            if ba.get("priority") is not None:
                pri = max(0.0, min(1.0, float(ba["priority"])))
        except (TypeError, ValueError):
            pri = 0.5
        res = str(ba.get("resource") or "none").strip().lower()[:32]
        cc = bool(ba.get("cancel_current")) if ba.get("cancel_current") is not None else False
        body_actions.append(
            {
                "label": label[:160],
                "layer": str(ba.get("layer") or "other").strip().lower()[:40],
                "command": str(ba.get("command") or "")[:120],
                "parameters": params,
                "supports_goal_id": str(ba.get("supports_goal_id") or "")[:80],
                "rationale": str(ba.get("rationale") or "")[:400],
                "priority": pri,
                "resource": res,
                "cancel_current": cc,
                "safety_class": sc,
                "duration_hint_sec": dur,
                "coordinate_frame": str(ba.get("coordinate_frame") or "")[:64],
                "preconditions": pre,
                "confidence": ba_conf,
            }
        )
    return body_actions[:_MAX_BODY_ACTIONS]


def parse_response(raw: Optional[str]) -> "LLMContextResult":
    """Parse the LLM's raw text into a validated ``LLMContextResult``."""
    if not raw:
        return LLMContextResult.empty("llm_empty_response")

    processed = _preprocess_raw_for_json(raw)
    obj = _extract_json(processed)
    if obj is None:
        obj = _extract_json((raw or "").strip())

    if obj is None:
        if _env_json_strict():
            return LLMContextResult.empty("json_parse_failed")
        # Models often return prose or markdown instead of strict JSON; still show it.
        plain = (raw or "").strip()
        if plain.startswith("```"):
            lines = plain.split("\n")
            if len(lines) >= 2 and lines[0].strip().startswith("```"):
                plain = "\n".join(lines[1:])
                if plain.rstrip().endswith("```"):
                    plain = plain.rstrip()[:-3].rstrip()
        plain = plain.strip()
        if plain:
            _plain = sanitize_llm_plain_answer(plain)
            if not _plain:
                return LLMContextResult.empty("json_parse_failed")
            return LLMContextResult(
                answer=_plain,
                confidence=0.45,
                reasoning_steps=[],
                requires_confirmation=True,
                source="llm_nonjson_reply",
            )
        return LLMContextResult.empty("json_parse_failed")

    ak = _json_answer_key()
    answer = obj.get(ak)
    if answer is None and ak != "answer":
        answer = obj.get("answer")
    if answer is not None:
        answer = str(answer).strip() or None
        if answer:
            answer = clean_packed_json_answer_text(answer)
            if not answer:
                answer = None

    confidence = 0.0
    try:
        confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.0))))
    except (TypeError, ValueError):
        pass

    steps: List[str] = []
    for s in obj.get("reasoning_steps") or []:
        s_str = str(s).strip()
        if s_str:
            steps.append(s_str[:300])
    steps = steps[:5]

    inferences: List[Dict[str, Any]] = []
    for inf in obj.get("inferences") or []:
        if not isinstance(inf, dict):
            continue
        subj = str(inf.get("subject", "")).strip()
        pred = str(inf.get("predicate", "")).strip()
        obj_name = str(inf.get("object", "")).strip()
        if subj and pred and obj_name:
            c = 0.5
            try:
                c = max(0.0, min(1.0, float(inf.get("confidence", 0.5))))
            except (TypeError, ValueError):
                pass
            inferences.append(
                {
                    "subject": subj,
                    "predicate": pred,
                    "object": obj_name,
                    "confidence": c,
                    "source": "llm_context_reasoning",
                }
            )
    inferences = inferences[:15]

    cited: List[int] = []
    for c_idx in obj.get("cited_memories") or []:
        try:
            cited.append(int(c_idx))
        except (TypeError, ValueError):
            pass

    requires_confirmation = bool(obj.get("requires_confirmation", True))

    raw_env = obj.get("env_updates")
    env_updates: Dict[str, Any] = {}
    if isinstance(raw_env, dict):
        env_updates = raw_env

    candidate_actions = _parse_candidate_actions_from_obj(obj)
    body_actions = _parse_body_actions_from_obj(obj)

    return LLMContextResult(
        answer=answer,
        confidence=confidence,
        reasoning_steps=steps,
        inferences=inferences,
        cited_memories=cited,
        requires_confirmation=requires_confirmation,
        source="llm_context_reasoning",
        raw_json=obj,
        env_updates=env_updates,
        candidate_actions=candidate_actions,
        body_actions=body_actions,
    )


class _DummyLLMBackendPlaceholder:
    """Stand-in when ``HAROMA_LLM_DUMMY_REPLY=1`` but no real GGUF/backend (prompt_info only)."""

    available = True
    _n_ctx = None


def _dummy_probe_result(
    *,
    answer_text: str,
    latency_ms: float,
    prompt_info: Optional[Dict[str, Any]],
    chat_only: bool,
) -> LLMContextResult:
    """Synthetic reply matching the real LLM path: JSON parse for packed prompts."""
    if chat_only:
        return LLMContextResult(
            answer=answer_text if answer_text.strip() else None,
            confidence=0.55,
            reasoning_steps=[],
            requires_confirmation=True,
            source="dummy_probe",
            latency_ms=latency_ms,
            prompt_info=prompt_info,
        )
    _override = str(os.environ.get("HAROMA_LLM_DUMMY_REPLY_JSON") or "").strip()
    if _override:
        _parsed = parse_response(_override)
        if _parsed.source != "json_parse_failed" and (
            _parsed.has_answer or (_parsed.raw_json is not None)
        ):
            _parsed.source = "dummy_probe"
            _parsed.latency_ms = latency_ms
            _parsed.prompt_info = prompt_info
            return _parsed
    ak = _json_answer_key()
    _obj: Dict[str, Any] = {
        ak: answer_text,
        "confidence": 1.0,
        "reasoning_steps": ["dummy_probe: skipped generate_chat (synthetic JSON)"],
        "requires_confirmation": False,
        "inferences": [],
        "cited_memories": [],
        "env_updates": {},
    }
    _raw = json.dumps(_obj, ensure_ascii=False)
    _result = parse_response(_raw)
    _result.source = "dummy_probe"
    _result.latency_ms = latency_ms
    _result.prompt_info = prompt_info
    return _result


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------


class LLMContextResult:
    """Structured output from a context-reasoning LLM pass."""

    __slots__ = (
        "answer",
        "confidence",
        "reasoning_steps",
        "inferences",
        "cited_memories",
        "requires_confirmation",
        "source",
        "raw_json",
        "latency_ms",
        "env_updates",
        "prompt_info",
        "candidate_actions",
        "body_actions",
    )

    def __init__(
        self,
        *,
        answer: Optional[str] = None,
        confidence: float = 0.0,
        reasoning_steps: Optional[List[str]] = None,
        inferences: Optional[List[Dict[str, Any]]] = None,
        cited_memories: Optional[List[int]] = None,
        requires_confirmation: bool = True,
        source: str = "llm_context_reasoning",
        raw_json: Optional[Dict[str, Any]] = None,
        latency_ms: float = 0.0,
        env_updates: Optional[Dict[str, Any]] = None,
        prompt_info: Optional[Dict[str, Any]] = None,
        candidate_actions: Optional[List[Dict[str, Any]]] = None,
        body_actions: Optional[List[Dict[str, Any]]] = None,
    ):
        self.answer = answer
        self.confidence = confidence
        self.reasoning_steps = reasoning_steps or []
        self.inferences = inferences or []
        self.cited_memories = cited_memories or []
        self.requires_confirmation = requires_confirmation
        self.source = source
        self.raw_json = raw_json
        self.latency_ms = latency_ms
        self.env_updates = env_updates or {}
        self.prompt_info = prompt_info
        self.candidate_actions = candidate_actions or []
        self.body_actions = body_actions or []

    @classmethod
    def empty(cls, source: str = "unavailable") -> "LLMContextResult":
        return cls(source=source)

    @property
    def has_answer(self) -> bool:
        return self.answer is not None and len(self.answer.strip()) > 0

    @property
    def is_grounded(self) -> bool:
        """True when the answer is backed by memory citations or high confidence."""
        if not self.has_answer:
            return False
        if self.requires_confirmation:
            return False
        return self.confidence >= 0.4 or len(self.cited_memories) > 0

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "answer": self.answer,
            "confidence": self.confidence,
            "reasoning_steps": self.reasoning_steps,
            "inferences": self.inferences,
            "cited_memories": self.cited_memories,
            "requires_confirmation": self.requires_confirmation,
            "source": self.source,
            "latency_ms": self.latency_ms,
            "env_updates": self.env_updates,
        }
        if self.prompt_info:
            out["prompt_info"] = self.prompt_info
        # Always include so downstream (deliberative, HTTP) sees a stable shape;
        # use [] when the model proposed no structured candidates.
        out["candidate_actions"] = self.candidate_actions
        out["body_actions"] = self.body_actions
        return out


# ------------------------------------------------------------------
# Orchestrator: build → call LLM → parse
# ------------------------------------------------------------------


def run_llm_context_reasoning(
    *,
    llm_backend: Any,
    user_text: str,
    recalled_memories: List[Dict[str, Any]],
    identity_summary: Dict[str, Any],
    personality_summary: Dict[str, float],
    active_goals: List[Dict[str, Any]],
    law_summary: Dict[str, Any],
    value_summary: Dict[str, Any],
    knowledge_triples: Optional[List[Any]] = None,
    discourse_context: str = "",
    nlu_result: Optional[Dict[str, Any]] = None,
    memory_forest_seed: str = "",
    llm_centric: bool = False,
    max_tokens: Optional[int] = None,
    temperature: float = 0.3,
    timeout_override: Optional[float] = None,
    deliberative: bool = False,
    agent_state_json: str = "",
    agent_environment: Optional[Dict[str, Any]] = None,
) -> LLMContextResult:
    """End-to-end: pack context → LLM call → parse JSON → result.

    When the backend is missing or not ``available``, returns the same synthetic
    reply as ``HAROMA_LLM_DUMMY_REPLY`` (``source=dummy_probe``) so callers always
    get a user-visible string unless ``generate_chat`` runs successfully.
    """
    _dummy = synthetic_llm_dummy_reply_env()
    _bk_ok = llm_backend is not None and getattr(llm_backend, "available", False)
    _use_synthetic = _dummy or not _bk_ok
    _effective_backend = llm_backend if _bk_ok else _DummyLLMBackendPlaceholder()

    _mt = max_tokens
    if _mt is None:
        try:
            _mt = int(os.environ.get("HAROMA_LLM_CONTEXT_MAX_TOKENS", "192") or "192")
        except (TypeError, ValueError):
            _mt = 256
        _mt = max(64, min(768, _mt))

    _chat_only = _env_truthy("HAROMA_LLM_CHAT_ONLY")
    _t_pack0 = time.perf_counter()
    if _chat_only:
        _ut = (user_text or "").strip() or "."
        messages = [{"role": "user", "content": _ut}]
        print(
            "[LLMContextReasoner] HAROMA_LLM_CHAT_ONLY=1 — single user message, "
            "no packed soul/recall/KG (plain text reply, not JSON).",
            flush=True,
        )
    elif _use_synthetic and synthetic_uses_placeholder_prompt(_dummy):
        _ut = (user_text or "").strip() or "."
        messages = [{"role": "user", "content": _ut}]
        print(
            "[LLMContextReasoner] synthetic path — minimal placeholder prompt "
            "(use HAROMA_LLM_DUMMY_FAST_PACK=1 or HAROMA_LLM_DUMMY_FULL_PACK=0; "
            "default dummy mode runs full build_messages).",
            flush=True,
        )
    else:
        messages = build_messages(
            user_text=user_text,
            recalled_memories=recalled_memories,
            identity_summary=identity_summary,
            personality_summary=personality_summary,
            active_goals=active_goals,
            law_summary=law_summary,
            value_summary=value_summary,
            knowledge_triples=knowledge_triples,
            discourse_context=discourse_context,
            nlu_result=nlu_result,
            memory_forest_seed=memory_forest_seed,
            llm_centric=llm_centric,
            deliberative=deliberative,
            agent_state_json=agent_state_json,
            agent_environment=agent_environment,
        )
    _pack_stats = packed_messages_stats(messages)
    _log_stats = _env_truthy("HAROMA_LLM_LOG_PACKED_STATS")
    _n_ctx = getattr(_effective_backend, "_n_ctx", None)
    _include_pi = _should_include_prompt_info(_chat_only, _use_synthetic)
    _pi = (
        _prompt_info_payload(_pack_stats, _effective_backend, chat_only=_chat_only)
        if _include_pi
        else None
    )
    if _log_stats or _use_synthetic or _chat_only:
        _log_packed_stats(_pack_stats, n_ctx=_n_ctx)

    if _use_synthetic:
        _ms = round((time.perf_counter() - _t_pack0) * 1000, 2)
        _verbose = _env_truthy("HAROMA_LLM_DUMMY_REPLY_VERBOSE")
        if _verbose:
            _msg = (
                "[dummy LLM probe] Skipped generate_chat. "
                f"Packed prompt: {_pack_stats['total_chars']} chars, "
                f"{_pack_stats['total_utf8_bytes']} UTF-8 bytes, "
                f"~{_pack_stats['est_tokens_approx']} est. tokens (chars/4), "
                f"{_pack_stats['message_count']} messages, "
                f"chars_by_role={_pack_stats['chars_by_role']!r}."
            )
        else:
            _msg = (os.environ.get("HAROMA_LLM_DUMMY_REPLY_TEXT") or "Testing reply").strip()
            if not _msg:
                _msg = "Testing reply"
        if not _verbose and not _chat_only:
            _why = (
                "HAROMA_LLM_DUMMY_REPLY=1"
                if _dummy
                else "no LLM backend"
            )
            print(
                f"[LLMContextReasoner] {_why} — skipped generate_chat; "
                f"synthetic JSON answer={_msg!r} | packed {_pack_stats['total_chars']} chars "
                f"~{_pack_stats['est_tokens_approx']} est. tok | pack+probe {_ms:.1f}ms. "
                "Set HAROMA_LLM_DUMMY_REPLY_VERBOSE=1 for long stats in answer field.",
                flush=True,
            )
        elif not _verbose and _chat_only:
            print(
                "[LLMContextReasoner] synthetic + CHAT_ONLY — plain answer "
                f"={_msg!r} | pack+probe {_ms:.1f}ms.",
                flush=True,
            )
        return _dummy_probe_result(
            answer_text=_msg,
            latency_ms=_ms,
            prompt_info=_pi,
            chat_only=_chat_only,
        )

    t0 = time.perf_counter()
    _tlim = timeout_override if timeout_override is not None else _llm_context_timeout_seconds()
    _exec_tlim = _tlim
    if _exec_tlim is None:
        _exec_tlim = _unbounded_generate_cap_seconds()
    if _exec_tlim is None:
        # Both HAROMA_LLM_CONTEXT_TIMEOUT_SEC and HAROMA_LLM_MAX_GENERATE_SEC were set
        # to unlimited — never call native llama-cpp synchronously (can hang forever).
        try:
            _fb = float(os.environ.get("HAROMA_LLM_FALLBACK_WALL_SEC", "300") or "300")
        except (TypeError, ValueError):
            _fb = 300.0
        _exec_tlim = min(7200.0, max(30.0, _fb))
        print(
            "[LLMContextReasoner] LLM wall time was unlimited — using "
            f"HAROMA_LLM_FALLBACK_WALL_SEC={_exec_tlim:.0f}s so the HTTP thread cannot "
            "hang forever in native decode. Set finite HAROMA_LLM_CONTEXT_TIMEOUT_SEC / "
            "HAROMA_LLM_MAX_GENERATE_SEC to control this explicitly.",
            flush=True,
        )

    def _do_generate():
        return _effective_backend.generate_chat(
            messages,
            max_tokens=_mt,
            temperature=temperature,
        )

    raw = None
    try:
        print(
            f"[LLMContextReasoner] generate_chat (timeout={_exec_tlim:.0f}s)…",
            flush=True,
        )
        _log_llm_payload_in(messages)
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            fut = pool.submit(_do_generate)
            raw = fut.result(timeout=_exec_tlim)
            _log_llm_payload_out(raw)
        except FutureTimeout:
            _log_llm_payload_out(None, error="llm_timeout")
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            print(
                f"[LLMContextReasoner] generate_chat timed out after "
                f"{elapsed_ms:.0f}ms (limit {_exec_tlim:.0f}s); continuing without "
                f"LLM answer. Native decode may still run in the worker — if the next "
                f"chat hangs, restart the process. Raise HAROMA_LLM_CONTEXT_TIMEOUT_SEC "
                f"or cap as needed.",
                flush=True,
            )
            result = LLMContextResult.empty("llm_timeout")
            result.latency_ms = elapsed_ms
            result.prompt_info = _pi
            return result
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    except Exception as exc:
        _log_llm_payload_out(None, error=str(exc))
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        print(
            f"[LLMContextReasoner] generate_chat error ({elapsed_ms:.0f}ms): {exc}",
            flush=True,
        )
        result = LLMContextResult.empty("llm_error")
        result.latency_ms = elapsed_ms
        result.prompt_info = _pi
        return result
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    if _chat_only:
        _plain = (raw or "").strip()
        return LLMContextResult(
            answer=_plain if _plain else None,
            confidence=0.55,
            reasoning_steps=[],
            requires_confirmation=True,
            source="chat_only",
            latency_ms=elapsed_ms,
            prompt_info=_pi,
        )

    result = parse_response(raw)
    result.latency_ms = elapsed_ms
    if _pi is not None:
        result.prompt_info = _pi
    return result
