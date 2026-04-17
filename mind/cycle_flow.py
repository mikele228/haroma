"""Cognitive cycle flow — shared phases, ordering, and helpers.

**Canonical ordering** — ``COGNITIVE_PIPELINE_STEPS`` (PHASE_BIND → PHASE_COGNITION
→ PHASE_ACT_LEARN) is the single reference for step *names* and *sequence*.
Heavy logic lives in ``ElarionController.run_cycle``; the same named phases are
reused by ``PersonaAgent`` via the helpers below (law/sidecar, curiosity,
reasoning, counterfactual, metacognition, imagination, action payload + generate);
goal synthesis and symbolic queue writes are controller-only helpers in this module.

**Two ways to run Elarion** (both valid; pick one per deployment):

1. **Embedded / library** — instantiate ``ElarionController`` and call
   ``run_cycle`` directly (no HTTP).
2. **Multi-agent (default)** — ``BootAgent`` → ``SharedResources`` (one KG/memory/reasoning
   stack) → ``TrueSelfAgent`` / ``PersonaAgent``; personas run the *same* phase
   order where implemented, minus perception already done in ``InputAgent``, with
   a **time budget** that may skip optional sub-steps.

**Observability** — Set env ``ELARION_CYCLE_TRACE=1`` or pass
``EpisodeContext(..., enable_step_trace=True)`` to record checkpoints. Controller
uses labels ``0.start``, ``5.workspace``, ``7.law_sidecar`` (via sidecar helper),
``13.2.reasoning_law``, ``14.pre_action``; personas should mirror those when the
corresponding step ran.

**Unified trace file** — ``ELARION_TRACE=1`` or ``ELARION_TRACE_FILE=<path>`` appends
one NDJSON record per controller cycle (see ``mind.cognitive_trace``).

**Ablation** — ``ELARION_ABLATION=step1,step2`` forces optional steps off; names match
``engine.ProcessGate.GATABLE_STEPS`` plus ``reconciliation`` (handled in control).

**Law / reasoning** — Tags for compliance are built once per cycle
(``build_law_tags``); reasoning may add KG laws, so law state is refreshed after
reasoning (``refresh_symbolic_law_post_reasoning`` / ``run_reasoning_phase``).
"""

from __future__ import annotations

import copy
import json as _json_mod
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from core.cognitive_null import is_cognitive_null
from mind.symbolic_sidecar import apply_derived_symbolic_sidecar

_CF_DEBUG = os.environ.get("ELARION_CYCLE_FLOW_DEBUG", "0") == "1"

# PersonaAgent neural read lock slices (see agents/persona_agent): embed/perception,
# then gate/backbone/self-model/discourse — then recall+ pipeline without neural lock.
# Single-message mid-cycle checkpoint/resume across ticks is intentionally out of scope;
# use short slices + off-lock phases + optional HAROMA_BG_MAX_TRAIN_MODULES_PER_TICK instead.
PERSONA_NEURAL_PHASES = (
    "neural_embed_perception",
    "neural_gate_discourse",
    "off_lock_recall_through_reasoning",
    "off_lock_packed_llm",
    "neural_post_llm",
)

_MAX_STATE_JSON_CHARS = max(
    512,
    min(100_000, int(os.environ.get("HAROMA_STATE_PROMPT_MAX_CHARS", "3000"))),
)


def _cf_warn(label: str, exc: Exception) -> None:
    """Print a one-line warning when ELARION_CYCLE_FLOW_DEBUG=1."""
    if _CF_DEBUG:
        print(f"  [cycle_flow] {label}: {type(exc).__name__}: {exc}", flush=True)


def _module_real(obj) -> bool:
    return obj is not None and not is_cognitive_null(obj)


# ------------------------------------------------------------------
# Deliberative state snapshot (injected into packed LLM prompt)
# ------------------------------------------------------------------


def build_trueself_state_snapshot(
    *,
    sensors: Optional[List[Dict[str, Any]]] = None,
    identity_summary: Optional[Dict[str, Any]] = None,
    personality_summary: Optional[Dict[str, float]] = None,
    active_goals: Optional[List[Dict[str, Any]]] = None,
    law_summary: Optional[Dict[str, Any]] = None,
    value_summary: Optional[Dict[str, Any]] = None,
    drives: Optional[Dict[str, Any]] = None,
    affect: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a compact JSON-serialisable state snapshot for the deliberative LLM prompt.

    Each section is optional; missing keys are simply omitted.
    """
    snap: Dict[str, Any] = {}

    if sensors:
        snap["sensors"] = sensors[:32]

    if identity_summary:
        snap["identity"] = identity_summary

    if personality_summary:
        snap["personality"] = {k: round(v, 3) for k, v in personality_summary.items()}

    if active_goals:
        snap["goals"] = [
            {
                "goal_id": g.get("goal_id", g.get("id", "")),
                "description": str(g.get("description", ""))[:200],
                "priority": g.get("priority", 0.5),
            }
            for g in active_goals[:12]
        ]

    if law_summary:
        snap["laws"] = law_summary

    if value_summary:
        snap["values"] = value_summary

    if drives:
        snap["drives"] = drives

    if affect:
        snap["affect"] = affect

    return snap


def serialize_state_snapshot(snap: Dict[str, Any]) -> str:
    """Serialize a state snapshot to a compact JSON string, capped for prompt size."""
    raw = _json_mod.dumps(snap, ensure_ascii=False, default=str)
    if len(raw) > _MAX_STATE_JSON_CHARS:
        raw = raw[:_MAX_STATE_JSON_CHARS] + "..."
    return raw


# Ordered phases for documentation / tracing (mirror mind/control.py sections)
PHASE_BIND = (
    "0.working_memory",
    "1.perception",
    "1.4.embed",
    "1.45.gate",
    "1.46.backbone",
    "1.5.self_prediction",
    "1.6.nlu",
    "1.65.discourse",
    "1.7.interlocutor",
    "2.recall",
    "2.5.knowledge_graph",
    "3.emotion",
    "4-5.workspace",
    "6.narrative",
    "7.law_value_myth_fusion",
    "8.dream",
    "8.5.reconciliation",
)

PHASE_COGNITION = (
    "9.identity",
    "10.goals",
    "10.5.drives",
    "10.7.appraisal",
    "10.8.modulation",
    "11-12.reflect",
    "13.curiosity",
    "13.2.reasoning",
    "13.2b.law_refresh",
    "13.3.counterfactual",
    "13.5.metacognition",
    "temporal",
    "13.7.imagination",
    "13.8.goal_synthesis",
    "13.9.symbolic_queue",
)

PHASE_ACT_LEARN = (
    "14.action",
    "15.evaluate",
    "15.x.learning",
    "16.consolidate",
)

COGNITIVE_PIPELINE_STEPS = PHASE_BIND + PHASE_COGNITION + PHASE_ACT_LEARN


def build_law_tags(
    symbolic_input: Dict[str, Any],
    payload: Dict[str, Any],
) -> List[str]:
    """Union of perception tags and optional ``law_tags`` on input / message payload."""
    law_tags: List[str] = [str(t) for t in (symbolic_input.get("tags") or []) if t is not None]
    if "law_tags" in payload:
        extra = payload["law_tags"]
        if isinstance(extra, str):
            law_tags.append(extra)
        elif isinstance(extra, (list, tuple, set)):
            law_tags.extend(str(x) for x in extra if x is not None)
    return law_tags


def run_law_value_myth_sidecar_phase(
    *,
    gate_enabled: bool,
    law,
    law_tags: List[str],
    episode: Any,
    workspace: Any,
    attention: Any,
    attention_ctx: Dict[str, Any],
    z_t: Any,
    value,
    myth,
    fusion,
    dream_mgr,
    symbolic_input: Dict[str, Any],
    explicit: Dict[str, Any],
    role: str,
    has_external: bool,
    broadcast_violations: bool = True,
    apply_explicit_bindings: bool = True,
    skip_derived_value_myth_fusion: bool = False,
    trace_label: Optional[str] = None,
) -> None:
    """Symbolic law check, episode bind, optional workspace broadcast, explicit
    value/myth/fusion, then derived sidecar — shared by controller and persona."""
    if not gate_enabled:
        return
    violations: List[Dict[str, Any]] = []
    if _module_real(law):
        try:
            raw = law.check(law_tags)
            if isinstance(raw, list):
                violations = raw
        except Exception as _e:
            _cf_warn("law.check", _e)
            violations = []
    episode.bind_symbolic_law(
        {
            "tags_checked": law_tags,
            "violations": violations,
            "compliant": len(violations) == 0,
        }
    )
    if violations and broadcast_violations:
        try:
            max_sev = max(float(v.get("severity", 1.0)) for v in violations)
            workspace.broadcast(
                "law",
                {"violations": violations, "tags": law_tags},
                salience=attention.adjust_salience(
                    "law", min(1.0, max_sev), attention_ctx, z_t=z_t
                ),
            )
        except Exception as _e:
            _cf_warn("law.broadcast", _e)
    if apply_explicit_bindings:
        if "value" in explicit and _module_real(value):
            weight = 1.0 + episode.affect["intensity"] * 0.5
            value.reinforce_value(explicit["value"], weight=weight)
        if "myth" in explicit and _module_real(myth):
            myth.bind(explicit["myth"], anchor=role)
        ft = explicit.get("fusion_targets")
        if ft and _module_real(fusion):
            if isinstance(ft, (list, tuple)) and len(ft) >= 2:
                fusion.fuse(ft[0], ft[1])
    apply_derived_symbolic_sidecar(
        value=value,
        myth=myth,
        fusion=fusion,
        dream_mgr=dream_mgr,
        episode=episode,
        symbolic_input=symbolic_input,
        explicit=explicit,
        role=role,
        has_external=bool(has_external),
        skip_derived_value_myth_fusion=skip_derived_value_myth_fusion,
    )
    if trace_label:
        try:
            episode.trace(trace_label)
        except Exception as _e:
            _cf_warn("episode.trace(law_sidecar)", _e)


EMPTY_CURIOSITY_RESULT: Dict[str, Any] = {
    "curiosity_score": 0.0,
    "novelty": 0.0,
    "prediction_error": 0.0,
    "questions": [],
}


def refresh_symbolic_law_post_reasoning(
    law,
    episode,
    law_tags: List[str],
    workspace,
    attention,
    attention_ctx,
    z_t,
    gate_law: bool,
    gate_reasoning: bool,
) -> None:
    """Re-run compliance after reasoning — e.g. KG may have added ``forbids_tag`` laws."""
    if not gate_law or not gate_reasoning:
        return
    if law is None or is_cognitive_null(law):
        return
    try:
        raw = law.check(law_tags)
    except Exception as _e:
        _cf_warn("law.check(post_reasoning)", _e)
        return
    violations = raw if isinstance(raw, list) else []
    episode.bind_symbolic_law(
        {
            "tags_checked": law_tags,
            "violations": violations,
            "compliant": len(violations) == 0,
            "post_reasoning_refresh": True,
        }
    )
    if violations:
        max_sev = max(
            (float(v.get("severity", 1.0)) for v in violations if isinstance(v, dict)),
            default=1.0,
        )
        try:
            workspace.broadcast(
                "law",
                {"violations": violations, "tags": law_tags, "post_reasoning": True},
                salience=attention.adjust_salience(
                    "law", min(1.0, max_sev), attention_ctx, z_t=z_t
                ),
            )
        except Exception as _e:
            _cf_warn("law.broadcast(post_reasoning)", _e)


def run_curiosity_phase(
    *,
    enabled: bool,
    episode: Any,
    curiosity: Any,
    emotion_summary: Dict[str, Any],
    knowledge_summary: Dict[str, Any],
    current_embedding: Any,
    last_strategy: str,
    forecast_for_eval: Dict[str, Any],
    knowledge_graph: Any = None,
    max_kg_gaps: int = 3,
    goal_manager: Any = None,
    workspace: Any = None,
    attention: Any = None,
    attention_ctx: Optional[Dict[str, Any]] = None,
    z_t: Any = None,
    modulation: Optional[Dict[str, Any]] = None,
    workspace_followup: bool = False,
) -> Dict[str, Any]:
    """Curiosity evaluate + episode bind; optional KG gap questions, goal harvest,
    and workspace broadcast/select (controller path)."""
    if not enabled or not _module_real(curiosity):
        return dict(EMPTY_CURIOSITY_RESULT)
    ks = dict(knowledge_summary)
    if knowledge_graph is not None and _module_real(knowledge_graph):
        try:
            kg_gaps = knowledge_graph.find_gaps(max_gaps=max_kg_gaps)
        except Exception as _e:
            _cf_warn("knowledge_graph.find_gaps", _e)
            kg_gaps = []
        ks["gap_count"] = len(kg_gaps)
    try:
        curiosity_result = curiosity.evaluate(
            episode.to_summary(),
            episode.recalled_memories,
            forecast_for_eval,
            emotion_summary,
            knowledge_summary=ks,
            current_embedding=current_embedding,
            last_strategy=last_strategy,
        )
    except Exception as _e:
        _cf_warn("curiosity.evaluate", _e)
        return dict(EMPTY_CURIOSITY_RESULT)
    qs = list(curiosity_result.get("questions", []))
    curiosity_result["questions"] = qs[:5]
    episode.bind_curiosity(
        novelty=curiosity_result.get("novelty", 0.0),
        uncertainty=curiosity_result.get("prediction_error", 0.0),
        questions=curiosity_result.get("questions", []),
    )
    if goal_manager is not None and _module_real(goal_manager):
        for cgoal in curiosity_result.get("generated_goals", []):
            try:
                goal_manager.register_goal(
                    cgoal["goal_id"],
                    cgoal["description"],
                    priority=cgoal["priority"],
                    source="curiosity",
                )
            except Exception as _e:
                _cf_warn("goal_manager.register(curiosity)", _e)
    if (
        workspace_followup
        and workspace is not None
        and attention is not None
        and attention_ctx is not None
    ):
        mod = modulation or {}
        damping = mod.get("curiosity_damping", 1.0)
        damped = curiosity_result.get("curiosity_score", 0.0) * damping
        try:
            workspace.broadcast(
                "curiosity",
                curiosity_result,
                salience=attention.adjust_salience("curiosity", damped, attention_ctx, z_t=z_t),
            )
            workspace.select()
            episode.bind_workspace(
                workspace.get_contents(),
                workspace.get_unconscious(),
            )
        except Exception as _e:
            _cf_warn("curiosity.workspace_followup", _e)
    return curiosity_result


def run_reasoning_phase(
    *,
    enabled: bool,
    reasoning_engine: Any,
    knowledge: Any,
    active_goals: List[Any],
    nlu_result: Dict[str, Any],
    max_depth: Any,
    memory: Any,
    episode: Any,
    law: Any,
    law_tags: List[str],
    workspace: Any,
    attention: Any,
    attention_ctx: Dict[str, Any],
    z_t: Any,
    gate_law_value_myth: bool,
    gate_reasoning_for_refresh: bool,
    trace_label: Optional[str] = "13.2.reasoning_law",
) -> Dict[str, Any]:
    """Reasoning + post-reasoning symbolic law refresh + optional trace."""
    out: Dict[str, Any] = {"reasoning_depth": 0}
    if not enabled or not _module_real(reasoning_engine):
        return out
    reasoning_result = reasoning_engine.reason(
        knowledge,
        active_goals,
        nlu_result,
        max_depth=max_depth,
        memory=memory,
    )
    episode.bind_reasoning(reasoning_result)
    if reasoning_result.get("reasoning_depth", 0) > 0:
        try:
            workspace.broadcast(
                "reasoning",
                reasoning_result,
                salience=attention.adjust_salience(
                    "reasoning",
                    min(1.0, reasoning_result["reasoning_depth"] * 0.15),
                    attention_ctx,
                    z_t=z_t,
                ),
            )
            workspace.select()
        except Exception as _e:
            _cf_warn("reasoning.workspace_broadcast", _e)
    refresh_symbolic_law_post_reasoning(
        law,
        episode,
        law_tags,
        workspace,
        attention,
        attention_ctx,
        z_t,
        gate_law_value_myth,
        gate_reasoning_for_refresh,
    )
    if trace_label:
        try:
            episode.trace(trace_label)
        except Exception as _e:
            _cf_warn("episode.trace(reasoning)", _e)
    return reasoning_result


# -- Organic confidence gate for packed-context LLM --------------------


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "")
    if raw:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    return default


ORGANIC_PACKED_LLM_SKIP_THRESHOLD: float = _env_float(
    "HAROMA_ORGANIC_LLM_SKIP",
    0.9,
)


def organic_confidence(
    recalled_memories: list,
    reasoning_result: Optional[Dict[str, Any]] = None,
    appraisal_result: Optional[Dict[str, Any]] = None,
) -> float:
    """Non-LLM confidence for deciding whether to skip the packed-context LLM.

    Uses only signals that indicate **structured reasoning already settled**
    the turn — not recall salience (memory nodes often use confidence=1.0
    for storage, which wrongly skipped the LLM and yielded empty replies).

    Combines: best reasoning inference confidence, and appraisal *relevance*
    (goal/input salience).  ``norm_compatibility`` is excluded: it reflects
    identity fit, not answer readiness.

    ``recalled_memories`` is accepted for API stability but **ignored** here.

    When the score is at or above ``ORGANIC_PACKED_LLM_SKIP_THRESHOLD``
    (default 0.9) the packed-context LLM pass may be skipped.
    """

    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    rr = reasoning_result or {}
    max_inf = 0.0
    for inf in rr.get("inferences", []):
        try:
            max_inf = max(max_inf, float(inf.get("confidence", 0.0)))
        except (TypeError, ValueError):
            pass

    ap = appraisal_result or {}
    ap_relevance = 0.0
    try:
        ap_relevance = float(ap.get("relevance", 0.0))
    except (TypeError, ValueError):
        pass

    return _clamp(max(_clamp(max_inf), _clamp(ap_relevance)))


def run_llm_context_reasoning_phase(
    *,
    enabled: bool,
    llm_backend: Any,
    user_text: str,
    recalled_memories: list,
    identity_summary: Dict[str, Any],
    personality_summary: Dict[str, float],
    active_goals: List[Any],
    law_summary: Dict[str, Any],
    value_summary: Dict[str, Any],
    knowledge_triples: Optional[List[Any]] = None,
    discourse_context: str = "",
    nlu_result: Optional[Dict[str, Any]] = None,
    memory_forest_seed: str = "",
    llm_centric: bool = False,
    episode: Any = None,
    memory_forest: Any = None,
    trace_label: Optional[str] = "13.3.llm_context_reasoning",
    timeout_override: Optional[float] = None,
    deliberative: bool = False,
    agent_state_json: str = "",
    bind_episode: bool = True,
    agent_environment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Packed-context LLM reasoning pass, gated by *enabled*.

    Produces a ``LLMContextResult`` dict with ``answer``, ``confidence``,
    ``inferences``, and ``cited_memories``.  Provisional memory nodes are
    written when the result has inferences with ``confidence >= 0.4``.

    Set ``bind_episode=False`` when the caller will merge extra fields (e.g.
    deliberative scores) and call ``episode.bind_llm_context`` once — avoids
    double-binding the same cycle payload.
    """
    empty: Dict[str, Any] = {"source": "skipped"}

    def _return_skipped(payload: Dict[str, Any]) -> Dict[str, Any]:
        if episode is not None and hasattr(episode, "bind_llm_context"):
            episode.bind_llm_context(payload)
        return payload

    if not enabled:
        return _return_skipped(dict(empty))

    try:
        from mind.cognitive_contracts import run_llm_context_reasoning
    except Exception as _imp_err:
        print(
            f"[cycle_flow] cognitive_contracts / packed LLM import failed: {_imp_err}",
            flush=True,
        )
        return _return_skipped(dict(empty))

    _ae = agent_environment
    if _ae is None and episode is not None:
        _meta_fn = getattr(episode, "environment_for_training_metadata", None)
        if callable(_meta_fn):
            _ae = _meta_fn()
        else:
            _ae = getattr(episode, "agent_environment", None)
    if not isinstance(_ae, dict):
        _ae = None

    result = run_llm_context_reasoning(
        llm_backend=llm_backend,
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
        agent_environment=_ae,
        timeout_override=timeout_override,
    )
    result_dict = result.to_dict()

    # Bind to episode (callers may set ``bind_episode=False`` and bind once after
    # merging deliberative scores to avoid double ``bind_llm_context``).
    if (
        bind_episode
        and episode is not None
        and hasattr(episode, "bind_llm_context")
    ):
        episode.bind_llm_context(result_dict)

    # Provisional memory writes for confident inferences
    if memory_forest is not None and hasattr(memory_forest, "add_node"):
        from core.Memory import MemoryNode

        for inf in result.inferences:
            if inf.get("confidence", 0) >= 0.4:
                node = MemoryNode(
                    content=(
                        f"[llm_hypothesis] {inf['subject']} "
                        f"--[{inf['predicate']}]--> {inf['object']} "
                        f"(conf={inf['confidence']:.2f})"
                    ),
                    emotion="neutral",
                    confidence=inf["confidence"] * 0.7,
                    tags=[
                        "llm_hypothesis",
                        "provisional",
                        inf.get("source", "llm_context_reasoning"),
                    ],
                )
                try:
                    memory_forest.add_node("thought_tree", "llm_hypotheses", node)
                except Exception:
                    pass

    if trace_label and episode is not None:
        try:
            episode.trace(trace_label)
        except Exception:
            pass

    return result_dict


def build_counterfactual_gate_features(
    knowledge_diff: Dict[str, Any],
    reasoning_result: Dict[str, Any],
    active_goals: List[Any],
    emotion_summary: Dict[str, Any],
    curiosity_result: Dict[str, Any],
    has_external: bool,
    cycle_count: int,
    counterfactual_engine: Any,
    prev_modulation: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """Feature vector for ``CounterfactualEngine.gate`` (controller + persona)."""
    prev_modulation = prev_modulation or {}
    hist = getattr(counterfactual_engine, "_history", None) or []
    tail = hist[-10:] if hist else []
    depth_avg = sum(float(h.get("counterfactual_depth", 0)) for h in tail) / max(len(tail), 1)
    cc = max(0, int(cycle_count))
    return [
        float(knowledge_diff.get("new_relations", 0)),
        float(knowledge_diff.get("new_entities", 0)),
        float(reasoning_result.get("reasoning_depth", 0)),
        float(len(active_goals)),
        float(emotion_summary.get("arousal", 0.0)),
        float(emotion_summary.get("valence", 0.0)),
        float(curiosity_result.get("curiosity_score", 0.0)),
        1.0 if has_external else 0.0,
        float(math.log1p(cc)),
        depth_avg,
        float(len(hist)),
        float(prev_modulation.get("novelty_bias", 0.0)),
    ]


def run_counterfactual_phase(
    *,
    enabled: bool,
    counterfactual_engine: Any,
    knowledge_graph: Any,
    reasoning_engine: Any,
    reasoning_result: Dict[str, Any],
    knowledge_diff: Dict[str, Any],
    active_goals: List[Any],
    nlu_result: Optional[Dict[str, Any]],
    episode: Any,
    gate_features: Optional[List[float]] = None,
    workspace: Any = None,
    attention: Any = None,
    attention_ctx: Optional[Dict[str, Any]] = None,
    z_t: Any = None,
    workspace_broadcast: bool = True,
) -> Dict[str, Any]:
    """Counterfactual evaluate with optional learned gate; workspace broadcast when depth > 0."""
    default: Dict[str, Any] = {"counterfactual_depth": 0, "branches": []}
    if not enabled or not _module_real(counterfactual_engine):
        return dict(default)
    gate_decision: Dict[str, Any] = {}
    if gate_features is not None:
        try:
            gate_decision = counterfactual_engine.gate(gate_features)
        except Exception as _e:
            _cf_warn("counterfactual.gate", _e)
            gate_decision = {}
    counterfactual_result = counterfactual_engine.evaluate(
        knowledge_graph=knowledge_graph,
        reasoning_engine=reasoning_engine,
        reasoning_result=reasoning_result,
        knowledge_diff=knowledge_diff,
        active_goals=active_goals,
        nlu_result=nlu_result,
        gate_decision=gate_decision,
    )
    episode.bind_counterfactual(counterfactual_result)
    if (
        workspace_broadcast
        and counterfactual_result.get("counterfactual_depth", 0) > 0
        and workspace is not None
        and attention is not None
        and attention_ctx is not None
    ):
        try:
            depth = counterfactual_result["counterfactual_depth"]
            workspace.broadcast(
                "counterfactual",
                counterfactual_result,
                salience=attention.adjust_salience(
                    "counterfactual", min(1.0, float(depth) * 0.25), attention_ctx, z_t=z_t
                ),
            )
            workspace.select()
        except Exception as _e:
            _cf_warn("counterfactual.workspace_broadcast", _e)
    return counterfactual_result


def run_metacognition_phase(
    *,
    enabled: bool,
    extended: bool,
    metacognition: Any,
    episode: Any,
    emotion_summary: Dict[str, Any],
    curiosity_result: Dict[str, Any],
    outcome_prev: Dict[str, Any],
    cycle_count: Optional[int] = None,
    prev_meta_assessment: Optional[Dict[str, Any]] = None,
    self_surprise: Optional[Dict[str, Any]] = None,
    controller_for_inspection: Any = None,
    temporal_engine: Any = None,
    goal_manager: Any = None,
    workspace: Any = None,
    attention: Any = None,
    attention_ctx: Optional[Dict[str, Any]] = None,
    z_t: Any = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Assess + bind meta. ``extended`` adds self-inspection, temporal concerns,
    meta-goals, workspace broadcast (controller)."""
    if not enabled or not _module_real(metacognition):
        return {}, None
    self_inspection: Optional[Dict[str, Any]] = None
    if extended and cycle_count is not None and controller_for_inspection is not None:
        try:
            if metacognition.should_inspect(cycle_count):
                self_inspection = metacognition.inspect_self(
                    assessment=prev_meta_assessment or {},
                    self_surprise=self_surprise or {},
                    controller=controller_for_inspection,
                )
        except Exception as _e:
            _cf_warn("self_model.inspect", _e)
            self_inspection = None
    meta_assessment = metacognition.assess(
        episode.to_payload(),
        emotion_summary,
        curiosity_result,
        outcome_prev,
        self_inspection=self_inspection,
    )
    episode.bind_meta(meta_assessment)
    if extended and temporal_engine is not None and _module_real(temporal_engine):
        try:
            repetition = temporal_engine.detect_repetition()
            if repetition:
                meta_assessment.setdefault("concerns", []).append(f"Temporal: {repetition}")
        except Exception as _e:
            _cf_warn("temporal.detect_repetition", _e)
    if extended and self_inspection:
        for adj in self_inspection.get("recommended_adjustments", []):
            if adj == "increase_novelty_bias":
                meta_assessment.setdefault("concerns", []).append(
                    "[self-inspection] Increasing novelty-seeking"
                )
            elif adj == "increase_self_model_training":
                meta_assessment.setdefault("concerns", []).append(
                    "[self-inspection] Self-model needs more training"
                )
    if extended and goal_manager is not None and _module_real(goal_manager):
        meta_goals = metacognition.generate_meta_goals(meta_assessment)
        if self_inspection:
            for adj in self_inspection.get("recommended_adjustments", []):
                meta_goals.append(
                    {
                        "goal_id": f"self_inspection_{adj}",
                        "description": f"Self-inspection recommends: {adj}",
                        "priority": 0.5,
                        "source": "self_inspection",
                    }
                )
        for mg in meta_goals:
            try:
                goal_manager.register_goal(
                    mg["goal_id"],
                    mg["description"],
                    priority=mg["priority"],
                    source=mg["source"],
                )
            except Exception as _e:
                _cf_warn("goal_manager.register(meta)", _e)
        n_concerns = len(meta_assessment.get("concerns", []))
        if (
            n_concerns > 0
            and workspace is not None
            and attention is not None
            and attention_ctx is not None
        ):
            try:
                workspace.broadcast(
                    "metacognition",
                    meta_assessment,
                    salience=attention.adjust_salience(
                        "metacognition", min(1.0, n_concerns * 0.3), attention_ctx, z_t=z_t
                    ),
                )
                workspace.select()
            except Exception as _e:
                _cf_warn("metacognition.workspace_broadcast", _e)
    return meta_assessment, self_inspection


def _active_goals_as_dicts(active_goals: List[Any], limit: int = 5) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for g in active_goals[:limit]:
        if hasattr(g, "to_dict"):
            out.append(g.to_dict())
        elif isinstance(g, dict):
            out.append(g)
        else:
            out.append({})
    return out


def run_imagination_phase(
    *,
    enabled: bool,
    imagination: Any,
    episode: Any,
    current_embedding: Any,
    emotion_summary: Dict[str, Any],
    curiosity_result: Dict[str, Any],
    dominant_drive: float,
    wm_load: float,
    outcome_prev: float,
    has_external: float,
    active_goals: List[Any],
    drive_state: Dict[str, Any],
    temporal_engine: Any = None,
    temporal_enrichment: bool = False,
) -> Tuple[Dict[str, Any], Optional[str], List[str]]:
    """Simulate, bind imagination, return (result, strategy_hint, imagined_plan)."""
    empty: Dict[str, Any] = {"scenarios": []}
    if not enabled or not _module_real(imagination):
        return dict(empty), None, []
    recent_emos: Optional[List[str]] = None
    repetition_flag: Optional[str] = None
    if temporal_enrichment and temporal_engine is not None and _module_real(temporal_engine):
        try:
            recent_temporal = (
                temporal_engine.get_recent_sequence(5)
                if hasattr(temporal_engine, "get_recent_sequence")
                else []
            )
            recent_emos = [
                s.get("emotion", "neutral") if isinstance(s, dict) else "neutral"
                for s in recent_temporal
            ]
            if not recent_emos:
                recent_emos = None
            rep = (
                temporal_engine.detect_repetition()
                if hasattr(temporal_engine, "detect_repetition")
                else None
            )
            if isinstance(rep, dict) and rep.get("detected"):
                repetition_flag = str(rep.get("type", "") or "")
        except Exception as _e:
            _cf_warn("imagination.pre_simulate_context", _e)
            recent_emos, repetition_flag = None, None
    goals_norm = _active_goals_as_dicts(active_goals)
    try:
        imagination_result = imagination.simulate(
            content_embedding=current_embedding,
            valence=float(emotion_summary.get("valence", 0.0)),
            arousal=float(emotion_summary.get("arousal", 0.0)),
            curiosity=float(curiosity_result.get("curiosity_score", 0.0)),
            dominant_drive=float(dominant_drive),
            wm_load=float(wm_load),
            outcome_prev=float(outcome_prev),
            has_external=float(has_external),
            recalled_memories=episode.recalled_memories,
            curiosity_questions=curiosity_result.get("questions", []),
            recent_emotions=recent_emos,
            repetition_flag=repetition_flag,
            emotion=emotion_summary,
            goals=goals_norm,
            drives=drive_state,
        )
    except Exception as _e:
        _cf_warn("imagination.simulate", _e)
        imagination_result = dict(empty)
    episode.bind_imagination(imagination_result)
    imagined_strategy: Optional[str] = None
    try:
        imagined_strategy = imagination.get_strategy_recommendation()
    except Exception as _e:
        _cf_warn("imagination.get_strategy_recommendation", _e)
    raw_plan = imagination_result.get("imagined_plan", []) or []
    imagined_plan = list(raw_plan) if raw_plan else []
    return imagination_result, imagined_strategy, imagined_plan


def run_goal_synthesis_phase(
    *,
    enabled: bool,
    goal_synthesizer: Any,
    goal_manager: Any,
    valence: float,
    arousal: float,
    curiosity_score: float,
    prediction_error: float,
    dominant_drive_level: float,
    wm_load: float,
    drift_score: float,
    outcome_prev: float,
    has_external: float,
    knowledge_entity_count: int,
    goal_count: int,
    cycle_count: int,
    z_t: Any,
    active_goals: List[Any],
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Learned goal discovery; registers new goals. Returns synthesized list + ctx."""
    if not enabled or not _module_real(goal_synthesizer) or not _module_real(goal_manager):
        return [], []
    try:
        synth_ctx = goal_synthesizer._build_context(
            valence=valence,
            arousal=arousal,
            curiosity=curiosity_score,
            prediction_error=prediction_error,
            dominant_drive_level=dominant_drive_level,
            wm_load=wm_load,
            drift_score=drift_score,
            outcome_prev=outcome_prev,
            has_external=has_external,
            knowledge_entity_count=knowledge_entity_count,
            goal_count=goal_count,
            cycle_count=cycle_count,
        )
        existing_goal_ids = [g.get("goal_id", "") for g in active_goals]
        synthesized = goal_synthesizer.synthesize(
            synth_ctx, existing_goal_ids, cycle_count, z_t=z_t
        )
        for sg in synthesized:
            goal_manager.register_goal(
                sg["goal_id"],
                sg["description"],
                priority=sg["priority"],
                source=sg["source"],
            )
        return synthesized, synth_ctx
    except Exception as _e:
        _cf_warn("goal_synthesis", _e)
        return [], []


def write_processor_symbolic_queue(
    symbolic_queue: Any,
    fingerprint_engine: Any,
    emotion_summary: Dict[str, Any],
    reasoning_result: Dict[str, Any],
    meta_assessment: Dict[str, Any],
) -> None:
    """Push emotion / reasoning / metacognition snapshots through the X7 queue."""
    if not _module_real(symbolic_queue) or not _module_real(fingerprint_engine):
        return
    try:
        from core.SymbolicQueue import SymbolicQueue

        processors = {
            "emotion": emotion_summary,
            "reasoning": reasoning_result,
            "metacognition": meta_assessment,
        }
        for mod_name, output in processors.items():
            if output:
                symbolic_queue.write(
                    SymbolicQueue.SLOT_PROCESSOR, f"processor.{mod_name}", "result", output
                )
                fingerprint_engine.is_novel(mod_name, output)
    except Exception as _e:
        _cf_warn("symbolic_queue_write", _e)


def workspace_contents_as_dicts(workspace: Any) -> List[Any]:
    """Snapshot ``GlobalWorkspace.get_contents()`` as plain dicts."""
    return [c.to_dict() if hasattr(c, "to_dict") else c for c in workspace.get_contents()]


def resolve_strategy_hint(
    memory_hint: Optional[str],
    imagined_strategy: Optional[str],
    current_plan: Optional[List[str]] = None,
    plan_step: int = 0,
) -> Optional[str]:
    """Memory suggestion, then imagination fill-in, then active plan step (if any)."""
    hint: Optional[str] = memory_hint
    if imagined_strategy and not hint:
        hint = imagined_strategy
    if current_plan and 0 <= int(plan_step) < len(current_plan):
        hint = current_plan[int(plan_step)]
    return hint


def build_action_episode_payload(
    episode: Any,
    current_embedding: Any,
    z_t: Any,
    knowledge_graph: Any,
    knowledge_summary: Dict[str, Any],
    nlu_result: Optional[Dict[str, Any]],
    environment_context: Optional[Dict[str, Any]] = None,
    memory_forest: Any = None,
) -> Tuple[Dict[str, Any], List[Any]]:
    """Augment ``episode.to_payload()`` for action generation (+ KG triples)."""
    from engine.LanguageComposer import LanguageComposer

    ep_payload = episode.to_payload()
    ep_payload["_content_embedding"] = current_embedding
    ep_payload["_z_t"] = z_t
    if environment_context is not None:
        ep_payload["environment_context"] = environment_context
    nlu_ent_names = [
        e.get("text", "") for e in (nlu_result.get("entities", []) if nlu_result else [])
    ]
    kg_triples = LanguageComposer.select_relevant_triples(
        knowledge_summary, knowledge_graph, nlu_ent_names
    )
    ep_payload["_knowledge_triples"] = kg_triples

    supplement: List[str] = []
    if memory_forest is not None and hasattr(memory_forest, "get_nodes"):
        for tree_name, branch in (
            ("thought_tree", "web_learn"),
            ("thought_tree", "autonomy"),
        ):
            try:
                nodes = memory_forest.get_nodes(tree_name, branch)
                pick = nodes[-3:] if len(nodes) > 3 else nodes
                for n in pick:
                    c = (getattr(n, "content", None) or "").strip()
                    if len(c) > 30:
                        supplement.append(c[:950])
            except Exception as _e:
                _cf_warn(f"memory_forest.get_nodes({tree_name}/{branch})", _e)
    if supplement:
        ep_payload["_llm_supplement"] = supplement

    return ep_payload, kg_triples


def run_deliberative_action(
    *,
    episode: Any,
    action_generator: Any,
    ep_payload: Dict[str, Any],
    ws_dicts: List[Any],
    strategy_hint: Optional[str],
    working_memory_context: str,
    conversation_context: str,
    is_in_conversation: bool,
    topic: str,
    last_input_content: str,
    topic_shifted: bool,
    knowledge_summary: Dict[str, Any],
    reasoning_result: Dict[str, Any],
    nlu_result: Optional[Dict[str, Any]],
    interlocutor: Dict[str, Any],
    counterfactual_result: Dict[str, Any],
    novelty_bias: float,
    personality: Optional[Dict[str, Any]] = None,
    utterance_style: Optional[str] = None,
    trace_pre_action: bool = True,
) -> Dict[str, Any]:
    """Trace ``14.pre_action`` and call ``ActionGenerator.generate``."""
    if trace_pre_action:
        try:
            episode.trace("14.pre_action")
        except Exception as _e:
            _cf_warn("episode.trace(pre_action)", _e)
    kwargs: Dict[str, Any] = {
        "working_memory_context": working_memory_context,
        "conversation_context": conversation_context,
        "is_in_conversation": is_in_conversation,
        "topic": topic,
        "last_input_content": last_input_content,
        "topic_shifted": topic_shifted,
        "knowledge_summary": knowledge_summary,
        "reasoning_result": reasoning_result,
        "nlu_result": nlu_result,
        "interlocutor": interlocutor,
        "counterfactual_result": counterfactual_result,
        "novelty_bias": novelty_bias,
    }
    if personality is not None:
        kwargs["personality"] = personality
    if utterance_style is not None:
        kwargs["utterance_style"] = utterance_style
    return action_generator.generate(ep_payload, ws_dicts, strategy_hint, **kwargs)


def run_multi_goal_deliberative_actions(
    *,
    episode: Any,
    action_generator: Any,
    ep_payload: Dict[str, Any],
    goal_batch: List[Dict[str, Any]],
    max_actions_per_goal: int,
    ws_dicts: List[Any],
    strategy_hint: Optional[str],
    working_memory_context: str,
    conversation_context: str,
    is_in_conversation: bool,
    topic: str,
    last_input_content: str,
    topic_shifted: bool,
    knowledge_summary: Dict[str, Any],
    reasoning_result: Dict[str, Any],
    nlu_result: Optional[Dict[str, Any]],
    interlocutor: Dict[str, Any],
    counterfactual_result: Dict[str, Any],
    novelty_bias: float,
    personality: Optional[Dict[str, Any]] = None,
    utterance_style: Optional[str] = None,
    trace_pre_action: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Deliberate once per goal in *goal_batch*; each goal yields 0..N actions.

    Returns ``(fused_action, groups)`` where *groups* is
    ``[{"goal_id": str, "actions": [action_dict, ...]}, ...]`` and
    *fused_action* merges non-empty texts for chat / outcome evaluation.
    """
    if trace_pre_action:
        try:
            episode.trace("14.pre_action")
        except Exception as _e:
            _cf_warn("episode.trace(pre_action)", _e)

    ag_kwargs: Dict[str, Any] = {
        "working_memory_context": working_memory_context,
        "conversation_context": conversation_context,
        "is_in_conversation": is_in_conversation,
        "topic": topic,
        "last_input_content": last_input_content,
        "topic_shifted": topic_shifted,
        "knowledge_summary": knowledge_summary,
        "reasoning_result": reasoning_result,
        "nlu_result": nlu_result,
        "interlocutor": interlocutor,
        "counterfactual_result": counterfactual_result,
        "novelty_bias": novelty_bias,
    }
    if personality is not None:
        ag_kwargs["personality"] = personality
    if utterance_style is not None:
        ag_kwargs["utterance_style"] = utterance_style

    mapg = max(1, int(max_actions_per_goal))
    groups: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    fused_base: Optional[Dict[str, Any]] = None

    if not goal_batch:
        act = action_generator.generate(ep_payload, ws_dicts, strategy_hint, **ag_kwargs)
        return act, []

    for g in goal_batch:
        if not isinstance(g, dict):
            continue
        pl = copy.deepcopy(ep_payload)
        pl["active_goals"] = [g]
        gid = str(g.get("goal_id", "") or "")
        if mapg <= 1:
            acts = [
                action_generator.generate(pl, ws_dicts, strategy_hint, **ag_kwargs),
            ]
        else:
            acts = action_generator.generate_multi_actions(
                pl,
                ws_dicts,
                strategy_hint,
                max_actions=mapg,
                **ag_kwargs,
            )
        stamped: List[Dict[str, Any]] = []
        for a in acts:
            if not isinstance(a, dict):
                continue
            ad = dict(a)
            if gid:
                ad["source_goal_id"] = gid
            stamped.append(ad)
            tx = (ad.get("text") or "").strip()
            if tx:
                text_parts.append(tx)
                if fused_base is None:
                    fused_base = ad
        groups.append({"goal_id": gid, "actions": stamped})

    if fused_base is None:
        fused = action_generator.generate(ep_payload, ws_dicts, strategy_hint, **ag_kwargs)
        fused = dict(fused)
        fused["multi_goal_groups"] = groups
        return fused, groups

    fused = dict(fused_base)
    fused["text"] = "\n\n".join(text_parts) if text_parts else fused.get("text")
    fused["multi_goal_groups"] = groups
    fused["strategy"] = "multi_goal"
    prev_r = str(fused.get("reasoning") or "")
    fused["reasoning"] = prev_r + " | multi_goal_cycle"
    return fused, groups
