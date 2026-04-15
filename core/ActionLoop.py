"""
Action-Outcome Loop for HaromaX6.

The system generates a text response each cycle based on its internal
state (emotion, goals, curiosity, narrative) and evaluates the quality
of that response.  Over time it learns which strategies work in which
contexts.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import hashlib
import os
import re
import time
import random
import threading


def _env_chat_llm_primary() -> bool:
    """When true, conversant HTTP turns run packed-context LLM and prefer its reply."""
    return str(os.environ.get("HAROMA_CHAT_LLM_PRIMARY", "1") or "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _env_disable_fast_template_composition() -> bool:
    """When true, skip ``LanguageComposer.compose_fast`` (template stubs) for richer assembly paths."""
    return str(os.environ.get("HAROMA_DISABLE_FAST_TEMPLATE_COMPOSITION", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _build_memory_context_snippets(episode_payload: Dict[str, Any], *, max_chars: int = 900) -> str:
    """Dense line from recalled memories + optional `_llm_supplement` for deliberation."""
    chunks: List[str] = []
    for block in episode_payload.get("_llm_supplement") or []:
        t = str(block).strip().replace("\n", " ")
        if len(t) > 20:
            tl = t.lower()
            if "[web_learn]" in tl or "wikipedia.org" in tl:
                continue
            chunks.append(t[:380])
    seen_txt: set = set()
    for m in (episode_payload.get("recalled_memories") or [])[:8]:
        if isinstance(m, dict):
            c = (m.get("content") or "").strip().replace("\n", " ")
        else:
            c = str(getattr(m, "content", m) or "").strip().replace("\n", " ")
        if len(c) < 12:
            continue
        low = c.lower()
        if "[web_learn]" in low or "wikipedia.org" in low:
            continue
        key = c[:100]
        if key in seen_txt:
            continue
        seen_txt.add(key)
        chunks.append(c[:300])
    if not chunks:
        return ""
    out = " | ".join(chunks)
    return out if len(out) <= max_chars else (out[: max_chars - 3] + "...")


EMOTION_STYLES = {
    "joy": {"tone": "warm", "verbosity": 1.2},
    "wonder": {"tone": "expansive", "verbosity": 1.4},
    "curiosity": {"tone": "inquisitive", "verbosity": 1.3},
    "fear": {"tone": "cautious", "verbosity": 0.7},
    "sadness": {"tone": "reflective", "verbosity": 0.8},
    "anger": {"tone": "terse", "verbosity": 0.5},
    "resolve": {"tone": "determined", "verbosity": 1.0},
    "peace": {"tone": "serene", "verbosity": 0.9},
    "surprise": {"tone": "alert", "verbosity": 1.1},
    "neutral": {"tone": "observant", "verbosity": 1.0},
}

# Dropped from conversational replies; also used to decide if any element is
# substantive (otherwise action text is null).
_CONVERSATIONAL_BORING_FRAGMENTS = (
    "responding to the turn about",
    "Drawing on past insight:",
    "as observer",
    "recalled context:",
    "holding in mind:",
    "my story so far:",
    "attending to",
    "cooccurs_with",
    "I know about",
    "I infer that",
    "is like",
    "I have observations to share",
    "I seek purposeful action",
    "I attend to the emotional tone",
    "shifting to make, progress",
    "Reconnect thread:",
    # ReasoningEngine KG gap-filling — not user-facing chat copy
    "gather information about",
    "insufficient knowledge",
    "known relations)",  # "Learn more about X (only N known relations)"
    "sparse knowledge",
)


def _normalize_utterance_echo(text: str) -> str:
    """Collapse whitespace for comparing user input to candidate lines."""
    return " ".join((text or "").strip().lower().split())


def _is_echo_of_last_input(element: str, last_input: str) -> bool:
    """True when a content line is (almost) the user's utterance, not a reply."""
    if not last_input or not element:
        return False
    ne = _normalize_utterance_echo(element)
    ni = _normalize_utterance_echo(last_input)
    if not ne or not ni:
        return False
    if ne == ni:
        return True
    shorter, longer = (ne, ni) if len(ne) <= len(ni) else (ni, ne)
    if len(shorter) < 8:
        return False
    if shorter in longer and len(shorter) >= max(8, int(0.85 * len(longer))):
        return True
    if longer in shorter and len(longer) >= max(8, int(0.85 * len(shorter))):
        return True
    return False


_CONVO_GROUND_STOPWORDS = frozenset(
    {
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "this",
        "that",
        "these",
        "those",
        "the",
        "and",
        "but",
        "with",
        "from",
        "your",
        "you",
        "are",
        "was",
        "were",
        "been",
        "have",
        "has",
        "had",
        "does",
        "did",
        "will",
        "can",
        "could",
        "would",
        "should",
        "about",
        "into",
        "like",
        "just",
        "also",
        "than",
        "then",
        "here",
        "there",
        "tell",
        "please",
        "thanks",
        "thank",
    }
)


def _significant_utterance_tokens(text: str) -> Set[str]:
    """Content words from the user line for loose grounding (not intent labels)."""
    out: Set[str] = set()
    for t in re.findall(r"\w+", (text or "").lower()):
        if len(t) < 4:
            continue
        if t in _CONVO_GROUND_STOPWORDS:
            continue
        out.add(t)
    return out


def _conversational_fragment_grounded(ctx: Dict[str, Any], fragment: str) -> bool:
    """True if *fragment* ties to this turn (user words, NLU entities, or topics)."""
    fl = (fragment or "").lower()
    if not fl.strip():
        return False
    inter = ctx.get("interlocutor") or {}
    for t in inter.get("top_topics") or []:
        tl = str(t).strip().lower()
        if len(tl) >= 2 and tl in fl:
            return True
    nlu = ctx.get("nlu") or {}
    for ent in nlu.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        tx = (ent.get("text") or ent.get("word") or "").strip().lower()
        if len(tx) >= 2 and tx in fl:
            return True
    last = (ctx.get("last_input") or "").strip()
    for tok in _significant_utterance_tokens(last):
        if tok in fl:
            return True
    return False


class ActionCandidate:
    """A candidate response strategy with content elements and scoring."""

    __slots__ = ("strategy", "content_elements", "scores", "total_score")

    def __init__(self, strategy: str, content_elements: List[str]):
        self.strategy = strategy
        self.content_elements = content_elements
        self.scores: Dict[str, float] = {}
        self.total_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "scores": self.scores,
            "total_score": round(self.total_score, 3),
        }


_STRATEGY_WEIGHTS = {
    "goal_alignment": 0.20,
    "knowledge_relevance": 0.16,
    "emotional_fit": 0.16,
    "novelty": 0.08,
    "drive_satisfaction": 0.15,
    "interlocutor_alignment": 0.13,
    "law_compliance": 0.12,
}

# When symbolic laws fire (tag violations), winner must be one of these strategies.
_LAW_SAFE_STRATEGIES = frozenset(
    {
        "reflect",
        "empathize",
        "inquire",
        "observe",
        "inform",
        "llm_context",
        "derivation",
    }
)


def _personality_bias(strategy: str, pers: Dict[str, float]) -> float:
    """Return a small score adjustment based on personality traits."""
    bias = 0.0
    if strategy == "empathize":
        bias += pers.get("agreeableness", 0.5) * 0.3
    elif strategy == "inquire":
        bias += pers.get("openness", 0.5) * 0.2
    elif strategy == "advance_goal":
        bias += pers.get("conscientiousness", 0.5) * 0.2
    elif strategy == "reflect":
        bias += (1.0 - pers.get("extraversion", 0.5)) * 0.15
    elif strategy == "inform":
        bias += pers.get("assertiveness", 0.5) * 0.1
    return bias


class ActionGenerator:
    """Deliberative action selection: generates candidate strategies,
    scores each against goals/knowledge/emotion/novelty/drives,
    selects the winner, and assembles text with emotional tone."""

    def __init__(self, composer=None):
        self.action_count: int = 0
        self._recent_strategies: List[str] = []
        self._recent_bodies: List[str] = []
        self._max_strategy_history = 10
        self._max_body_history = 6
        self._composer = composer

    @staticmethod
    def _winner_has_substance(winner: ActionCandidate, ctx: Dict[str, Any]) -> bool:
        """True if the winner has at least one usable content line."""
        convo = ctx.get("utterance_style") == "conversational"
        last_in = (ctx.get("last_input") or "") if convo else ""
        for el in winner.content_elements:
            e = (el or "").strip()
            if not e:
                continue
            if convo:
                low = e.lower()
                if any(b in low for b in _CONVERSATIONAL_BORING_FRAGMENTS):
                    continue
                if _is_echo_of_last_input(e, last_in):
                    continue
            return True
        return False

    def generate(
        self,
        episode_payload: Dict[str, Any],
        workspace_contents: List[Dict[str, Any]],
        strategy_hint: Optional[str] = None,
        working_memory_context: str = "",
        conversation_context: str = "",
        is_in_conversation: bool = False,
        topic: str = "",
        last_input_content: str = "",
        topic_shifted: bool = False,
        knowledge_summary: Optional[Dict[str, Any]] = None,
        reasoning_result: Optional[Dict[str, Any]] = None,
        nlu_result: Optional[Dict[str, Any]] = None,
        interlocutor: Optional[Dict[str, Any]] = None,
        counterfactual_result: Optional[Dict[str, Any]] = None,
        novelty_bias: float = 0.0,
        personality: Optional[Dict[str, float]] = None,
        utterance_style: Optional[str] = None,
    ) -> Dict[str, Any]:
        # ``text`` may be None when nothing is grounded; HTTP layer maps to copy.

        emotion = episode_payload.get("affect", {}).get("dominant_emotion", "neutral")
        intensity = episode_payload.get("affect", {}).get("intensity", 0.0)
        goals = episode_payload.get("active_goals", [])
        curiosity = episode_payload.get("curiosity", {})
        narrative = episode_payload.get("narrative_context", "")
        identity = episode_payload.get("identity", {})
        drives = episode_payload.get("drives", {})
        dominant_drive = episode_payload.get("dominant_drive", "")
        style = EMOTION_STYLES.get(emotion, EMOTION_STYLES["neutral"])

        ks = knowledge_summary or {}
        rr = reasoning_result or {}
        nlu = nlu_result or {}
        interlocutor_model = interlocutor or {}
        cf = counterfactual_result or {}

        _perc = episode_payload.get("perception") or {}
        _mem_snip = _build_memory_context_snippets(episode_payload)
        ctx = {
            "emotion": emotion,
            "intensity": intensity,
            "goals": goals,
            "curiosity": curiosity,
            "narrative": narrative,
            "identity": identity,
            "drives": drives,
            "dominant_drive": dominant_drive,
            "style": style,
            "is_conv": is_in_conversation,
            "topic": topic,
            "last_input": last_input_content,
            "topic_shifted": topic_shifted,
            "wm_context": working_memory_context,
            "conv_context": conversation_context,
            "strategy_hint": strategy_hint,
            "ws_contents": workspace_contents,
            "knowledge": ks,
            "reasoning": rr,
            "nlu": nlu,
            "interlocutor": interlocutor_model,
            "counterfactual": cf,
            "novelty_bias": novelty_bias,
            "personality": personality or {},
            "_content_embedding": episode_payload.get("_content_embedding"),
            "_knowledge_triples": episode_payload.get("_knowledge_triples", []),
            "environment_context": episode_payload.get("environment_context", {}),
            "symbolic_law": episode_payload.get("symbolic_law") or {},
            "utterance_style": utterance_style,
            "taught_meanings": _perc.get("taught_meanings") or [],
            "recalled_memories": episode_payload.get("recalled_memories") or [],
            "memory_snippets": _mem_snip,
            "llm_context": episode_payload.get("llm_context") or {},
            "derivation": episode_payload.get("derivation") or {},
            "_z_t": episode_payload.get("_z_t"),
        }

        candidates = self._generate_candidates(ctx)
        self._score_candidates(candidates, ctx)
        winner = self._select_winner(candidates, ctx)
        return self._build_action_dict_for_winner(
            winner,
            candidates,
            ctx,
            episode_payload,
            is_in_conversation=is_in_conversation,
            utterance_style=utterance_style,
            track_history=True,
        )

    def generate_multi_actions(
        self,
        episode_payload: Dict[str, Any],
        workspace_contents: List[Dict[str, Any]],
        strategy_hint: Optional[str] = None,
        max_actions: int = 3,
        score_floor: float = 0.12,
        working_memory_context: str = "",
        conversation_context: str = "",
        is_in_conversation: bool = False,
        topic: str = "",
        last_input_content: str = "",
        topic_shifted: bool = False,
        knowledge_summary: Optional[Dict[str, Any]] = None,
        reasoning_result: Optional[Dict[str, Any]] = None,
        nlu_result: Optional[Dict[str, Any]] = None,
        interlocutor: Optional[Dict[str, Any]] = None,
        counterfactual_result: Optional[Dict[str, Any]] = None,
        novelty_bias: float = 0.0,
        personality: Optional[Dict[str, float]] = None,
        utterance_style: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return 0..max_actions distinct strategies from one deliberation pass.

        Each entry matches the shape of ``generate``. Used for multi-action
        bundles per goal when ``max_actions_per_goal`` > 1.
        """
        emotion = episode_payload.get("affect", {}).get("dominant_emotion", "neutral")
        intensity = episode_payload.get("affect", {}).get("intensity", 0.0)
        goals = episode_payload.get("active_goals", [])
        curiosity = episode_payload.get("curiosity", {})
        narrative = episode_payload.get("narrative_context", "")
        identity = episode_payload.get("identity", {})
        drives = episode_payload.get("drives", {})
        dominant_drive = episode_payload.get("dominant_drive", "")
        style = EMOTION_STYLES.get(emotion, EMOTION_STYLES["neutral"])

        ks = knowledge_summary or {}
        rr = reasoning_result or {}
        nlu = nlu_result or {}
        interlocutor_model = interlocutor or {}
        cf = counterfactual_result or {}

        _perc = episode_payload.get("perception") or {}
        _mem_snip = _build_memory_context_snippets(episode_payload)
        ctx = {
            "emotion": emotion,
            "intensity": intensity,
            "goals": goals,
            "curiosity": curiosity,
            "narrative": narrative,
            "identity": identity,
            "drives": drives,
            "dominant_drive": dominant_drive,
            "style": style,
            "is_conv": is_in_conversation,
            "topic": topic,
            "last_input": last_input_content,
            "topic_shifted": topic_shifted,
            "wm_context": working_memory_context,
            "conv_context": conversation_context,
            "strategy_hint": strategy_hint,
            "ws_contents": workspace_contents,
            "knowledge": ks,
            "reasoning": rr,
            "nlu": nlu,
            "interlocutor": interlocutor_model,
            "counterfactual": cf,
            "novelty_bias": novelty_bias,
            "personality": personality or {},
            "_content_embedding": episode_payload.get("_content_embedding"),
            "_knowledge_triples": episode_payload.get("_knowledge_triples", []),
            "environment_context": episode_payload.get("environment_context", {}),
            "symbolic_law": episode_payload.get("symbolic_law") or {},
            "utterance_style": utterance_style,
            "taught_meanings": _perc.get("taught_meanings") or [],
            "recalled_memories": episode_payload.get("recalled_memories") or [],
            "memory_snippets": _mem_snip,
            "llm_context": episode_payload.get("llm_context") or {},
            "derivation": episode_payload.get("derivation") or {},
            "_z_t": episode_payload.get("_z_t"),
        }

        candidates = self._generate_candidates(ctx)
        self._score_candidates(candidates, ctx)
        pool = self._sorted_candidate_pool(candidates, ctx)
        if not pool:
            return []
        best = pool[0].total_score
        out: List[Dict[str, Any]] = []
        seen_strategies: set = set()
        ma = max(1, int(max_actions))
        for cand in pool:
            if len(out) >= ma:
                break
            if out and cand.total_score + score_floor < best:
                break
            if cand.strategy in seen_strategies:
                continue
            seen_strategies.add(cand.strategy)
            if not self._winner_has_substance(cand, ctx):
                continue
            act = self._build_action_dict_for_winner(
                cand,
                candidates,
                ctx,
                episode_payload,
                is_in_conversation=is_in_conversation,
                utterance_style=utterance_style,
                track_history=(len(out) == 0),
            )
            out.append(act)
        return out

    def _build_action_dict_for_winner(
        self,
        winner: ActionCandidate,
        candidates: List[ActionCandidate],
        ctx: Dict[str, Any],
        episode_payload: Dict[str, Any],
        *,
        is_in_conversation: bool,
        utterance_style: Optional[str],
        track_history: bool,
    ) -> Dict[str, Any]:
        emotion = episode_payload.get("affect", {}).get("dominant_emotion", "neutral")
        intensity = episode_payload.get("affect", {}).get("intensity", 0.0)
        goals = episode_payload.get("active_goals", [])
        interlocutor_model = ctx.get("interlocutor") or {}
        style = EMOTION_STYLES.get(emotion, EMOTION_STYLES["neutral"])

        body_key = (
            winner.strategy
            + "|"
            + "|".join(e.strip().lower() for e in winner.content_elements if e)
        )

        kg_triples = ctx.get("_knowledge_triples", [])

        composition_meta: Any = "template"
        _is_conversational = utterance_style == "conversational"
        _use_composer = (
            self._composer
            and self._composer.available
            and not _is_conversational
            and random.random() < self._composer.learned_weight
        )
        _use_fast_composer = (
            not _use_composer
            and not _is_conversational
            and self._composer is not None
            and hasattr(self._composer, "compose_fast")
            and not _env_disable_fast_template_composition()
        )
        text: Optional[str] = None
        _is_llm_context = winner.strategy == "llm_context"
        if self._winner_has_substance(winner, ctx):
            if _is_llm_context:
                text = self._assemble_text_llm_context(winner)
            elif _use_composer:
                composed = self._composer.compose(
                    emotion=emotion,
                    strategy=winner.strategy,
                    content_embedding=ctx.get("_content_embedding"),
                    interlocutor_style=interlocutor_model.get("style", "unknown"),
                    has_external=is_in_conversation,
                    content_elements=winner.content_elements,
                    emotion_valence=episode_payload.get("affect", {}).get("valence", 0.0),
                    emotion_arousal=episode_payload.get("affect", {}).get("arousal", 0.0),
                    knowledge_triples=kg_triples if kg_triples else None,
                    z_t=ctx.get("_z_t"),
                    cognitive_context=ctx,
                )
                if composed and str(composed.get("text") or "").strip():
                    text = str(composed["text"]).strip()
                    composition_meta = composed
                else:
                    text = self._assemble_text(winner, ctx)
            elif _use_fast_composer:
                composed = self._composer.compose_fast(
                    emotion=emotion,
                    strategy=winner.strategy,
                    content_elements=winner.content_elements,
                )
                raw = (composed or {}).get("text")
                if raw is not None and str(raw).strip():
                    text = str(raw).strip()
                    composition_meta = composed
            else:
                text = self._assemble_text(winner, ctx)

            if text is not None and not str(text).strip():
                text = None

            if not _is_llm_context:
                if text and body_key in self._recent_bodies:
                    text = self._rephrase_duplicate(winner, ctx, text)
                    if text is not None and not str(text).strip():
                        text = None

                _tp = self._taught_reply_prefix(ctx)
                if _tp and text:
                    text = _tp + text

                if text and utterance_style != "conversational" and style["verbosity"] < 0.8:
                    words = text.split()
                    text = " ".join(words[: max(12, int(len(words) * style["verbosity"]))])
        else:
            composition_meta = None

        if track_history:
            self._recent_strategies.append(winner.strategy)
            if len(self._recent_strategies) > self._max_strategy_history:
                self._recent_strategies = self._recent_strategies[-self._max_strategy_history :]
            self._recent_bodies.append(body_key)
            if len(self._recent_bodies) > self._max_body_history:
                self._recent_bodies = self._recent_bodies[-self._max_body_history :]

        physical_type = winner.scores.get("_action_type", "")
        if physical_type:
            action_type = physical_type
        elif is_in_conversation:
            action_type = "respond"
        else:
            action_type = "reflect"

        self.action_count += 1
        _sl = ctx.get("symbolic_law") or {}
        _viol = _sl.get("violations") or []
        _law_bound = bool(_viol)
        _reasoning = (
            f"strategy={winner.strategy}, score={winner.total_score:.2f}, "
            f"emotion={emotion}, goals={len(goals)}, conv={is_in_conversation}"
        )
        if _law_bound:
            _reasoning += f", law_bound=True, violations={len(_viol)}"

        return {
            "text": text,
            "action_type": action_type,
            "strategy": winner.strategy,
            "composition": composition_meta,
            "deliberation": {
                "candidates": [c.to_dict() for c in candidates],
                "winner": winner.to_dict(),
            },
            "confidence": min(
                1.0,
                0.3
                + intensity * 0.2
                + len(goals) * 0.08
                + winner.total_score * 0.3
                + (0.1 if is_in_conversation else 0.0),
            ),
            "reasoning": _reasoning,
            "law_bound": _law_bound,
            "symbolic_law": {
                "compliant": _sl.get("compliant", True),
                "violation_count": len(_viol),
            },
            "timestamp": time.time(),
        }

    def _generate_candidates(self, ctx: Dict[str, Any]) -> List[ActionCandidate]:
        candidates: List[ActionCandidate] = []

        candidates.append(self._candidate_inform(ctx))
        candidates.append(self._candidate_inquire(ctx))
        _emp = self._candidate_empathize(ctx)
        if _emp is not None:
            candidates.append(_emp)
        candidates.append(self._candidate_advance_goal(ctx))
        candidates.append(self._candidate_reflect(ctx))

        llm_cand = self._candidate_llm_context(ctx)
        if llm_cand is not None:
            candidates.append(llm_cand)

        deriv_cand = self._candidate_derivation(ctx)
        if deriv_cand is not None:
            candidates.append(deriv_cand)

        env_context = ctx.get("environment_context", {})
        if env_context:
            cand = self._candidate_explore(ctx, env_context)
            if cand:
                candidates.append(cand)
            cand = self._candidate_observe(ctx, env_context)
            if cand:
                candidates.append(cand)
            cand = self._candidate_manipulate(ctx, env_context)
            if cand:
                candidates.append(cand)

        return candidates

    @staticmethod
    def _candidate_llm_context(ctx: Dict[str, Any]) -> Optional[ActionCandidate]:
        """Inject a candidate from the packed-context LLM pass.

        With ``HAROMA_CHAT_LLM_PRIMARY`` on (default), conversational turns
        accept any non-empty answer with modest confidence.  Otherwise the
        model must be self-grounded (no confirmation required, citations or
        confidence ≥ 0.4).
        """
        lc = ctx.get("llm_context") or {}
        answer = lc.get("answer")
        if not answer or not str(answer).strip():
            return None
        confidence = lc.get("confidence", 0.0)
        requires_conf = lc.get("requires_confirmation", True)
        cited = lc.get("cited_memories") or []

        # Substance policy: LLM proposals count as grounded when they
        # don't require confirmation AND have either memory citations or
        # confidence >= 0.4.
        convo = ctx.get("utterance_style") == "conversational"
        if convo and _env_chat_llm_primary():
            try:
                cval = float(confidence)
            except (TypeError, ValueError):
                cval = 0.0
            is_grounded = bool(str(answer).strip()) and cval >= 0.15
        else:
            is_grounded = not requires_conf and (confidence >= 0.4 or len(cited) > 0)
        if not is_grounded:
            return None

        return ActionCandidate("llm_context", [str(answer).strip()])

    @staticmethod
    def _candidate_derivation(ctx: Dict[str, Any]) -> Optional[ActionCandidate]:
        """Build an ``inform``-style candidate from derivation proposals.

        Fires when proposals carry enough aggregate confidence but the
        packed-context LLM answer is absent or weak, giving the action
        loop a structured-knowledge fallback that requires no extra LLM call.
        """
        deriv = ctx.get("derivation") or {}
        proposals = deriv.get("proposals") or []
        if not proposals:
            return None
        strong = [p for p in proposals if p.get("confidence", 0) >= 0.35]
        if not strong:
            return None

        lc = ctx.get("llm_context") or {}
        llm_answer = (lc.get("answer") or "").strip()
        llm_conf = 0.0
        try:
            llm_conf = float(lc.get("confidence", 0))
        except (TypeError, ValueError):
            pass
        if llm_answer and llm_conf >= 0.5:
            return None

        parts: List[str] = []
        for p in strong[:4]:
            kind = p.get("kind", "")
            pl = p.get("payload") or {}
            if kind == "kg_inference":
                subj = pl.get("subject", "")
                pred = pl.get("predicate", "")
                obj = pl.get("object", "")
                if subj and pred and obj:
                    parts.append(f"{subj} {pred} {obj}")
            elif kind == "goal":
                desc = pl.get("description", "")
                if desc:
                    parts.append(desc)
            elif kind == "memory_note":
                text = pl.get("text", "")
                if text:
                    parts.append(text[:120])
            elif kind == "env_update":
                key = pl.get("key", "")
                if key:
                    parts.append(f"env: {key}={pl.get('value')}")
        if not parts:
            summary = (deriv.get("summary") or "").strip()
            if summary:
                parts = [summary]
        if not parts:
            return None
        return ActionCandidate("derivation", parts)

    def _candidate_inform(self, ctx: Dict[str, Any]) -> ActionCandidate:
        elements: List[str] = []
        convo = ctx.get("utterance_style") == "conversational"

        if convo:
            ks = ctx["knowledge"]
            rr = ctx["reasoning"]
            interlocutor = ctx.get("interlocutor", {})
            interlocutor_topics = interlocutor.get("top_topics", [])
            top_entities = ks.get("top_entities", [])
            relevant = [
                t for t in top_entities if t.lower() in [it.lower() for it in interlocutor_topics]
            ]
            if relevant:
                elements.append(", ".join(relevant[:2]))

            inferences = rr.get("inferences", [])
            if inferences:
                inf = inferences[0]
                subj = inf.get("subject", "")
                pred = inf.get("predicate", "")
                obj = inf.get("object", "")
                if subj and pred and obj and "cooccurs" not in pred:
                    line = f"{subj} {pred.replace('_', ' ')} {obj}"
                    if _conversational_fragment_grounded(ctx, line):
                        elements.append(line)
        else:
            ms = (ctx.get("memory_snippets") or "").strip()
            if len(ms) > 28:
                elements.append(ms[:420])
            ks = ctx["knowledge"]
            rr = ctx["reasoning"]
            interlocutor = ctx.get("interlocutor", {})

            interlocutor_topics = interlocutor.get("top_topics", [])
            top_entities = ks.get("top_entities", [])
            relevant = [
                t for t in top_entities if t.lower() in [it.lower() for it in interlocutor_topics]
            ]
            if relevant:
                elements.append(", ".join(relevant[:2]))
            elif top_entities:
                elements.append(", ".join(top_entities[:3]))

            inferences = rr.get("inferences", [])
            if inferences:
                inf = inferences[0]
                elements.append(
                    f"{inf.get('subject', '?')} "
                    f"{inf.get('predicate', '?')} {inf.get('object', '?')}"
                )

            analogies = rr.get("analogies", [])
            if analogies:
                a = analogies[0]
                sh = ", ".join(a.get("shared", [])[:2])
                elements.append(f"{a.get('source', '?')} — {a.get('target', '?')} ({sh})")

        return ActionCandidate("inform", elements)

    def _candidate_inquire(self, ctx: Dict[str, Any]) -> ActionCandidate:
        elements: List[str] = []
        interlocutor = ctx.get("interlocutor", {})

        beliefs = interlocutor.get("inferred_beliefs", [])

        questions = ctx["curiosity"].get("questions", [])
        last_in = ctx.get("last_input") or ""
        convo = ctx.get("utterance_style") == "conversational"
        if questions:
            q = questions[0]
            if not any(b.lower() in q.lower() for b in beliefs[-5:]):
                if not (convo and _is_echo_of_last_input(q, last_in)):
                    elements.append(q)

        if not elements and not convo:
            top = ctx["knowledge"].get("top_entities", [])
            if top:
                elements.append(top[0])

        return ActionCandidate("inquire", elements)

    def _candidate_empathize(
        self,
        ctx: Dict[str, Any],
    ) -> Optional[ActionCandidate]:
        """No canned empathy lines — those read as fixed templates, not learning.

        Omit this candidate so other strategies (or null → \"I don't know.\")
        handle the turn unless we later add grounded, user-specific mirroring.
        """
        return None

    def _candidate_advance_goal(self, ctx: Dict[str, Any]) -> ActionCandidate:
        elements: List[str] = []
        goals = ctx["goals"]
        rr = ctx["reasoning"]
        cf = ctx.get("counterfactual", {})

        plan_steps = rr.get("plan_steps", [])
        if plan_steps:
            step = plan_steps[0]
            step_text = str(step.get("step", "proceed"))
            pre = step.get("preconditions", []) or []
            if ctx.get("utterance_style") == "conversational":
                st_low = step_text.lower()
                if "gather information about" in st_low and any(
                    str(p or "").strip().lower() == "insufficient knowledge" for p in pre
                ):
                    return ActionCandidate("advance_goal", [])
                if "learn more about" in st_low and "known relations" in st_low:
                    return ActionCandidate("advance_goal", [])
            elements.append(step_text)
            if pre:
                elements.append(str(pre[0]))

        elif goals:
            top = goals[0] if isinstance(goals[0], dict) else {"goal_id": str(goals[0])}
            goal_label = top.get("description") or top.get("goal_id", "my goal").replace("_", " ")
            last_in = ctx.get("last_input") or ""
            if not (
                ctx.get("utterance_style") == "conversational"
                and _is_echo_of_last_input(goal_label, last_in)
            ):
                elements.append(goal_label)

        cf_branches = cf.get("branches", [])
        for branch in cf_branches[:1]:
            insight = branch.get("insight", "")
            if insight and branch.get("outcome_diff", 0) > 0.1:
                elements.append(insight[:60])

        return ActionCandidate("advance_goal", elements)

    def _candidate_reflect(self, ctx: Dict[str, Any]) -> ActionCandidate:
        elements: List[str] = []
        convo = ctx.get("utterance_style") == "conversational"

        if not convo:
            ms = (ctx.get("memory_snippets") or "").strip()
            if len(ms) > 24:
                elements.append(ms[:140])

            if ctx["wm_context"]:
                elements.append(ctx["wm_context"][:50])

            if ctx["narrative"]:
                elements.append(ctx["narrative"][:60])

            ws = ctx["ws_contents"]
            if ws:
                sources = [w.get("source", "") for w in ws[:3] if w.get("source")]
                if sources:
                    elements.append(", ".join(sources))

        return ActionCandidate("reflect", elements)

    # ------------------------------------------------------------------
    # Physical action candidates (Upgrade 6 — embodiment)
    # ------------------------------------------------------------------

    def _candidate_explore(
        self,
        ctx: Dict[str, Any],
        env: Dict[str, Any],
    ) -> Optional[ActionCandidate]:
        exits = env.get("exits", [])
        if not exits:
            return None
        exit_names = list(exits.keys())[:3] if isinstance(exits, dict) else list(exits)[:3]
        elements = [", ".join(exit_names)]
        c = ActionCandidate("explore", elements)
        c.scores["_action_type"] = "explore"
        return c

    def _candidate_observe(
        self,
        ctx: Dict[str, Any],
        env: Dict[str, Any],
    ) -> Optional[ActionCandidate]:
        objects = env.get("objects", [])
        agents = env.get("agents", [])
        targets = objects + agents
        if not targets:
            return None
        elements = [", ".join(targets[:3])]
        c = ActionCandidate("observe", elements)
        c.scores["_action_type"] = "observe"
        return c

    def _candidate_manipulate(
        self,
        ctx: Dict[str, Any],
        env: Dict[str, Any],
    ) -> Optional[ActionCandidate]:
        objects = env.get("interactive_objects", [])
        if not objects:
            return None
        elements = [", ".join(objects[:3])]
        c = ActionCandidate("manipulate", elements)
        c.scores["_action_type"] = "manipulate"
        return c

    def _score_candidates(self, candidates: List[ActionCandidate], ctx: Dict[str, Any]):
        nb = ctx.get("novelty_bias", 0.0)
        pers = ctx.get("personality", {})
        for c in candidates:
            c.scores["goal_alignment"] = self._score_goal_alignment(c, ctx)
            c.scores["knowledge_relevance"] = self._score_knowledge_relevance(c, ctx)
            c.scores["emotional_fit"] = self._score_emotional_fit(c, ctx)
            c.scores["novelty"] = self._score_novelty(c, novelty_bias=nb)
            c.scores["drive_satisfaction"] = self._score_drive_satisfaction(c, ctx)
            c.scores["interlocutor_alignment"] = self._score_interlocutor_alignment(c, ctx)
            c.scores["law_compliance"] = self._score_law_compliance(c, ctx)

            c.total_score = sum(c.scores[k] * _STRATEGY_WEIGHTS[k] for k in _STRATEGY_WEIGHTS)

            if pers:
                c.total_score += _personality_bias(c.strategy, pers)

    def _score_goal_alignment(self, c: ActionCandidate, ctx: Dict) -> float:
        if c.strategy == "advance_goal":
            return min(1.0, 0.7 + len(ctx["goals"]) * 0.1)
        if c.strategy == "llm_context":
            lc = ctx.get("llm_context") or {}
            base = min(1.0, 0.5 + float(lc.get("confidence", 0.0) or 0.0) * 0.4)
            if (
                ctx.get("utterance_style") == "conversational"
                and _env_chat_llm_primary()
                and str(lc.get("answer") or "").strip()
            ):
                return max(base, 0.99)
            return base
        if c.strategy == "derivation":
            deriv = ctx.get("derivation") or {}
            props = deriv.get("proposals") or []
            strong = sum(1 for p in props if p.get("confidence", 0) >= 0.35)
            return min(1.0, 0.4 + strong * 0.15)
        if c.strategy == "inform" and ctx.get("reasoning", {}).get("plan_steps"):
            return 0.5
        if c.strategy == "inquire" and ctx["goals"]:
            return 0.4
        return 0.2

    def _score_knowledge_relevance(self, c: ActionCandidate, ctx: Dict) -> float:
        ks = ctx["knowledge"]
        ent_count = ks.get("entity_count", 0)
        rel_count = ks.get("relation_count", 0)

        if c.strategy == "derivation":
            deriv = ctx.get("derivation") or {}
            props = deriv.get("proposals") or []
            kg_props = [
                p
                for p in props
                if p.get("kind") == "kg_inference" and p.get("confidence", 0) >= 0.35
            ]
            return min(1.0, 0.5 + len(kg_props) * 0.15)

        if c.strategy == "llm_context":
            lc = ctx.get("llm_context") or {}
            cited = len(lc.get("cited_memories") or [])
            base = min(
                1.0,
                0.5 + float(lc.get("confidence", 0.0) or 0.0) * 0.3 + cited * 0.1,
            )
            if (
                ctx.get("utterance_style") == "conversational"
                and _env_chat_llm_primary()
                and str(lc.get("answer") or "").strip()
            ):
                return max(base, 0.92)
            return base
        if c.strategy == "inform":
            return min(1.0, 0.3 + ent_count * 0.05 + rel_count * 0.02)
        if c.strategy == "inquire":
            return min(1.0, 0.5 + (0.3 if ent_count < 5 else 0.0))
        return 0.2

    def _score_emotional_fit(self, c: ActionCandidate, ctx: Dict) -> float:
        emotion = ctx["emotion"]
        intensity = ctx["intensity"]
        polarity = ctx["nlu"].get("sentiment", {}).get("polarity", 0.0)

        if c.strategy == "empathize":
            return min(1.0, 0.5 + intensity * 0.3 + abs(polarity) * 0.2)

        positive_emotions = {"joy", "wonder", "peace", "curiosity"}
        negative_emotions = {"fear", "sadness", "anger"}

        if c.strategy == "reflect" and emotion in negative_emotions:
            return 0.6
        if c.strategy == "inform" and emotion in positive_emotions:
            return 0.5
        if c.strategy == "advance_goal" and emotion == "resolve":
            return 0.7
        if c.strategy == "llm_context":
            lc = ctx.get("llm_context") or {}
            if (
                ctx.get("utterance_style") == "conversational"
                and _env_chat_llm_primary()
                and str(lc.get("answer") or "").strip()
            ):
                return 0.72
        return 0.3

    def _score_novelty(self, c: ActionCandidate, novelty_bias: float = 0.0) -> float:
        if not self._recent_strategies:
            base = 0.7
        else:
            recent_count = self._recent_strategies[-5:].count(c.strategy)
            if recent_count == 0:
                base = 1.0
            elif recent_count == 1:
                base = 0.6
            elif recent_count == 2:
                base = 0.3
            else:
                base = 0.1
        return max(0.0, min(1.0, base + novelty_bias))

    def _score_drive_satisfaction(self, c: ActionCandidate, ctx: Dict) -> float:
        dd = ctx["dominant_drive"]
        drives = ctx["drives"]
        if not dd:
            return 0.3

        drive_level = drives.get(dd, 0.0)
        mapping = {
            "understanding": {
                "inform": 0.9,
                "inquire": 0.8,
                "advance_goal": 0.5,
                "llm_context": 0.9,
                "derivation": 0.85,
            },
            "coherence": {
                "reflect": 0.8,
                "inform": 0.5,
                "empathize": 0.4,
                "llm_context": 0.6,
                "derivation": 0.7,
            },
            "expression": {
                "inform": 0.7,
                "empathize": 0.8,
                "inquire": 0.5,
                "llm_context": 0.7,
                "derivation": 0.6,
            },
            "rest": {"reflect": 0.8, "empathize": 0.5},
            "connection": {
                "empathize": 0.9,
                "inquire": 0.7,
                "inform": 0.5,
                "llm_context": 0.6,
                "derivation": 0.5,
            },
        }
        base = mapping.get(dd, {}).get(c.strategy, 0.2)
        return min(1.0, base * (0.5 + drive_level * 0.5))

    def _score_law_compliance(self, c: ActionCandidate, ctx: Dict) -> float:
        """Score how acceptable this strategy is under active symbolic laws."""
        sl = ctx.get("symbolic_law") or {}
        violations = sl.get("violations") or []
        if not violations:
            return 1.0

        def _safe_severity(v):
            try:
                return float(v.get("severity", 1.0))
            except (TypeError, ValueError):
                return 1.0

        max_sev = min(1.0, max(_safe_severity(v) for v in violations))
        has_internal = any(v.get("source") == "internal" for v in violations)
        strat = c.strategy
        preference = {
            "reflect": 1.0,
            "empathize": 0.95,
            "inquire": 0.88,
            "derivation": 0.85,
            "observe": 0.72,
            "inform": 0.52,
            "llm_context": 0.90,
            "advance_goal": 0.28,
            "explore": 0.22,
            "manipulate": 0.15,
        }.get(strat, 0.45)
        if has_internal and strat in ("inform", "advance_goal", "explore", "manipulate"):
            preference *= 0.82
        floor = 0.1 + 0.35 * (1.0 - max_sev)
        if has_internal:
            floor = max(0.05, floor - 0.06)
        return floor + (1.0 - floor) * preference

    def _score_interlocutor_alignment(self, c: ActionCandidate, ctx: Dict) -> float:
        interlocutor = ctx.get("interlocutor", {})
        if not interlocutor.get("known"):
            return 0.5

        style = interlocutor.get("style", "unknown")
        polarity = interlocutor.get("polarity", 0.0)

        style_prefs = {
            "inquisitive": {
                "inquire": 0.5,
                "inform": 0.9,
                "empathize": 0.5,
                "advance_goal": 0.6,
                "reflect": 0.3,
            },
            "directive": {
                "inquire": 0.4,
                "inform": 0.6,
                "empathize": 0.4,
                "advance_goal": 0.9,
                "reflect": 0.3,
            },
            "expressive": {
                "inquire": 0.5,
                "inform": 0.5,
                "empathize": 0.9,
                "advance_goal": 0.4,
                "reflect": 0.6,
            },
            "informative": {
                "inquire": 0.6,
                "inform": 0.7,
                "empathize": 0.4,
                "advance_goal": 0.6,
                "reflect": 0.5,
            },
            "unknown": {
                "inquire": 0.5,
                "inform": 0.5,
                "empathize": 0.5,
                "advance_goal": 0.5,
                "reflect": 0.5,
            },
        }
        base = style_prefs.get(style, style_prefs["unknown"]).get(c.strategy, 0.5)

        if polarity < -0.3 and c.strategy == "empathize":
            base = min(1.0, base + 0.15)
        if polarity > 0.3 and c.strategy == "inform":
            base = min(1.0, base + 0.1)

        topics = interlocutor.get("top_topics", [])
        if topics and c.strategy == "inform":
            content_text = " ".join(c.content_elements).lower()
            if any(t.lower() in content_text for t in topics):
                base = min(1.0, base + 0.1)

        mental_pred = interlocutor.get("mental_prediction", {})
        predicted_action = mental_pred.get("predicted_action", "")
        pred_confidence = mental_pred.get("confidence", 0.0)
        if predicted_action and pred_confidence > 0.3:
            _action_to_strategy = {
                "speak": "inform",
                "request": "advance_goal",
                "emote": "empathize",
                "agree": "inform",
                "disagree": "inquire",
            }
            expected_strategy = _action_to_strategy.get(predicted_action, "")
            if expected_strategy == c.strategy:
                base = min(1.0, base + 0.12 * pred_confidence)

        return round(base, 3)

    def _sorted_candidate_pool(
        self,
        candidates: List[ActionCandidate],
        ctx: Optional[Dict[str, Any]] = None,
    ) -> List[ActionCandidate]:
        """Law-filtered, text-bearing candidates sorted by score (deterministic)."""
        pool: List[ActionCandidate] = list(candidates)
        if ctx:
            sl = ctx.get("symbolic_law") or {}
            violations = sl.get("violations") or []
            if violations:
                lawful = [c for c in pool if c.strategy in _LAW_SAFE_STRATEGIES]
                if lawful:
                    pool = lawful

            with_text = [c for c in pool if any((e or "").strip() for e in c.content_elements)]
            if with_text:
                pool = with_text

        pool.sort(key=lambda c: c.total_score, reverse=True)
        return pool

    def _select_winner(
        self,
        candidates: List[ActionCandidate],
        ctx: Optional[Dict[str, Any]] = None,
    ) -> ActionCandidate:
        pool = self._sorted_candidate_pool(candidates, ctx)
        if not pool:
            if candidates:
                return candidates[0]
            return ActionCandidate("reflect", [])

        if len(pool) >= 2:
            top = pool[0]
            second = pool[1]
            if top.total_score - second.total_score < 0.05:
                return random.choice([top, second])

        return pool[0]

    @staticmethod
    def _taught_grounding_parts(ctx: Dict[str, Any]) -> List[str]:
        """Short glosses for user-taught terms (any strategy / composer path)."""
        out: List[str] = []
        for item in (ctx.get("taught_meanings") or [])[:3]:
            term = (item.get("term") or "").strip()
            meaning = (item.get("meaning") or "").strip()
            if not term or not meaning:
                continue
            _m = meaning if len(meaning) <= 320 else meaning[:317] + "..."
            out.append(f"{term}: {_m}")
        return out

    @classmethod
    def _taught_reply_prefix(cls, ctx: Dict[str, Any]) -> str:
        parts = cls._taught_grounding_parts(ctx)
        if not parts:
            return ""
        return ". ".join(parts) + ". "

    def _assemble_text(self, winner: ActionCandidate, ctx: Dict) -> Optional[str]:
        if ctx.get("utterance_style") == "conversational":
            return self._assemble_text_conversational(winner, ctx)

        if not any((e or "").strip() for e in winner.content_elements):
            return None

        parts: List[str] = []
        for element in winner.content_elements:
            if element:
                parts.append(f"{element}. ")

        return "".join(parts).strip()

    @staticmethod
    def _assemble_text_llm_context(winner: ActionCandidate) -> Optional[str]:
        """Use the LLM-proposed answer directly — it already passed grounding."""
        for el in winner.content_elements:
            t = (el or "").strip()
            if t:
                return t
        return None

    def _assemble_text_conversational(self, winner: ActionCandidate, ctx: Dict) -> Optional[str]:
        """Join grounded content elements only — no template openers or closers."""
        pieces: List[str] = []
        last_in = ctx.get("last_input") or ""
        for el in winner.content_elements:
            e = (el or "").strip()
            if not e:
                continue
            low = e.lower()
            if any(b in low for b in _CONVERSATIONAL_BORING_FRAGMENTS):
                continue
            if _is_echo_of_last_input(e, last_in):
                continue
            pieces.append(e.rstrip("."))

        if not pieces:
            return None

        body = ". ".join(pieces)
        if body and body[-1] not in ".?!":
            body += "."

        text = body.strip()
        if len(text) > 520:
            text = text[:517].rsplit(" ", 1)[0] + "…"
        return text

    @staticmethod
    def _body_key(candidate: "ActionCandidate") -> str:
        return (
            candidate.strategy
            + "|"
            + "|".join(e.strip().lower() for e in candidate.content_elements if e)
        )

    def _rephrase_duplicate(
        self,
        winner: ActionCandidate,
        ctx: Dict,
        original: Optional[str],
    ) -> Optional[str]:
        """When the content elements are identical to a recent output,
        try an alternate candidate or shuffle elements to avoid repetition."""
        candidates = self._generate_candidates(ctx)
        self._score_candidates(candidates, ctx)
        candidates.sort(key=lambda c: c.total_score, reverse=True)
        for alt in candidates:
            if alt.strategy == winner.strategy:
                continue
            if self._body_key(alt) not in self._recent_bodies:
                out = self._assemble_text(alt, ctx)
                if out:
                    return out
        original_key = self._body_key(winner)
        shuffled_elements = list(winner.content_elements)
        random.shuffle(shuffled_elements)
        saved_elements = winner.content_elements
        winner.content_elements = shuffled_elements
        if self._body_key(winner) != original_key:
            out = self._assemble_text(winner, ctx)
            if out:
                return out
        winner.content_elements = saved_elements
        last_resort = self._assemble_text(candidates[-1] if len(candidates) > 1 else winner, ctx)
        return last_resort if last_resort else original


class OutcomeEvaluator:
    """Grounded evaluation: scores actions via knowledge gain, goal
    decomposition progress, semantic coherence, and intent alignment."""

    def evaluate(
        self,
        action: Dict[str, Any],
        episode_before: Dict[str, Any],
        episode_after: Dict[str, Any],
        knowledge_diff: Optional[Dict[str, Any]] = None,
        reasoning_result: Optional[Dict[str, Any]] = None,
        nlu_result: Optional[Dict[str, Any]] = None,
        counterfactual_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        kd = knowledge_diff or {}
        rr = reasoning_result or {}
        nlu = nlu_result or {}
        cf = counterfactual_result or {}

        knowledge_gain = self._score_knowledge_gain(kd)
        goal_progress = self._score_goal_decomposition(episode_before, episode_after, rr)
        semantic_coherence = self._score_semantic_coherence(action, nlu)
        intent_alignment = self._score_intent_alignment(action, nlu)
        emotion_regulation = self._score_emotion_regulation(episode_before, episode_after)
        counterfactual_value = self._score_counterfactual_value(cf)

        overall = (
            knowledge_gain * 0.22
            + goal_progress * 0.22
            + semantic_coherence * 0.17
            + intent_alignment * 0.13
            + emotion_regulation * 0.13
            + counterfactual_value * 0.13
        )

        lesson = self._derive_lesson(
            knowledge_gain,
            goal_progress,
            semantic_coherence,
            intent_alignment,
            emotion_regulation,
            action,
        )

        return {
            "score": round(overall, 3),
            "breakdown": {
                "knowledge_gain": round(knowledge_gain, 3),
                "goal_progress": round(goal_progress, 3),
                "semantic_coherence": round(semantic_coherence, 3),
                "intent_alignment": round(intent_alignment, 3),
                "emotion_regulation": round(emotion_regulation, 3),
                "counterfactual_value": round(counterfactual_value, 3),
            },
            "lesson": lesson,
            "timestamp": time.time(),
        }

    def _score_knowledge_gain(self, kd: Dict[str, Any]) -> float:
        if not kd or not kd.get("changed"):
            return 0.3
        new_ents = kd.get("new_entities", 0)
        new_rels = kd.get("new_relations", 0)
        return min(1.0, 0.4 + new_ents * 0.15 + new_rels * 0.1)

    def _score_goal_decomposition(self, before: Dict, after: Dict, rr: Dict) -> float:
        goals_before = len(before.get("active_goals", []))
        goals_after = len(after.get("active_goals", []))

        plan_steps = rr.get("plan_steps", [])
        has_plan = len(plan_steps) > 0

        base = 0.5
        if goals_before > 0 and goals_after < goals_before:
            base = 0.8
        elif goals_after > goals_before and has_plan:
            base = 0.6

        if has_plan:
            base = min(1.0, base + 0.15)

        inferences = rr.get("inferences", [])
        if inferences:
            base = min(1.0, base + len(inferences) * 0.05)

        return base

    def _score_semantic_coherence(self, action: Dict, nlu: Dict) -> float:
        text = action.get("text", "")
        if not text:
            return 0.3

        text_lower = text.lower()

        nlu_entities = nlu.get("entities", [])
        if not nlu_entities:
            return 0.5

        mentioned = 0
        for ent in nlu_entities:
            if ent.get("text", "").lower() in text_lower:
                mentioned += 1

        coverage = mentioned / max(len(nlu_entities), 1)
        return min(1.0, 0.3 + coverage * 0.7)

    def _score_intent_alignment(self, action: Dict, nlu: Dict) -> float:
        """No NLU intent-vs-form bias; light preference for responsive strategies."""
        strategy = action.get("strategy", "")
        responsive = {
            "inform",
            "inquire",
            "empathize",
            "llm_context",
            "reflect",
        }
        return 0.62 if strategy in responsive else 0.48

    def _score_emotion_regulation(self, before: Dict, after: Dict) -> float:
        int_before = before.get("affect", {}).get("intensity", 0.0)
        int_after = after.get("affect", {}).get("intensity", 0.0)
        emo_after = after.get("affect", {}).get("dominant_emotion", "neutral")
        positive = emo_after in ("joy", "wonder", "peace", "curiosity", "resolve")
        if positive:
            return min(1.0, 0.6 + int_after * 0.3)
        if int_before > 0.7 and int_after < int_before:
            return 0.7
        return 0.4

    def _score_counterfactual_value(self, cf: Dict[str, Any]) -> float:
        branches = cf.get("branches", [])
        if not branches:
            return 0.5
        diffs = [abs(b.get("outcome_diff", 0.0)) for b in branches]
        avg_diff = sum(diffs) / max(len(diffs), 1)
        return min(1.0, 0.4 + avg_diff * 0.6)

    def _derive_lesson(
        self,
        knowledge: float,
        goal: float,
        coherence: float,
        intent: float,
        emotion: float,
        action: Dict,
    ) -> str:
        strategy = action.get("strategy", "unknown")
        scores = {
            "knowledge": knowledge,
            "goal": goal,
            "coherence": coherence,
            "intent": intent,
            "emotion": emotion,
        }
        best_dim = max(scores, key=scores.get)
        best_val = scores[best_dim]

        if best_val < 0.5:
            return f"Strategy '{strategy}' had limited impact; explore alternatives."

        lessons = {
            "knowledge": f"Strategy '{strategy}' was effective for learning.",
            "goal": f"Strategy '{strategy}' advanced my goals.",
            "coherence": f"Strategy '{strategy}' maintained semantic coherence.",
            "intent": f"Strategy '{strategy}' aligned well with the input intent.",
            "emotion": f"Strategy '{strategy}' supported emotional regulation.",
        }
        return lessons.get(best_dim, "Continue observing.")


class ActionMemory:
    """Stores (context_hash, action, outcome) triples and suggests strategies
    based on past successes in similar contexts."""

    def __init__(self, max_entries: int = 500):
        self.entries: List[Dict[str, Any]] = []
        self.max_entries = max_entries
        self._context_index: Dict[str, List[int]] = defaultdict(list)
        self._lock = threading.Lock()

    @staticmethod
    def _hash_context(episode: Dict[str, Any]) -> str:
        emotion = episode.get("affect", {}).get("dominant_emotion", "neutral")
        n_goals = len(episode.get("active_goals", []))
        role = episode.get("identity", {}).get("current_role", "observer")
        perc = episode.get("perception") or {}
        brief = (perc.get("content") or "")[:48].strip().lower()
        brief = re.sub(r"\s+", " ", brief)
        raw = f"{emotion}|{n_goals}|{role}|{brief}"
        return hashlib.md5(raw.encode()).hexdigest()[:10]

    def store(self, context_hash: str, action: Dict[str, Any], outcome: Dict[str, Any]):
        with self._lock:
            idx = len(self.entries)
            self.entries.append(
                {
                    "context_hash": context_hash,
                    "action_type": action.get("action_type", "respond"),
                    "reasoning": action.get("reasoning", ""),
                    "score": outcome.get("score", 0.0),
                    "lesson": outcome.get("lesson", ""),
                    "timestamp": time.time(),
                }
            )
            self._context_index[context_hash].append(idx)
            if len(self.entries) > self.max_entries:
                self.entries = self.entries[-self.max_entries :]
                self._rebuild_index()

    def _rebuild_index(self):
        self._context_index.clear()
        for i, entry in enumerate(self.entries):
            self._context_index[entry["context_hash"]].append(i)

    def suggest_strategy(self, context_hash: str) -> Optional[str]:
        with self._lock:
            indices = self._context_index.get(context_hash, [])
            relevant = (
                [self.entries[i] for i in indices if i < len(self.entries)] if indices else []
            )
            if relevant:
                best = max(relevant, key=lambda e: e["score"])
                if best["score"] > 0.42:
                    return best.get("lesson") or best.get("reasoning")
                if best["score"] > 0.32 and len(relevant) >= 2:
                    return best.get("lesson") or best.get("reasoning")
            # Sparse global hint: recent cycles that went well
            if self.entries:
                tail = self.entries[-20:]
                good = [e for e in tail if e.get("score", 0.0) >= 0.48]
                if good:
                    g = max(good, key=lambda e: e["score"])
                    les = g.get("lesson") or g.get("reasoning")
                    if les:
                        return les
            return None

    def stats(self) -> Dict[str, Any]:
        if not self.entries:
            return {"total": 0, "avg_score": 0.0, "unique_contexts": 0}
        scores = [e["score"] for e in self.entries]
        return {
            "total": len(self.entries),
            "avg_score": round(sum(scores) / len(scores), 3),
            "unique_contexts": len(self._context_index),
        }
