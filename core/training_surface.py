"""
Background neural training map — single registry for :class:`agents.background_agent.BackgroundAgent`.

Foreground cognitive loops (``mind.control``, :class:`agents.persona_agent.PersonaAgent`) produce
snapshots like :class:`core.self_model_train_batch.SelfModelTrainBatch` on ``SharedResources``;
this module defines which modules run in the background tick and how each ``train_step`` is invoked.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, TYPE_CHECKING

from core.cognitive_null import is_cognitive_null
from core.self_model_train_batch import SelfModelTrainBatch

if TYPE_CHECKING:
    from agents.shared_resources import SharedResources

TrainStepFn = Callable[[], Optional[float]]
BackgroundTrainMap = List[Tuple[str, TrainStepFn]]


def _is_real(obj) -> bool:
    if obj is None:
        return False
    if is_cognitive_null(obj):
        return False
    if hasattr(type(obj), "_is_null_stub"):
        return False
    return True


def _encoder_train_step(shared: "SharedResources") -> Optional[float]:
    """Contrastive encoder step using recent memory and action-memory traces."""
    enc = getattr(shared, "encoder", None)
    if not _is_real(enc) or not hasattr(enc, "train_step"):
        return None
    texts: List[str] = []
    mem = getattr(shared, "memory", None)
    if _is_real(mem):
        for tree_name, branch in (
            ("thought_tree", "common"),
            ("thought_tree", "web_learn"),
        ):
            try:
                nodes = mem.get_nodes(tree_name, branch)
                for n in nodes[-8:]:
                    c = (getattr(n, "content", None) or "").strip()
                    if len(c) > 24:
                        texts.append(c[:480])
            except Exception:
                continue
    am = getattr(shared, "action_memory", None)
    if am and getattr(am, "entries", None):
        for e in am.entries[-10:]:
            r = (e.get("reasoning") or "").strip()
            if len(r) > 24:
                texts.append(r[:480])
    seen = set()
    uniq: List[str] = []
    for t in texts:
        k = t[:120]
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    if len(uniq) < 3:
        return None
    anchor = uniq[-1]
    positive = uniq[-2]
    negative = (
        uniq[0]
        if uniq[0] not in (anchor, positive)
        else (
            uniq[1]
            if len(uniq) > 2 and uniq[1] not in (anchor, positive)
            else "static placeholder unrelated cognitive noise zyqxw"
        )
    )
    try:
        loss = enc.train_step(anchor, [positive], [negative])
        return float(loss) if loss is not None else None
    except Exception:
        return None


def _self_model_background_train_step(shared: "SharedResources") -> Optional[float]:
    """Use last persona :class:`SelfModelTrainBatch` (same triple as ``mind.control``)."""
    sm = getattr(shared, "self_model", None)
    if not _is_real(sm) or not getattr(sm, "available", False):
        return None
    ctx = getattr(shared, "_self_model_last_train_ctx", None)
    if ctx is None:
        return None
    if isinstance(ctx, SelfModelTrainBatch):
        emb = ctx.embedding
        prev = ctx.prev_state
        actual = ctx.actual_state
    else:
        try:
            emb, prev, actual = ctx
        except Exception:
            return None
    if emb is None:
        return None
    return sm.train_step(emb, prev, actual)


def _grounder_train_step(shared: "SharedResources") -> Optional[float]:
    g = getattr(shared, "grounder", None)
    if not _is_real(g) or not getattr(g, "available", False):
        return None
    if not hasattr(g, "train_step"):
        return None
    return g.train_step()


def build_background_train_map(shared: "SharedResources") -> BackgroundTrainMap:
    """Ordered (module_name, train_fn) pairs for one background training pass."""
    s = shared
    return [
        ("encoder", lambda: _encoder_train_step(s)),
        ("backbone", lambda: s.backbone.train_step() if _is_real(s.backbone) else None),
        ("attention", lambda: s.attention.train_step() if _is_real(s.attention) else None),
        (
            "process_gate",
            lambda: s.process_gate.train_step() if _is_real(s.process_gate) else None,
        ),
        ("self_model", lambda: _self_model_background_train_step(s)),
        ("appraisal", lambda: s.appraisal.train_step() if _is_real(s.appraisal) else None),
        ("modulation", lambda: s.modulation.train_step() if _is_real(s.modulation) else None),
        (
            "goal_synth",
            lambda: s.goal_synthesizer.train_step() if _is_real(s.goal_synthesizer) else None,
        ),
        (
            "imagination",
            lambda: (
                s.imagination.train_step()
                if _is_real(s.imagination) and s.imagination.available
                else None
            ),
        ),
        (
            "metacog",
            lambda: s.metacognition.train_step() if _is_real(s.metacognition) else None,
        ),
        ("composer", lambda: s.composer.train_step() if _is_real(s.composer) else None),
        (
            "generative",
            lambda: s.composer.train_generative_step() if _is_real(s.composer) else None,
        ),
        (
            "counterfactual",
            lambda: (
                s.counterfactual.train_step()
                if _is_real(s.counterfactual)
                and getattr(s.counterfactual, "_gate_available", False)
                else None
            ),
        ),
        ("grounder", lambda: _grounder_train_step(s)),
        (
            "mental_sim",
            lambda: (
                s.mental_simulator.train_step()
                if _is_real(s.mental_simulator) and s.mental_simulator.available
                else None
            ),
        ),
        (
            "arch_search",
            lambda: (
                s.arch_searcher.train_step()
                if _is_real(s.arch_searcher) and s.arch_searcher.available
                else None
            ),
        ),
        (
            "llm_reward",
            # PyTorch reward + optional Vowpal Wabbit; see docs/gymnasium-bridge.md (BackgroundAgent).
            lambda: s.llm_backend.train_reward_model() if _is_real(s.llm_backend) else None,
        ),
    ]
