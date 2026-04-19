"""Cognitive loop groups — multi-tick phased execution for :class:`agents.persona_agent.PersonaAgent`.

The monolithic ``_process_message`` pipeline is split into **phase boundaries**
(yields). When ``HAROMA_PERSONA_PHASED_CYCLE=1``, each call to
``_process_message`` advances up to ``HAROMA_PERSONA_PHASE_STEPS_PER_TICK``
boundaries; remaining work resumes on later agent ticks (and optional extra
``next()`` calls from the same invocation).

**Groups (yield order)**

1. ``neural_gate`` — After perception, memory encounter, embedding; before semantic recall.
2. ``pre_llm`` — After recall, affect, workspace, reasoning prep; before packed LLM I/O.
3. ``post_llm`` — After packed LLM; before counterfactual / action / outcome / response.

A full loop therefore may take **several** ticks when phased mode is on.

**Parallelism**

Independent CPU-bound slices could run concurrently in principle; the default
implementation keeps **sequential** semantics. Set ``HAROMA_PERSONA_PARALLEL_GROUPS=1``
to allow :func:`run_parallel_prep` to use a small thread pool for registered
pairs (conservative; extend where profiling shows safe disjoint reads).

See also :data:`mind.cycle_flow.PERSONA_NEURAL_PHASES` (neural vs off-lock naming).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Tuple

# Yield labels (stable for logging / metrics)
COGNITIVE_PHASE_NEURAL_GATE = "neural_gate"
COGNITIVE_PHASE_PRE_LLM = "pre_llm"
COGNITIVE_PHASE_POST_LLM = "post_llm"

COGNITIVE_LOOP_GROUP_ORDER: Tuple[str, ...] = (
    COGNITIVE_PHASE_NEURAL_GATE,
    COGNITIVE_PHASE_PRE_LLM,
    COGNITIVE_PHASE_POST_LLM,
)


def cognitive_phases_enabled() -> bool:
    v = str(os.environ.get("HAROMA_PERSONA_PHASED_CYCLE", "") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def phase_steps_per_invocation() -> int:
    """Max phase boundaries advanced per ``_process_message`` call when phased."""
    raw = str(os.environ.get("HAROMA_PERSONA_PHASE_STEPS_PER_TICK", "1") or "").strip()
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = 1
    return max(1, min(32, n))


def parallel_groups_enabled() -> bool:
    v = str(os.environ.get("HAROMA_PERSONA_PARALLEL_GROUPS", "") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def run_parallel_prep(
    tasks: List[Tuple[str, Callable[[], Any]]],
    *,
    max_workers: int = 2,
) -> List[Tuple[str, Any]]:
    """Run independent prep callables in parallel when enabled; else sequential.

    *tasks* is ``(label, fn)`` pairs. Returns ``(label, result_or_exc)`` in
    completion order when parallel; insertion order when sequential.
    """
    if not tasks:
        return []
    if not parallel_groups_enabled() or len(tasks) <= 1:
        out: List[Tuple[str, Any]] = []
        for label, fn in tasks:
            try:
                out.append((label, fn()))
            except Exception as exc:
                out.append((label, exc))
        return out
    workers = max(1, min(max_workers, len(tasks)))
    out_list: List[Tuple[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn): label for label, fn in tasks}
        for fut in as_completed(futs):
            label = futs[fut]
            try:
                out_list.append((label, fut.result()))
            except Exception as exc:
                out_list.append((label, exc))
    return out_list
