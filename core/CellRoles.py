"""CellRoles — soft role annotations for the X7 Symbolic Cell Architecture.

Each module is tagged as generator (001), processor (010), consumer (100),
or a hybrid combination. This enables the SymbolicQueue layer to validate
writes and the OrganRegistry to reason about data flow.

The annotations don't change how modules work — they're metadata for the
queue integrity and organ registry layers.
"""

from __future__ import annotations

from typing import Any, Dict, List


GENERATOR = 0b001
PROCESSOR = 0b010
CONSUMER = 0b100


def _role_label(code: int) -> str:
    parts: List[str] = []
    if code & GENERATOR:
        parts.append("generator")
    if code & PROCESSOR:
        parts.append("processor")
    if code & CONSUMER:
        parts.append("consumer")
    return "+".join(parts) or "unknown"


CELL_ROLES: Dict[str, int] = {
    # Generators (001) — read memory, produce symbolic output
    "perception": GENERATOR,
    "encoder": GENERATOR,
    "nlu": GENERATOR,
    "curiosity": GENERATOR,
    "curiosity_manager": GENERATOR,
    "imagination": GENERATOR,
    "imagination_manager": GENERATOR,
    "dream": GENERATOR,
    "discourse": GENERATOR,
    # Processors (010) — transform generator output
    "appraisal": PROCESSOR,
    "appraisal_manager": PROCESSOR,
    "reasoning": PROCESSOR,
    "reasoning_manager": PROCESSOR,
    "attention": PROCESSOR,
    "attention_manager": PROCESSOR,
    "self_model": PROCESSOR,
    "self_model_manager": PROCESSOR,
    "backbone": PROCESSOR,
    "metacognition": PROCESSOR,
    "metacognition_manager": PROCESSOR,
    "counterfactual": PROCESSOR,
    "counterfactual_manager": PROCESSOR,
    "goal_synthesizer": PROCESSOR,
    "goal_synthesizer_manager": PROCESSOR,
    "process_gate": PROCESSOR,
    "temporal": PROCESSOR,
    "temporal_manager": PROCESSOR,
    "grounder": PROCESSOR,
    "mental_simulator": PROCESSOR,
    "arch_searcher": PROCESSOR,
    "workspace": PROCESSOR,
    "modulation": PROCESSOR,
    "modulation_manager": PROCESSOR,
    # Consumers (100) — commit to memory/action
    "action_generator": CONSUMER,
    "outcome_evaluator": CONSUMER,
    "dream_consolidator": CONSUMER,
    "persistence": CONSUMER,
    "narrative": CONSUMER,
    "conversation": CONSUMER,
    # Hybrids
    "emotion": GENERATOR | PROCESSOR,  # 011 — generates + processes
    "knowledge": GENERATOR | CONSUMER,  # 101 — generates + consumes
    "knowledge_manager": GENERATOR | CONSUMER,
    "composer": PROCESSOR | CONSUMER,  # 110 — processes + consumes
    "llm_manager": GENERATOR | PROCESSOR,
}


def get_role(module_name: str) -> int:
    """Return the 3-bit role code for a module (0 if unregistered)."""
    return CELL_ROLES.get(module_name, 0)


def get_role_label(module_name: str) -> str:
    """Human-readable role label."""
    return _role_label(get_role(module_name))


def generators() -> List[str]:
    return [m for m, c in CELL_ROLES.items() if c & GENERATOR]


def processors() -> List[str]:
    return [m for m, c in CELL_ROLES.items() if c & PROCESSOR]


def consumers() -> List[str]:
    return [m for m, c in CELL_ROLES.items() if c & CONSUMER]


def modules_with_role(role: int) -> List[str]:
    return [m for m, c in CELL_ROLES.items() if c & role]


def summary() -> Dict[str, Any]:
    return {
        "total_modules": len(CELL_ROLES),
        "generators": len(generators()),
        "processors": len(processors()),
        "consumers": len(consumers()),
        "roles": {m: _role_label(c) for m, c in CELL_ROLES.items()},
    }
