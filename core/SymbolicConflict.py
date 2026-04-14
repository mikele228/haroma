"""SymbolicConflict — context binding and pairwise contradiction traces for ontology bridge.

Supports :class:`OntologyBridgeEngine` in ``KnowledgeBase``: enriched logic tuples are
compared with lightweight structural rules (polarity, competing goals, time).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from utils.module_base import ModuleBase


class ContextInterpreter(ModuleBase):
    """Bind free variables in symbolic statements to memory / session context (stub for extension)."""

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(module_name="ContextInterpreter")
        self.context = context or {}

    def interpret(self, stmt: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "statement": stmt,
            "context_keys": list(self.context.keys()),
        }


class ContradictionTracer(ModuleBase):
    """Pairwise contradiction heuristic between two logic-shaped dicts."""

    def __init__(self):
        super().__init__(module_name="ContradictionTracer")

    def trace(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        subj_a = str(a.get("subject", "") or "")
        subj_b = str(b.get("subject", "") or "")
        then_a = str(a.get("THEN") or "").strip().lower()
        then_b = str(b.get("THEN") or "").strip().lower()
        neg_a = bool(a.get("NEGATION"))
        neg_b = bool(b.get("NEGATION"))
        contra = False
        reason = "none"

        same_subject = bool(subj_a and subj_b and subj_a == subj_b)

        if same_subject and then_a and then_b:
            if then_a == then_b and neg_a != neg_b:
                contra = True
                reason = "polarity_mismatch"
            elif (
                then_a != then_b
                and a.get("TYPE") == "goal_intent"
                and b.get("TYPE") == "goal_intent"
            ):
                contra = True
                reason = "competing_goals"

        if not contra and then_a and then_b and then_a == then_b and neg_a != neg_b:
            contra = True
            reason = "negation_clash"

        time_a = a.get("TIME")
        time_b = b.get("TIME")
        if (
            not contra
            and time_a is not None
            and time_b is not None
            and time_a != time_b
            and same_subject
            and then_a
            and then_b
            and then_a == then_b
        ):
            contra = True
            reason = "temporal_clash"

        return {
            "subject_a": subj_a,
            "subject_b": subj_b,
            "then_a": then_a,
            "then_b": then_b,
            "contradiction": bool(contra),
            "reason": reason,
        }
