"""OrganRegistry — maps HaromaX6 modules into the X7 15-organ taxonomy.

Each organ groups related modules for discoverability, health monitoring,
and eventual modular loading/unloading. The registry is populated at boot
from a static catalog, then enriched with live module references so
``organ_status()`` can report real-time health per organ.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from core.CellRoles import get_role_label, summary as cell_roles_summary


# ═══════════════════════════════════════════════════════════════════════
# Static catalog — organ_id -> metadata
# ═══════════════════════════════════════════════════════════════════════

ORGAN_CATALOG: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "Symbolic Core",
        "modules": [
            "knowledge",
            "knowledge_manager",
            "reasoning",
            "reasoning_manager",
            "encoder",
        ],
    },
    2: {
        "name": "Emotional Regulation",
        "modules": [
            "emotion",
            "appraisal",
            "appraisal_manager",
            "modulation",
            "modulation_manager",
            "drift",
        ],
    },
    3: {
        "name": "Belief & Ethics",
        "modules": [
            "value",
            "law",
            "myth",
        ],
    },
    4: {
        "name": "Cognitive Management",
        "modules": [
            "metacognition",
            "metacognition_manager",
            "curiosity",
            "curiosity_manager",
            "attention",
            "attention_manager",
            "goal_synthesizer_manager",
            "process_gate",
            "backbone",
            "workspace",
            "training_scheduler",
            "arch_searcher",
        ],
    },
    5: {
        "name": "Identity & Agents",
        "modules": [
            "soul_binder",
            "identity",
            "interlocutor_model",
            "mental_simulator",
        ],
    },
    6: {
        "name": "Memory",
        "modules": [
            "memory",
            "memory_core",
            "working_memory",
            "persistence",
            "loop_logger",
            "memory_summarizer",
        ],
    },
    7: {
        "name": "Dream & Imagination",
        "modules": [
            "dream_consolidator",
            "dream",
            "imagination",
            "imagination_manager",
        ],
    },
    8: {
        "name": "Goals & Motivation",
        "modules": [
            "goal",
            "drives",
            "goal_synthesizer",
            "collapse",
        ],
    },
    9: {
        "name": "Conflict Resolution",
        "modules": [
            "fusion",
            "counterfactual",
            "counterfactual_manager",
            "reconciliation",
        ],
    },
    10: {
        "name": "Tagging & Classification",
        "modules": [
            "perception",
            "discourse",
        ],
    },
    11: {
        "name": "Predictive & Temporal",
        "modules": [
            "temporal",
            "temporal_manager",
            "self_model",
            "self_model_manager",
            "grounder",
        ],
    },
    12: {
        "name": "Meta-Reflection",
        "modules": [
            "reflector",
            "symbolic_loop",
            "memory_summarizer",
        ],
    },
    13: {
        "name": "Stability & Integrity",
        "modules": [
            "symbolic_queue",
            "fingerprint_engine",
        ],
    },
    14: {
        "name": "Narrative & Dialogue",
        "modules": [
            "narrative",
            "forecast",
            "conversation",
            "llm_manager",
            "composer",
            "action_generator",
        ],
    },
    15: {
        "name": "Bootstrapping",
        "modules": [
            "fabric",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════


class OrganRegistry:
    """Runtime registry mapping live module references to the 15-organ taxonomy."""

    def __init__(self):
        self._organs: Dict[int, Dict[str, Any]] = {}
        self._module_refs: Dict[str, Any] = {}
        self._module_organ: Dict[str, int] = {}
        self._error_counts: Dict[str, int] = {}
        self._last_active: Dict[str, float] = {}
        self._boot_time = time.time()

        for organ_id, meta in ORGAN_CATALOG.items():
            self._organs[organ_id] = {
                "name": meta["name"],
                "modules": list(meta["modules"]),
            }
            for mod in meta["modules"]:
                self._module_organ[mod] = organ_id

    def register_module(self, name: str, ref: Any) -> None:
        """Register a live module reference."""
        self._module_refs[name] = ref
        self._last_active[name] = time.time()

    def record_error(self, module_name: str) -> None:
        self._error_counts[module_name] = self._error_counts.get(module_name, 0) + 1

    def touch(self, module_name: str) -> None:
        """Mark a module as active this cycle."""
        self._last_active[module_name] = time.time()

    def get_organ(self, organ_id: int) -> Optional[Dict[str, Any]]:
        return self._organs.get(organ_id)

    def get_organ_for_module(self, module_name: str) -> Optional[int]:
        return self._module_organ.get(module_name)

    def organ_status(self) -> Dict[int, Dict[str, Any]]:
        """Per-organ health: module count, registered count, errors, last active."""
        result: Dict[int, Dict[str, Any]] = {}
        for organ_id, meta in self._organs.items():
            modules = meta["modules"]
            registered = [m for m in modules if m in self._module_refs]
            total_errors = sum(self._error_counts.get(m, 0) for m in modules)
            last_active_times = [
                self._last_active.get(m, 0.0) for m in modules if m in self._last_active
            ]
            last_active = max(last_active_times) if last_active_times else 0.0

            result[organ_id] = {
                "name": meta["name"],
                "module_count": len(modules),
                "registered": len(registered),
                "errors": total_errors,
                "last_active": last_active,
                "healthy": len(registered) > 0 and total_errors < 10,
                "cell_roles": {m: get_role_label(m) for m in registered},
            }
        return result

    def summary(self) -> Dict[str, Any]:
        """Compact summary for API responses."""
        status = self.organ_status()
        healthy_count = sum(1 for v in status.values() if v["healthy"])
        total_registered = sum(v["registered"] for v in status.values())
        total_errors = sum(v["errors"] for v in status.values())
        return {
            "organ_count": len(self._organs),
            "healthy_organs": healthy_count,
            "total_registered_modules": total_registered,
            "total_errors": total_errors,
            "uptime_seconds": round(time.time() - self._boot_time, 1),
            "cell_roles_catalog": cell_roles_summary(),
            "organs": {
                oid: {
                    "name": v["name"],
                    "healthy": v["healthy"],
                    "registered": v["registered"],
                    "module_count": v["module_count"],
                }
                for oid, v in status.items()
            },
        }
