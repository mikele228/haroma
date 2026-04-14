"""
Organizational goal governance: board (personas) propose, president (TrueSelf)
arbitrates consensus, CEO (ActionAgent) executes the ratified mandate.

Personas attach ``board_goal_proposal`` to response payloads; TrueSelf records
proposals and calls ``president_try_ratify``. When enough distinct personas
align on the same normalized description, a mandate is issued and the CEO
ticks until completion.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Dict, Optional, Set

from core.cognitive_null import has_runtime_feature


def _normalize_key(description: str) -> str:
    t = " ".join(str(description).lower().strip().split())[:160]
    return hashlib.sha256(t.encode("utf-8", errors="ignore")).hexdigest()[:24]


class OrganizationalGoalBoard:
    """President / CEO goal arbitration state (thread-safe)."""

    def __init__(
        self,
        *,
        consensus_votes: int = 2,
        ceo_ticks_to_complete: int = 3,
    ):
        self._lock = threading.RLock()
        self.consensus_votes = max(1, int(consensus_votes))
        self.ceo_ticks_to_complete = max(1, int(ceo_ticks_to_complete))
        # norm_key -> aggregate proposal
        self._proposals: Dict[str, Dict[str, Any]] = {}
        self._mandate: Optional[Dict[str, Any]] = None

    def clear_proposals(self) -> None:
        with self._lock:
            self._proposals.clear()

    def record_proposal(self, persona_id: str, prop: Dict[str, Any]) -> None:
        """Record a persona's goal proposal (board member voice)."""
        if not prop or not isinstance(prop, dict):
            return
        gid = str(prop.get("goal_id", "")).strip()
        desc = str(prop.get("description", "")).strip()
        if not gid or not desc:
            return
        try:
            pri = float(prop.get("priority", 0.5))
            pri = max(0.0, min(1.0, pri))
        except (TypeError, ValueError):
            pri = 0.5
        stance = str(prop.get("stance", "propose")).strip().lower()
        nk = _normalize_key(desc)
        with self._lock:
            if self._mandate is not None:
                return
            slot = self._proposals.get(nk)
            if slot is None:
                self._proposals[nk] = {
                    "goal_id": gid,
                    "description": desc[:500],
                    "priority": pri,
                    "supporters": {persona_id},
                    "stance": stance,
                }
            else:
                slot["supporters"].add(persona_id)
                if pri > slot.get("priority", 0.0):
                    slot["priority"] = pri
                if stance == "propose" and slot.get("stance") != "propose":
                    slot["stance"] = "propose"

    def has_active_mandate(self) -> bool:
        with self._lock:
            return self._mandate is not None and self._mandate.get("status") == "executing"

    def get_mandate_summary_for_prompt(self) -> Optional[Dict[str, Any]]:
        """Short dict for LLM / episode prompts when mandate is active."""
        with self._lock:
            if self._mandate is None:
                return None
            return {
                "goal_id": self._mandate.get("goal_id"),
                "description": self._mandate.get("description"),
                "priority": self._mandate.get("priority", 0.9),
                "status": self._mandate.get("status"),
            }

    def president_try_ratify(self, shared: Any) -> bool:
        """If consensus reached, install mandate and register goal. Returns True if new mandate."""
        _mcopy: Optional[Dict[str, Any]] = None
        with self._lock:
            if self._mandate is not None:
                return False
            winner: Optional[Dict[str, Any]] = None
            for _nk, slot in self._proposals.items():
                n_sup = len(slot.get("supporters") or set())
                if n_sup >= self.consensus_votes:
                    if winner is None or slot.get("priority", 0) > winner.get("priority", 0):
                        winner = dict(slot)
            if winner is None:
                return False
            supporters: Set[str] = set(winner.get("supporters") or [])
            self._mandate = {
                "goal_id": winner["goal_id"],
                "description": winner["description"],
                "priority": winner.get("priority", 0.7),
                "status": "executing",
                "ratified_at": time.time(),
                "supporters": sorted(supporters),
                "ceo_ticks": 0,
            }
            self._proposals.clear()
            _mcopy = dict(self._mandate)

        # Register in shared goal engine (outside lock — may persist)
        try:
            g = getattr(shared, "goal", None)
            if has_runtime_feature(g, "register_goal"):
                g.register_goal(
                    _mcopy["goal_id"],
                    _mcopy["description"],
                    priority=min(1.0, float(_mcopy.get("priority", 0.7)) + 0.1),
                    source="board_mandate",
                )
                if has_runtime_feature(g, "prioritize"):
                    g.prioritize()
        except Exception as exc:
            print(f"[GoalBoard/President] register_goal failed: {exc}", flush=True)
        return True

    def tick_ceo_execution(self, shared: Any, ceo_agent_id: str) -> bool:
        """One CEO tick: progress mandate; return True if mandate just completed."""
        from core.Memory import MemoryNode

        completed = False
        with self._lock:
            if self._mandate is None or self._mandate.get("status") != "executing":
                return False
            m = self._mandate
            m["ceo_ticks"] = int(m.get("ceo_ticks", 0)) + 1
            ticks = m["ceo_ticks"]
            gid = m["goal_id"]
            desc = m["description"]
            done = ticks >= self.ceo_ticks_to_complete

        mem = getattr(shared, "memory", None)
        if mem is not None and hasattr(mem, "add_node"):
            try:
                node = MemoryNode(
                    content=(f"[CEO tick {ticks}] Executing board mandate: {gid} — {desc[:120]}"),
                    tags=["ceo", "board_mandate", gid[:40]],
                    confidence=0.85,
                )
                mem.add_node("action_tree", ceo_agent_id, node)
            except Exception:
                pass

        g = getattr(shared, "goal", None)
        if g is not None and hasattr(g, "record_mission"):
            try:
                g.record_mission(
                    ceo_agent_id,
                    gid,
                    "completed" if done else f"progress_{ticks}",
                )
            except Exception:
                pass

        if done and getattr(shared, "knowledge", None) is not None:
            try:
                kg = shared.knowledge
                if hasattr(kg, "integrate_world_state"):
                    kg.integrate_world_state(
                        [
                            {
                                "subject": gid,
                                "predicate": "completed_by",
                                "object": ceo_agent_id,
                                "confidence": 0.95,
                                "source": "ceo",
                            }
                        ],
                        cycle_id=getattr(shared, "cycle_count", 0) or 0,
                    )
            except Exception:
                pass

        if done:
            with self._lock:
                if self._mandate is not None:
                    self._mandate["status"] = "completed"
                    self._mandate["completed_at"] = time.time()
                self._mandate = None
            completed = True
            print(
                f"[GoalBoard/CEO] Mandate {gid!r} completed by {ceo_agent_id}",
                flush=True,
            )
        return completed

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "consensus_votes": self.consensus_votes,
                "open_proposals": len(self._proposals),
                "mandate": dict(self._mandate) if self._mandate else None,
            }
