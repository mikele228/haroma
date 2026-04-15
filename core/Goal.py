from typing import Dict, List, Any, Optional
from core.Memory import MemoryForest


class GoalCollapseDetector:
    def __init__(self, forest: MemoryForest):
        self.forest = forest

    def detect(self) -> Dict[str, Any]:
        goal_nodes = self._collect_nodes("goal_tree")
        action_nodes = self._collect_nodes("action_tree")

        # Map first tag to goal node (if any tags present)
        goal_ids = {n.tags[0]: n for n in goal_nodes if isinstance(n.tags, list) and n.tags}
        action_refs = {tag for a in action_nodes if isinstance(a.tags, list) for tag in a.tags}

        collapsed = []
        for gid, goal_node in goal_ids.items():
            if gid not in action_refs:
                collapsed.append(
                    {
                        "goal_id": gid,
                        "content": goal_node.content,
                        "confidence": goal_node.confidence,
                        "tags": goal_node.tags,
                    }
                )

        return {
            "total_goals": len(goal_ids),
            "fulfilled_goals": len(goal_ids) - len(collapsed),
            "collapsed_goals": collapsed,
        }

    def _collect_nodes(self, tree_name: str) -> List[Any]:
        if tree_name not in self.forest.trees:
            return []
        return [
            node
            for branch in self.forest.trees[tree_name].branches.values()
            for node in branch.nodes
        ]


from utils.module_base import ModuleBase
import threading
import time

# Single process-wide goal store so HTTP/agents, library controllers, and
# OrganizationalGoalBoard share one FIFO queue (minded architecture: Fuel).
_shared_goal_engine: Optional["GoalEngine"] = None
_shared_goal_lock = threading.Lock()


def get_shared_goal_engine() -> "GoalEngine":
    """Return the shared ``GoalEngine`` used by all ``GoalManager`` instances."""
    global _shared_goal_engine
    with _shared_goal_lock:
        if _shared_goal_engine is None:
            _shared_goal_engine = GoalEngine()
        return _shared_goal_engine


def reset_shared_goal_engine_for_tests() -> None:
    """Clear the singleton (pytest / isolated runs only)."""
    global _shared_goal_engine
    with _shared_goal_lock:
        if _shared_goal_engine is not None:
            _shared_goal_engine.reset()
            _shared_goal_engine = None


def _normalize_action_items(raw: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """Turn strings or dicts into ``{id, description, done}`` rows."""
    out: List[Dict[str, Any]] = []
    for i, it in enumerate(raw or []):
        if isinstance(it, str):
            aid = f"a_{i}"
            out.append({"id": aid, "description": it.strip(), "done": False})
        elif isinstance(it, dict):
            aid = str(it.get("id") or f"a_{i}")
            out.append(
                {
                    "id": aid,
                    "description": str(it.get("description") or "").strip(),
                    "done": bool(it.get("done")),
                }
            )
    return out


class GoalEngine(ModuleBase):
    """
    Tier MAX+ Engine: Manages symbolic goals, task priorities, activation strategies, and mission continuity.

    Input-derived goals live in a FIFO ``input_goal_queue`` and are
    prioritized in arrival order ahead of non-input goals.

    Hierarchical goals: optional ``child_goal_ids`` and ``action_items``. A goal
    is completed only when every child goal is completed and every action item
    is marked done; completing the last child auto-completes ancestors when
    their remaining constraints are satisfied.
    """

    def __init__(self, context_engine: Optional[Any] = None):
        super().__init__("GoalEngine")
        self.context_engine = context_engine
        self.load_state()

        self._lock = threading.RLock()
        self.goals: Dict[str, Dict[str, Any]] = self.state.get("goals", {})
        self.priorities: List[str] = self.state.get("priorities", [])
        self.input_goal_queue: List[str] = self.state.get("input_goal_queue", [])
        self.trace_log: Dict[str, List[Dict[str, Any]]] = self.state.get("trace_log", {})
        self.strategy: str = self.state.get("strategy", "default")
        self.last_decision: Optional[Dict[str, Any]] = self.state.get("last_decision", None)

    def align_goal_belief(self, goal, belief):
        if self.context_engine:
            return self.context_engine.fuzzy_alignment_scorer_handler(
                {"goal": goal, "belief": belief}
            )
        return {"score": None, "reason": "no context engine"}

    def register_goal(
        self,
        goal_id: str,
        description: str,
        priority: float = 0.5,
        source: str = "system",
        *,
        child_goal_ids: Optional[List[str]] = None,
        action_items: Optional[List[Any]] = None,
        parent_goal_id: Optional[str] = None,
    ):
        with self._lock:
            row: Dict[str, Any] = {
                "description": description,
                "priority": round(priority, 3),
                "source": source,
                "timestamp": time.time(),
                "completed": False,
                "child_goal_ids": list(child_goal_ids) if child_goal_ids else [],
                "action_items": _normalize_action_items(action_items),
            }
            if parent_goal_id:
                row["parent_goal_id"] = parent_goal_id
            self.goals[goal_id] = row
            self.state["goals"] = self.goals

    def bump_goal_priority(self, goal_id: str, delta: float) -> bool:
        """Adjust an existing goal's priority by *delta* (clamped to [0, 1])."""
        with self._lock:
            entry = self.goals.get(goal_id)
            if entry is None:
                return False
            try:
                p = float(entry.get("priority", 0.5)) + float(delta)
            except (TypeError, ValueError):
                return False
            p = max(0.0, min(1.0, p))
            entry["priority"] = round(p, 3)
            self.state["goals"] = self.goals
            return True

    def register_input_goal(
        self,
        goal_id: str,
        description: str,
        priority: float = 0.5,
        source: str = "input",
        meta: Optional[Dict[str, Any]] = None,
        *,
        child_goal_ids: Optional[List[str]] = None,
        action_items: Optional[List[Any]] = None,
        parent_goal_id: Optional[str] = None,
    ) -> None:
        """Register an input-derived goal and append to the FIFO queue."""
        with self._lock:
            self.goals[goal_id] = {
                "description": description,
                "priority": round(priority, 3),
                "source": source,
                "timestamp": time.time(),
                "completed": False,
                "input_meta": meta or {},
                "child_goal_ids": list(child_goal_ids) if child_goal_ids else [],
                "action_items": _normalize_action_items(action_items),
            }
            if parent_goal_id:
                self.goals[goal_id]["parent_goal_id"] = parent_goal_id
            if goal_id not in self.input_goal_queue:
                self.input_goal_queue.append(goal_id)
            self.state["goals"] = self.goals
            self.state["input_goal_queue"] = self.input_goal_queue

    def _blocked_by_incomplete_children(self, goal_id: str) -> bool:
        g = self.goals.get(goal_id)
        if g is None:
            return True
        for cid in g.get("child_goal_ids") or []:
            c = self.goals.get(cid)
            if c is None or not c.get("completed"):
                return True
        return False

    def _pending_actions_remain(self, goal_id: str) -> bool:
        g = self.goals.get(goal_id)
        if not g:
            return True
        for it in g.get("action_items") or []:
            if not it.get("done"):
                return True
        return False

    def _can_mark_complete(self, goal_id: str) -> bool:
        if self._blocked_by_incomplete_children(goal_id):
            return False
        if self._pending_actions_remain(goal_id):
            return False
        return True

    def _drain_completed_input_queue_heads(self) -> None:
        while self.input_goal_queue:
            head = self.input_goal_queue[0]
            head_entry = self.goals.get(head)
            if head_entry is None:
                self.input_goal_queue.pop(0)
                continue
            if head_entry.get("completed"):
                self.input_goal_queue.pop(0)
                continue
            break

    def _maybe_autocomplete_parents(self, completed_child_id: str) -> None:
        parents = [
            pid
            for pid, ent in self.goals.items()
            if completed_child_id in (ent.get("child_goal_ids") or [])
        ]
        for pid in parents:
            p = self.goals.get(pid)
            if p is None or p.get("completed"):
                continue
            if not self._can_mark_complete(pid):
                continue
            p["completed"] = True
            p["completed_at"] = time.time()
            self.state["goals"] = self.goals
            self._drain_completed_input_queue_heads()
            self._maybe_autocomplete_parents(pid)

    def complete_goal(self, goal_id: str) -> bool:
        """Mark a goal completed only when all child goals and action items are done.

        Parents are auto-completed when their last required child completes.
        """
        with self._lock:
            entry = self.goals.get(goal_id)
            if entry is None or entry.get("completed"):
                return False
            if not self._can_mark_complete(goal_id):
                return False
            entry["completed"] = True
            entry["completed_at"] = time.time()
            self.state["goals"] = self.goals
            self._drain_completed_input_queue_heads()
            self._maybe_autocomplete_parents(goal_id)
            return True

    def complete_input_goal(self, goal_id: str) -> bool:
        """Same as :meth:`complete_goal` (FIFO input goals use the same rules)."""
        return self.complete_goal(goal_id)

    def mark_action_done(self, goal_id: str, action_id: str) -> bool:
        """Mark one action item under *goal_id* done; may complete the goal and roll up."""
        with self._lock:
            g = self.goals.get(goal_id)
            if g is None or g.get("completed"):
                return False
            found = False
            for it in g.get("action_items") or []:
                if str(it.get("id")) == str(action_id):
                    it["done"] = True
                    found = True
                    break
            if not found:
                return False
            self.state["goals"] = self.goals
            if self._can_mark_complete(goal_id):
                g["completed"] = True
                g["completed_at"] = time.time()
                self.state["goals"] = self.goals
                self._drain_completed_input_queue_heads()
                self._maybe_autocomplete_parents(goal_id)
            return True

    def prioritize_workfront(self) -> List[str]:
        """Like :meth:`prioritize` but omits goals blocked by incomplete child goals."""
        self.prioritize()
        with self._lock:
            return [gid for gid in self.priorities if not self._blocked_by_incomplete_children(gid)]

    def current_input_goal(self) -> Optional[str]:
        """Return the oldest incomplete, unblocked-by-children input goal id, or None."""
        with self._lock:
            for gid in self.input_goal_queue:
                g = self.goals.get(gid)
                if g is None or g.get("completed"):
                    continue
                if self._blocked_by_incomplete_children(gid):
                    continue
                return gid
            return None

    def prioritize(self):
        """FIFO input goals first, then non-input goals by descending priority."""
        with self._lock:
            fifo = [
                gid
                for gid in self.input_goal_queue
                if gid in self.goals and not self.goals[gid].get("completed")
            ]
            fifo_set = set(fifo)
            non_input = sorted(
                [
                    (gid, g)
                    for gid, g in self.goals.items()
                    if gid not in fifo_set and not g.get("completed")
                ],
                key=lambda x: -x[1].get("priority", 0.0),
            )
            self.priorities = fifo + [gid for gid, _ in non_input]
            self.state["priorities"] = self.priorities
            return self.priorities

    def activate(self, strategy: Optional[str] = None):
        with self._lock:
            strategy = strategy or self.strategy
            self.strategy = strategy
            self.state["strategy"] = strategy

            if strategy == "max_priority" and self.priorities:
                chosen = self.priorities[0]
                self.last_decision = {
                    "goal": chosen,
                    "action": f"activate:{chosen}",
                    "timestamp": time.time(),
                }
                self.state["last_decision"] = self.last_decision
                return self.last_decision
            return {}

    def record_mission(self, agent: str, goal_id: str, outcome: str):
        entry = {"goal": goal_id, "outcome": outcome, "timestamp": time.time()}
        with self._lock:
            agent_log = self.trace_log.setdefault(agent, [])
            agent_log.append(entry)
            if len(agent_log) > 200:
                self.trace_log[agent] = agent_log[-200:]
            self.state["trace_log"] = self.trace_log
        olow = (outcome or "").strip().lower()
        if olow in ("completed", "success", "done", "succeeded"):
            self.complete_goal(goal_id)

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            top = self.priorities[0] if self.priorities else None
            return {
                "goal_count": len(self.goals),
                "top_priority": top,
                "strategy": self.strategy,
                "last_action": self.last_decision,
                "agents_tracked": len(self.trace_log),
            }

    def reset(self):
        with self._lock:
            self.goals.clear()
            self.priorities.clear()
            self.input_goal_queue.clear()
            self.trace_log.clear()
            self.strategy = "default"
            self.last_decision = None
            self.state.clear()

    def __repr__(self):
        return f"<GoalEngine goals={len(self.goals)} top={self.priorities[0] if self.priorities else None}>"

    # === Stubbed Handlers (Injectable Into SelfReflector) ===

    def echo_handler(self, data):
        pass

    def eternityloop_handler(self, data):
        pass

    def feedbackloop_handler(self, data):
        pass

    def loopentropyevaluator_handler(self, data):
        pass

    def transcendenceloop_handler(self, data):
        pass

    def integrityloop_handler(self, data):
        pass

    def loopclosure_handler(self, data):
        pass

    def selfaxiomloop_handler(self, data):
        pass

    def coherencematrix_handler(self, data):
        pass

    def purposealignment_handler(self, data):
        pass

    def metaalignmentmatrix_handler(self, data):
        pass

    def narrativerole_handler(self, data):
        pass

    def identitycollapseguard_handler(self, data):
        pass

    def identitydelta_handler(self, data):
        pass

    def identityechoamplifier_handler(self, data):
        pass

    def identityfusionmatrix_handler(self, data):
        pass

    def identity_handler(self, data):
        pass

    def rolemapper_handler(self, data):
        pass
