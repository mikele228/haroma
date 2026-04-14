from typing import Dict, List, Any
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
from typing import Dict, List, Any, Optional
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


class GoalEngine(ModuleBase):
    """
    Tier MAX+ Engine: Manages symbolic goals, task priorities, activation strategies, and mission continuity.

    Input-derived goals live in a FIFO ``input_goal_queue`` and are
    prioritized in arrival order ahead of non-input goals.
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
        self, goal_id: str, description: str, priority: float = 0.5, source: str = "system"
    ):
        with self._lock:
            self.goals[goal_id] = {
                "description": description,
                "priority": round(priority, 3),
                "source": source,
                "timestamp": time.time(),
            }
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
            }
            if goal_id not in self.input_goal_queue:
                self.input_goal_queue.append(goal_id)
            self.state["goals"] = self.goals
            self.state["input_goal_queue"] = self.input_goal_queue

    def complete_input_goal(self, goal_id: str) -> bool:
        """Mark an input goal as completed. Returns True if found and marked."""
        with self._lock:
            entry = self.goals.get(goal_id)
            if entry is None or entry.get("completed"):
                return False
            entry["completed"] = True
            # Drain completed (or orphaned) ids from the head of the queue
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
            self.state["goals"] = self.goals
            self.state["input_goal_queue"] = self.input_goal_queue
            return True

    def current_input_goal(self) -> Optional[str]:
        """Return the oldest incomplete input goal id, or None."""
        with self._lock:
            for gid in self.input_goal_queue:
                g = self.goals.get(gid)
                if g is not None and not g.get("completed"):
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
