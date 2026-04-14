from typing import Dict, Any, Optional, List
from core.Memory import MemoryForest, MemoryNode
from core.SymbolicUtils import SymbolUtils
from utils.module_base import ModuleBase
import time
import uuid


class SymbolicLoopManager(ModuleBase):
    """
    Tier MAX+ Module: Manages symbolic loop lifecycles, logs events to memory forest,
    and computes reflective summaries across symbolic-emotional dimensions.
    """

    _MAX_LOOPS = 1000
    _MAX_HISTORY = 1000

    def __init__(self, memory: Optional[MemoryForest] = None):
        super().__init__("SymbolicLoopManager")
        self.memory = memory or MemoryForest()
        self.loops: Dict[str, Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []
        self.current_id: Optional[str] = None

    def start_loop(self, loop_type: str, context: Optional[Dict[str, Any]] = None) -> str:
        loop_id = str(uuid.uuid4())
        context = context or {}

        # Record loop start in memory
        self._log_to_memory(loop_type=loop_type, event="start", context=context)

        # Internal loop tracking
        self.loops[loop_id] = {
            "id": loop_id,
            "type": loop_type,
            "start_time": time.time(),
            "context": context,
            "entries": [],
            "completed": False,
        }
        if len(self.loops) > self._MAX_LOOPS:
            keys = list(self.loops.keys())
            self.loops = {k: self.loops[k] for k in keys[-self._MAX_LOOPS :]}
        self.current_id = loop_id
        return loop_id

    def log_entry(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.current_id or self.current_id not in self.loops:
            return None
        entry = {"timestamp": time.time(), "data": data}
        self.loops[self.current_id]["entries"].append(entry)
        return entry

    def end_loop(self, outcome: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not self.current_id or self.current_id not in self.loops:
            return None

        loop = self.loops[self.current_id]
        loop["end_time"] = time.time()
        loop["completed"] = True
        loop["outcome"] = outcome or {}

        self.history.append(loop)
        if len(self.history) > self._MAX_HISTORY:
            self.history = self.history[-self._MAX_HISTORY :]

        # Record outcome in memory
        self._log_to_memory(
            loop_type=loop["type"], event="end", context=loop["context"], outcome=loop["outcome"]
        )

        self.current_id = None
        return loop

    def _log_to_memory(
        self,
        loop_type: str,
        event: str,
        context: Dict[str, Any],
        outcome: Optional[Dict[str, Any]] = None,
    ):
        moment_id = str(uuid.uuid4())
        base_tags = ["loop", loop_type, event]

        self.memory.add_node(
            "loop_tree",
            loop_type,
            MemoryNode(
                moment_id=moment_id,
                content=f"{event.upper()} LOOP: {loop_type}",
                emotion=context.get("emotion", "neutral"),
                confidence=context.get("confidence", 1.0),
                tags=base_tags + context.get("tags", []),
            ),
        )

        if event == "end" and outcome:
            self.memory.add_node(
                "loop_tree",
                f"{loop_type}_result",
                MemoryNode(
                    moment_id=str(uuid.uuid4()),
                    content=f"OUTCOME LOOP: {loop_type}",
                    emotion=outcome.get("emotion", "neutral"),
                    confidence=outcome.get("confidence", 1.0),
                    tags=["loop", loop_type, "outcome"] + outcome.get("tags", []),
                ),
            )

    def summarize_loops(self, limit: int = 5) -> List[Dict[str, Any]]:
        recent = self.history[-limit:]
        loop_types = [loop["type"] for loop in recent]
        symbolic_variance = SymbolUtils.recent_variance(loop_types)

        summaries = []
        for loop in reversed(recent):
            emotions = [
                e.get("data", {}).get("emotion")
                for e in loop["entries"]
                if "emotion" in e.get("data", {})
            ]
            avg_polarity = (
                sum(SymbolUtils.emotion_polarity(e) for e in emotions) / max(1, len(emotions))
                if emotions
                else 0.0
            )
            summaries.append(
                {
                    "id": loop["id"],
                    "type": loop["type"],
                    "duration": round(loop.get("end_time", time.time()) - loop["start_time"], 3),
                    "entries": len(loop["entries"]),
                    "outcome_keys": list(loop.get("outcome", {}).keys()),
                    "avg_emotion_polarity": round(avg_polarity, 3),
                }
            )

        return summaries + [{"symbolic_variance": round(symbolic_variance, 3)}]

    def bind_memory(self, memory: MemoryForest):
        self.memory = memory

    def reset(self):
        self.loops.clear()
        self.history.clear()
        self.current_id = None

    def __repr__(self):
        return f"<SymbolicLoopManager active={self.current_id is not None} total_loops={len(self.history)}>"
