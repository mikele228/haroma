from typing import Dict, Any, Optional
from utils.module_base import ModuleBase
from utils.coerce_bool import env_flag
from core.Memory import MemoryForest, MemoryNode
import time
import uuid


class LoopMemoryLogger(ModuleBase):
    """Logs cognitive loop events to a memory forest."""

    _MAX_NODES_PER_BRANCH = 200

    def __init__(self, forest: Optional[MemoryForest] = None):
        super().__init__("LoopMemoryLogger")
        self.forest = forest or MemoryForest()

    def log_loop_event(
        self,
        loop_type: str,
        context: Dict[str, Any],
        event: str = "start",
        outcome: Optional[Dict[str, Any]] = None,
    ) -> str:
        moment_id = str(uuid.uuid4())
        tags = ["loop", loop_type, event]
        if outcome:
            tags.append("outcome")
        node = MemoryNode(
            moment_id=moment_id,
            content=f"{event.upper()} LOOP: {loop_type}",
            emotion=context.get("emotion", "neutral"),
            tags=tags,
        )
        self.forest.add_node("loop_tree", loop_type, node)
        try:
            tree = self.forest.trees.get("loop_tree")
            if tree:
                branch = tree.branches.get(loop_type)
                if branch and len(branch.nodes) > self._MAX_NODES_PER_BRANCH:
                    branch.nodes = branch.nodes[-self._MAX_NODES_PER_BRANCH :]
        except Exception as _e:
            print(f"[LoopMemoryLogger] branch trim error: {_e}", flush=True)
        if env_flag("HAROMA_LOOP_MEMORY_CONSOLE", False):
            print(
                f"[CognitiveLoop] memory_log loop_type={loop_type!r} event={event!r} "
                f"moment_id={moment_id}",
                flush=True,
            )
        return moment_id
