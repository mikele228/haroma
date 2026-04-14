import time
from typing import List, Dict, Any
from collections import defaultdict, deque

global_event_bus = {}


class ModuleBase:
    """Reboot-safe base class with signal history, state management, and metrics."""

    def __init__(self, module_name: str = None):
        self.module_name = module_name or self.__class__.__name__
        self.signal_history: deque = deque(maxlen=200)
        self.state: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = defaultdict(int)

    def load_state(self):
        """Load state from persistent storage. Override in subclasses."""
        pass

    def save_state(self):
        """Save state to persistent storage. Override in subclasses."""
        pass

    def get_merged_branch(self, memory_engine, tree: str, agent_id: str):
        """Merge memory from 'common' + agent_id in the given tree."""
        tree_obj = memory_engine.trees.get(tree)
        if not tree_obj:
            return []
        common = tree_obj.branches.get("common")
        agent = tree_obj.branches.get(agent_id)
        common_nodes = common.nodes if common else []
        agent_nodes = agent.nodes if agent else []
        return common_nodes + agent_nodes

    def get_agents(self, memory_engine):
        """Get list of agent IDs from memory engine's agent_tree."""
        if not memory_engine:
            return []
        trees = getattr(memory_engine, "trees", None)
        if not trees:
            return []
        agent_tree = trees.get("agent_tree")
        if not agent_tree:
            return []
        return [bid for bid in agent_tree.branches.keys() if bid != "common"]

    def install(self, moment, tree, branch):
        """Install a moment into memory. Override for actual memory integration."""
        pass

    def __repr__(self):
        return f"<{self.module_name}>"
