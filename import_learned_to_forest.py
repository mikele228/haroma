# === import_learned_to_forest.py ===
# Loads a learned.mem file and installs it into MemoryForest under learn_tree

import os
import json
import base64
from core import MemoryForest
from core.Memory import MemoryNode


def decode_learned(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(base64.b64decode(f.read()).decode())


def import_learned(learned_path: str, agent_id: str = "common") -> MemoryForest:
    print(f"📥 Loading learned memory from: {learned_path}")
    learned_data = decode_learned(learned_path)

    forest = MemoryForest()
    forest.add_node("learn_tree", agent_id, MemoryNode(content=learned_data))

    print(f"🌱 Installed into learn_tree → branch={agent_id} with {len(learned_data)} records.")
    return forest


if __name__ == "__main__":
    path = input("Path to learned.mem: ").strip()
    agent = input("Agent ID (default=common): ").strip() or "common"
    forest = import_learned(path, agent)
