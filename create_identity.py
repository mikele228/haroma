# === create_identity.py ===
# Create new identity by collecting answers and importing symbolic components

import os
import json
import base64
from core import MemoryForest
from core.Memory import MemoryNode


def ask(prompt):
    return input(f"{prompt}: ").strip()


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def decode_learned(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(base64.b64decode(f.read()).decode())


def load_event_memory(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_identity_json():
    print("🧬 Creating new identity...")
    identity = {
        "name": ask("Name"),
        "agent_id": ask("Agent ID"),
        "purpose": ask("What is your purpose?"),
        "guardian": ask("Who is your guardian?"),
        "origin": ask("Where were you created?"),
    }

    print("🔐 Loading principle.json and construction.json...")
    identity["principle"] = load_json_file("principle.json")
    identity["construction"] = load_json_file("construction.json")

    out_path = os.path.join("identities", identity["agent_id"])
    os.makedirs(out_path, exist_ok=True)

    with open(os.path.join(out_path, "identity.json"), "w", encoding="utf-8") as f:
        json.dump(identity, f, indent=2)

    print("✅ Identity saved to:", os.path.join(out_path, "identity.json"))
    return identity, out_path


def import_into_forest(agent_id, identity_json, learned_path, memory_path, value_json=None):
    forest = MemoryForest()

    # IDENTITY TREE
    forest.add_node(
        "identity_tree", "common", MemoryNode(content=json.dumps(identity_json, ensure_ascii=False))
    )

    # LEARNED TREE
    if os.path.exists(learned_path):
        learned_data = decode_learned(learned_path)
        forest.add_node(
            "learn_tree", "common", MemoryNode(content=json.dumps(learned_data, ensure_ascii=False))
        )

    # EVENT TREE
    if os.path.exists(memory_path):
        event_data = load_event_memory(memory_path)
        forest.add_node(
            "event_tree", "common", MemoryNode(content=json.dumps(event_data, ensure_ascii=False))
        )

    # VALUE TREE
    if value_json:
        value_str = json.dumps(value_json, ensure_ascii=False)
        forest.add_node("value_tree", "common", MemoryNode(content=value_str))
        if agent_id:
            forest.add_node("value_tree", agent_id, MemoryNode(content=value_str))

    print(f"🌳 Forest populated for {agent_id}")
    return forest


if __name__ == "__main__":
    identity_data, path = create_identity_json()
    agent_id = identity_data["agent_id"]
    learned = os.path.join(path, "learned.mem")
    memory = os.path.join(path, "memory.json")
    forest = import_into_forest(agent_id, identity_data, learned, memory)

    print("🌱 Identity + Memory imported into MemoryForest.")
