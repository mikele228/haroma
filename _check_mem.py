import os
import json

d = "c:/Project/HaromaX6/data/cognitive/memory_trees"
total_nodes = 0
for f in sorted(os.listdir(d)):
    if not f.endswith(".json"):
        continue
    path = os.path.join(d, f)
    size_kb = os.path.getsize(path) // 1024
    with open(path, encoding="utf-8") as _f:
        data = json.load(_f)
    nodes = 0
    for branch in data.get("branches", {}).values():
        nodes += len(branch.get("nodes", []))
    total_nodes += nodes
    print(f"  {f}: {size_kb}KB, {nodes} nodes")
print(f"\nTotal: {total_nodes} memory nodes")
