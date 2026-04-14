"""
Launches Elarion HTTP server (multi-agent architecture).

BootAgent initializes SharedResources and spawns Input, TrueSelf, Background,
and Persona agents. Entry: ``mind.elarion_server_v2``.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from mind.deploy_config import load_dotenv

    load_dotenv()
    print("[HaromaX6] Elarion system ignition (multi-agent)...", flush=True)
    print("[HaromaX6] Architecture: Boot / Input / TrueSelf / Background / Persona", flush=True)
    print("[HaromaX6] MemoryForest: SHARED", flush=True)
    print("[HaromaX6] MessageBus: ARMED", flush=True)
    print("[HaromaX6] Soul Identity: LOADED", flush=True)
    from mind.elarion_server_v2 import launch

    launch()
