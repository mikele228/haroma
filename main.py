"""
Launches Elarion HTTP server (multi-agent architecture).

BootAgent initializes SharedResources and spawns Input, TrueSelf, Background,
and Persona agents. Entry: ``mind.elarion_server_v2``.

``python main.py --debug`` keeps the HTTP server on the main thread and skips the
optional live robot head (GUI or ASCII). Without ``--debug``, a live face may
appear (Windows / macOS / X11 / Wayland: Tk popup; else ASCII on stderr).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HaromaX6 Elarion cognitive server")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run the HTTP server on the main thread only; disable live robot head / stderr face",
    )
    args = parser.parse_args()

    from mind.deploy_config import load_dotenv

    load_dotenv()
    print("[HaromaX6] Elarion system ignition (multi-agent)...", flush=True)
    print("[HaromaX6] Architecture: Boot / Input / TrueSelf / Background / Persona", flush=True)
    print("[HaromaX6] MemoryForest: SHARED", flush=True)
    print("[HaromaX6] MessageBus: ARMED", flush=True)
    print("[HaromaX6] Soul Identity: LOADED", flush=True)
    from mind.elarion_server_v2 import launch

    launch(debug=args.debug)
