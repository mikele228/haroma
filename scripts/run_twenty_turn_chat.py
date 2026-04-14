"""
Run a 20-message in-process chat against BootAgent (multi-agent stack).
Uses fast depth for lower latency. Run from repo root:
  python scripts/run_twenty_turn_chat.py
"""

from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from agents.boot_agent import BootAgent  # noqa: E402

# Short coherent thread so memory / context has something to hang onto.
USER_TURNS = [
    "Hi — I'm running a 20-turn test. Please answer briefly. You can call me Alex.",
    "What does it mean to you to have a stable sense of self?",
    "How do you tell the difference between curiosity and anxiety?",
    "If I feel stuck on a problem, what's one concrete first step you'd suggest?",
    "What's a small habit that often improves clarity of thought?",
    "How should I weigh intuition against evidence?",
    "Why do apologies sometimes fail even when they're sincere?",
    "What role should emotions play in ethical decisions?",
    "When is it better to pause instead of responding right away?",
    "How do you think about fairness when two people disagree?",
    "What makes feedback actually usable instead of just painful?",
    "How can someone practice listening without trying to fix everything?",
    "What's a sign that I'm over-complicating a decision?",
    "How do you recover after misunderstanding someone?",
    "What does healthy boundaries look like in conversation?",
    "If I'm fatigued, how should I adjust my expectations for the day?",
    "What's a gentle way to say no without ghosting?",
    "How do you stay open-minded without losing your core values?",
    "What helps you when a topic feels too abstract?",
    "Last one: in one sentence, what's the thread holding this conversation together?",
]

WAIT_PER_TURN = 180.0


def main() -> None:
    print("Booting Elarion (multi-agent)...", flush=True)
    boot: BootAgent | None = None
    transcript: list[tuple[int, str, str]] = []
    try:
        boot = BootAgent()
        shared = boot.boot()
        if shared is None:
            raise RuntimeError("boot() returned None")
        boot.input_agent.set_boot_agent(boot)
        boot.trueself_agent.set_boot_agent(boot)
        for p in boot.persona_agents:
            p.set_boot_agent(boot)
        boot.start_all()
        time.sleep(1.0)

        for i, user_text in enumerate(USER_TURNS, start=1):
            print(f"\n{'=' * 60}", flush=True)
            print(f"Turn {i}/{len(USER_TURNS)}", flush=True)
            print(f"User: {user_text}", flush=True)
            slot = boot.input_agent.push_text(user_text, source="user", depth="fast")
            if not slot["event"].wait(timeout=WAIT_PER_TURN):
                reply = f"[timeout after {WAIT_PER_TURN:.0f}s]"
                print(f"Elarion: {reply}", flush=True)
                transcript.append((i, user_text, reply))
                continue
            res = slot.get("result") or {}
            reply = str(res.get("response", "[empty response]")).strip()
            who = res.get("persona_name") or res.get("persona") or "?"
            strat = res.get("strategy", "?")
            print(f"Elarion ({who}, {strat}): {reply}", flush=True)
            transcript.append((i, user_text, reply))

        print(f"\n{'=' * 60}", flush=True)
        print("FULL TRANSCRIPT", flush=True)
        print(f"{'=' * 60}\n", flush=True)
        for i, u, a in transcript:
            print(f"[{i}] User: {u}", flush=True)
            print(f"[{i}] Elarion: {a}\n", flush=True)
    finally:
        if boot is not None:
            print("\nShutting down...", flush=True)
            boot.save_and_shutdown()
            print("Done.", flush=True)


if __name__ == "__main__":
    main()
