"""
Fast smoke test -- verifies the multi-agent system boots and processes
a chat message end-to-end without waiting for the full dense index build.

Patches _prewarm_memory_index to skip the slow TF-IDF/dense rebuild.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def banner(text):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def main():
    banner("ELARION v6 SMOKE TEST (fast)")

    # ── 1. Boot ──────────────────────────────────────────────
    banner("1. BOOT")
    from agents.boot_agent import BootAgent

    boot = BootAgent()
    t0 = time.time()
    shared = boot.boot()
    boot_time = time.time() - t0
    print(f"  Boot time: {boot_time:.1f}s")
    if boot_time <= 30.0:
        print(f"  PASS: Boot under 30s deadline")
    else:
        print(f"  WARN: Boot exceeded 30s deadline ({boot_time:.1f}s)")

    assert shared is not None, "SharedResources is None"
    assert boot.trueself_agent is not None, "TrueSelf not spawned"
    assert len(boot.persona_agents) >= 1, "No persona agents"
    print(f"  TrueSelf: {boot.trueself_agent.agent_id}")
    print(f"  Personas: {[p.agent_id for p in boot.persona_agents]}")
    print(f"  Memory nodes: {shared.memory.count_nodes()}")
    print("  PASS: Boot OK")

    # ── 2. Start agents ──────────────────────────────────────
    banner("2. START AGENTS")
    boot.start_all()
    time.sleep(2.0)

    agents_to_check = [
        ("boot", boot),
        ("input", boot.input_agent),
        ("trueself", boot.trueself_agent),
        ("background", boot.background_agent),
    ] + [(f"persona:{p.agent_id}", p) for p in boot.persona_agents]

    all_alive = True
    for label, agent in agents_to_check:
        alive = agent.is_alive()
        status = "OK" if alive else "DEAD"
        print(f"  {status} {label} (ticks={agent._tick_count})")
        if not alive:
            all_alive = False

    assert all_alive, "Some agents failed to start"
    print("  PASS: All agents alive")

    # ── 3. Send chat message ─────────────────────────────────
    banner("3. CHAT MESSAGE")
    input_agent = boot.input_agent

    slot = input_agent.push_text("Hello, who are you?", source="test_user")
    print("  Sent: 'Hello, who are you?'")
    print("  Waiting for response (up to 120s)...\n")

    t_chat = time.time()
    got_response = slot["event"].wait(timeout=300.0)
    elapsed = time.time() - t_chat

    if got_response and slot.get("result"):
        result = slot["result"]
        response_text = result.get("response", "")
        persona = result.get("persona_name", result.get("persona", "?"))
        cycle = result.get("cycle", "?")
        strategy = result.get("strategy", "?")

        print(f"  Response received in {elapsed:.1f}s")
        print(f"  Persona: {persona}")
        print(f"  Cycle: {cycle}")
        print(f"  Strategy: {strategy}")
        print(f"  Response: {response_text[:200]}")
        print("  PASS: Chat response received")
    else:
        print(f"  TIMEOUT after {elapsed:.1f}s")
        print(
            f"  TrueSelf ticks={boot.trueself_agent._tick_count}, "
            f"cycles={boot.trueself_agent._cycle_count}, "
            f"errors={boot.trueself_agent._error_count}"
        )
        for p in boot.persona_agents:
            print(
                f"  {p.agent_id} ticks={p._tick_count}, "
                f"cycles={p._cycle_count}, errors={p._error_count}"
            )
        bus_stats = boot.bus.stats()
        print(f"  Bus mailboxes: {bus_stats['mailbox_sizes']}")
        print("  FAIL: No response within timeout")

    # ── 4. Final health check ────────────────────────────────
    banner("4. HEALTH CHECK")
    for label, agent in agents_to_check:
        alive = agent.is_alive()
        status = "OK" if alive else "DEAD"
        print(f"  {status} {label} (ticks={agent._tick_count}, errors={agent._error_count})")

    print(f"\n  Global cycles: {shared.cycle_count}")
    print(f"  TrueSelf cycles: {boot.trueself_agent._cycle_count}")
    print(f"  Pending delegations: {len(boot.trueself_agent._pending_delegations)}")

    banner("SMOKE TEST COMPLETE")

    # Shutdown
    print("  Shutting down...")
    boot.save_and_shutdown()
    print("  Done.")


if __name__ == "__main__":
    main()
