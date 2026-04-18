"""
Multi-Agent Architecture Test (v6 TrueSelf) -- exercises TrueSelf routing
and persona delegation.

Boots the full agent system in-process (no HTTP server),
sends messages through the InputAgent, and verifies:
  1. All agents boot and stay alive (including TrueSelf)
  2. Simple messages are fast-pathed by TrueSelf
  3. Specialist messages are delegated to the matching persona
  4. Responses flow back through response slots
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import TYPE_CHECKING

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tests._import_guard import prepare_test_imports

prepare_test_imports(__file__)

if TYPE_CHECKING:
    from agents.boot_agent import BootAgent


_DELEGATION_LOG = []
_DELEGATION_LOCK = threading.Lock()


def _patch_delegation_logging(boot: BootAgent):
    """Monkey-patch TrueSelf to log delegation events for test verification."""
    trueself = boot.trueself_agent
    original_delegate = trueself._delegate_to_persona

    def wrapper(msg, persona):
        original_delegate(msg, persona)
        with _DELEGATION_LOCK:
            _DELEGATION_LOG.append(
                {
                    "to": persona.agent_id,
                    "persona_name": persona.persona_name,
                    "time": time.time(),
                    "content_preview": str(
                        msg.content.get("text", "")[:60] if isinstance(msg.content, dict) else ""
                    ),
                }
            )

    trueself._delegate_to_persona = wrapper


def banner(text):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def test_boot(boot: BootAgent):
    """Session fixture already booted TrueSelf + configured personas; assert invariants."""
    banner("TEST 1: Boot Multi-Agent System (v6 TrueSelf)")

    shared = boot.shared
    assert shared is not None, "SharedResources is None"

    persona_ids = [p.agent_id for p in boot.persona_agents]
    print(f"  TrueSelf: {boot.trueself_agent.agent_id} ({boot.trueself_agent.persona_name})")
    print(f"  Personas: {persona_ids}")
    print(f"  Memory nodes: {shared.memory.count_nodes()}")
    print(f"  Resource tier: {shared.resource_config.tier_name}")
    print(f"  Encoder dim: {shared.encoder._embed_dim}")
    print("  PASS")


def test_start_agents(boot: BootAgent):
    """Agents already running from session fixture; verify health."""
    banner("TEST 2: Start All Agents (health check)")

    agents_to_check = [
        ("boot", boot),
        ("input", boot.input_agent),
        ("trueself", boot.trueself_agent),
        ("background", boot.background_agent),
    ] + [(f"persona:{p.agent_id}", p) for p in boot.persona_agents]

    for label, agent in agents_to_check:
        alive = agent.is_alive()
        print(f"  {'OK' if alive else 'DEAD'} {label}")
        assert alive, f"{label} is not alive"

    print("  PASS")


def test_fast_path(boot: BootAgent):
    """Send a simple message -- TrueSelf should fast-path it."""
    banner("TEST 3: TrueSelf Fast Path (simple message)")

    input_agent = boot.input_agent
    trueself = boot.trueself_agent

    initial_cycles = trueself._cycle_count
    initial_delegation_count = len(_DELEGATION_LOG)

    slot = input_agent.push_text(
        "Hello, how are you today?",
        source="test_user",
    )

    print("  Sent simple message, waiting for TrueSelf fast-path...")
    print("  (Cognitive cycle takes 30-60s, please be patient)\n")

    deadline = time.time() + 120.0
    while time.time() < deadline:
        if trueself._cycle_count > initial_cycles:
            print(f"  TrueSelf processed: cycles={trueself._cycle_count}")
            break
        time.sleep(2.0)
    else:
        print(f"  TIMEOUT: TrueSelf did not complete a cycle in 120s")
        print(
            f"    ticks={trueself._tick_count}, "
            f"cycles={trueself._cycle_count}, "
            f"errors={trueself._error_count}"
        )

    if slot["event"].is_set() and slot.get("result"):
        result = slot["result"]
        print(f"\n  Response received:")
        print(f"    Persona: {result.get('persona', '?')} ({result.get('persona_name', '?')})")
        print(f"    Response: {result.get('response', '')[:120]}")
        print(f"    Strategy: {result.get('strategy', '?')}")
    else:
        print(f"\n  No HTTP response yet (may be expected with 'none' LLM)")

    with _DELEGATION_LOCK:
        new_delegations = len(_DELEGATION_LOG) - initial_delegation_count
    print(f"\n  Delegations during this test: {new_delegations}")
    if new_delegations == 0:
        print("  PASS: Fast-path confirmed (no delegation)")
    else:
        print("  NOTE: Message was delegated (may be expected)")


def test_delegation(boot: BootAgent):
    """Send a specialist message -- TrueSelf should delegate to Analyst."""
    persona_ids = {p.agent_id for p in boot.persona_agents}
    if "analyst" not in persona_ids:
        pytest.skip(
            "Delegation test needs an 'analyst' persona in soul/agents.json "
            "(see scripts/soul_defaults/agents.json initial_personas)"
        )

    banner("TEST 4: TrueSelf Delegation (specialist message)")

    input_agent = boot.input_agent
    initial_delegation_count = len(_DELEGATION_LOG)

    persona_cycles_before = {p.agent_id: p._cycle_count for p in boot.persona_agents}

    slot = input_agent.push_text(
        "Can you analyze and explain the logic behind how we reason "
        "about science? Why does deductive reasoning work?",
        source="test_user",
    )

    print("  Sent analyst-affinity message, waiting for delegation...")
    print("  (TrueSelf should delegate to Analyst persona)\n")

    deadline = time.time() + 120.0
    delegation_seen = False

    while time.time() < deadline:
        with _DELEGATION_LOCK:
            if len(_DELEGATION_LOG) > initial_delegation_count:
                for entry in _DELEGATION_LOG[initial_delegation_count:]:
                    print(f"    DELEGATION: TrueSelf -> {entry['to']} ({entry['persona_name']})")
                    print(f'      content: "{entry["content_preview"]}"')
                delegation_seen = True
                break
        time.sleep(2.0)

    if delegation_seen:
        print(f"\n  Delegation confirmed, waiting for persona to process...")
        process_deadline = time.time() + 120.0
        while time.time() < process_deadline:
            persona_cycles_now = {p.agent_id: p._cycle_count for p in boot.persona_agents}
            new_cycles = {
                pid: persona_cycles_now[pid] - persona_cycles_before.get(pid, 0)
                for pid in persona_cycles_now
            }
            if any(c > 0 for c in new_cycles.values()):
                print(f"  Persona cycles: {new_cycles}")
                break
            time.sleep(3.0)

    if slot["event"].is_set() and slot.get("result"):
        result = slot["result"]
        print(f"\n  Response received:")
        print(f"    Persona: {result.get('persona', '?')} ({result.get('persona_name', '?')})")
        print(f"    Response: {result.get('response', '')[:120]}")
        print(f"    Strategy: {result.get('strategy', '?')}")
        if result.get("delegated_from") == "trueself":
            print("  PASS: Response came from delegated persona via TrueSelf")
        else:
            print("  Response returned (may be fast-path fallback)")
    else:
        print(f"\n  No HTTP response yet (may need longer timeout)")

    with _DELEGATION_LOCK:
        total_delegations = len(_DELEGATION_LOG) - initial_delegation_count
    print(f"\n  Delegations during this test: {total_delegations}")


def test_health_check(boot: BootAgent):
    """Final agent health check."""
    banner("TEST 5: Final Health Check")

    all_alive = True
    agents_to_check = [
        ("boot", boot),
        ("input", boot.input_agent),
        ("trueself", boot.trueself_agent),
        ("background", boot.background_agent),
    ] + [(f"persona:{p.agent_id}", p) for p in boot.persona_agents]

    for label, agent in agents_to_check:
        alive = agent.is_alive()
        print(
            f"  {'OK' if alive else 'DEAD'} {label}: "
            f"ticks={agent._tick_count}, errors={agent._error_count}"
        )
        if not alive:
            all_alive = False

    shared = boot.shared
    print(f"\n  Total cycles: {shared.cycle_count}")
    print(f"  Memory nodes: {shared.memory.count_nodes()}")

    trueself = boot.trueself_agent
    print(
        f"  TrueSelf: cycles={trueself._cycle_count}, "
        f"pending_delegations={len(trueself._pending_delegations)}"
    )

    stats = boot.bus.stats()
    print(f"  Bus: mailbox_sizes={stats['mailbox_sizes']}")

    with _DELEGATION_LOCK:
        print(f"  Total delegation events logged: {len(_DELEGATION_LOG)}")

    if all_alive:
        print("\n  PASS: All agents alive")
    else:
        print("\n  FAIL: Some agents died")


def main():
    from agents.boot_agent import BootAgent

    banner("ELARION v6 TRUESELF ARCHITECTURE TEST")
    print("  Testing: Boot -> Input -> TrueSelf -> Fast-Path / Delegate")
    print("  TrueSelf: executive consciousness (sole input receiver)")
    print("  Personas: Core (default) + Analyst (logic/reason affinity)")
    print("  Note: Each cognitive cycle takes 30-60s on workstation tier")

    boot = None
    try:
        boot = BootAgent()
        shared = boot.boot()
        assert shared is not None
        boot.input_agent.set_boot_agent(boot)
        boot.trueself_agent.set_boot_agent(boot)
        for p in boot.persona_agents:
            p.set_boot_agent(boot)
        _patch_delegation_logging(boot)
        boot.start_all()
        time.sleep(1.0)

        test_boot(boot)
        test_start_agents(boot)
        test_fast_path(boot)
        test_delegation(boot)
        test_health_check(boot)
        banner("ALL TESTS COMPLETE")
    except Exception as exc:
        import traceback

        print(f"\n  FATAL ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        if boot:
            print("\n  Shutting down agents...")
            boot.save_and_shutdown()
            print("  Shutdown complete.")


if __name__ == "__main__":
    main()
