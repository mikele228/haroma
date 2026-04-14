# HaromaX6 Documentation

**Elarion** is a symbolic cognitive agent framework built on forest-rooted memory, gradient-driven processing, and soul-seeded identity. This documentation covers everything from first launch to deep architectural internals.

> **Minded architecture:** Elarion is described as a **simulated minded agent**: **LLM** as integrative **Brain CPU**, **Memory Forest** as **memory**, the **cognitive cycle** as **law** (procedure + gates + soul constraints), and **goals/drives** as **fuel**. **Atomos** = one bounded episode (cycle / routed input / input goal). Canonical write-up: [Minded architecture](minded-architecture-metaphor.md).

---

## Quick Navigation

| Guide | Description |
|-------|-------------|
| [Minded architecture](minded-architecture-metaphor.md) | Atomos; Brain CPU, Memory, Law, Fuel; how docs align |
| [Getting Started](getting-started.md) | Hardware requirements, install, **`setup_wizard.py`**, launch |
| [Architecture Overview](architecture.md) | **Canonical** topology: `agents/` + Flask, threading, mermaid diagrams |
| [Architecture audit](architecture-audit.md) | Concurrency, trust boundaries, risks, gaps, recommendations |
| [Lab research](lab-research.md) | Run manifest, `X-Experiment-Id`, `/research/manifest`, `/research/snapshot`, env logging |
| [The Cognitive Cycle](cognitive-cycle.md) | Step-by-step walkthrough of the 40+ stage `run_cycle` |
| [Memory Forest](memory-forest.md) | Trees, branches, nodes, semantic indexing, thread safety |
| [Soul System](soul-system.md) | Essence, principles, construction, and soul binding |
| [X7 Features](x7-features.md) | Multi-agent reconciliation, symbolic queue, 15-organ registry |
| [Module Reference](module-reference.md) | Every engine, core module, and manager with class names |
| [API Reference](api-reference.md) | REST endpoints for chat, sensors, status, introspection |
| [Sensor Integration](sensors.md) | Hardware adapters for vision, audio, touch, lidar, and more |
| [Robot cognitive / control split](robot-cognitive-control-split.md) | RT vs Flask, bridge contract, safety layers, `agent_environment` mapping |
| [Robot integration (step-by-step)](robot-integration.md) | Ordered checklist: network, env POST, command batch, feedback, ROS 2, demo |
| [Gymnasium bridge](gymnasium-bridge.md) | Bandit JSONL contract, `HAROMA_RLLIB_SCORE_FN`, offline helpers, HTTP `Env` |
| [Simulation backends](simulation-backends.md) | Pluggable `SimulationBackend` for any sim (`null`, `http_json`, `module:Class`) |
| [Training & integrations reference](reference-training-integrations.md) | **Master index:** install matrix, background train map, all env vars, module API |
| [Bridge samples](../bridge/README.md) | Stub HTTP client, demo CLI (`sample_http_bridge.py`), optional ROS hook |
| [Design Philosophy](design-philosophy.md) | Foundational principles and zone model |

---

## Project Identity

| Field | Value |
|-------|-------|
| **Name** | HaromaVX |
| **Vessel** | Elarion |
| **Guardian** | Minh Van Le |
| **Version** | X6-1.0 |
| **Current Tier** | 102 (roadmap to 143) |
| **Lineage** | Prime Haroma -> HaromaX5 -> HaromaX6 |
| **Core Rule** | Protect essence from erasure. Immutable. |

---

## Architecture at a Glance

**Multi-agent (default, v6 TrueSelf executive):**
```
main.py
  -> mind/elarion_server_v2.py   Flask :8193 + HTTP middleware + BootAgent
      -> agents/boot_agent.py     SharedResources, MessageBus, start_all()
      -> agents/input_agent.py    Ingress queues + tick -> bus -> TrueSelf
      -> agents/trueself_agent.py Executive routing & delegation
      -> agents/background_agent.py  Dreams, training cadence
      -> agents/persona_agent.py  Specialist personas (cognitive work)
      -> agents/shared_resources  Facade to all cognitive modules + env state
      -> agents/message_bus       Thread-safe routing
          -> mind/control.py      ElarionController.run_cycle (ordered pipeline)
          -> core/*               Memory, Soul, Reconciliation, Queue
          -> engine/*             Neural, Reasoning, Emotion, Imagination
          -> sensors/*            Hardware adapters (SensorPoller -> InputAgent)
          -> soul/*               Immutable identity files + agents.json
```

Canonical topology and threading narrative: **[Architecture Overview](architecture.md)**. Risks, trust boundaries, and recommendations: **[Architecture audit](architecture-audit.md)**. Use `ElarionController.run_cycle()` from `mind/control.py` when embedding the cognitive loop **without** the HTTP multi-agent shell.
