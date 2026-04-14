# Architecture audit (HaromaX6)

[<- Back to Index](index.md)

**Scope:** Structural review of the default deployment path (`main.py` → `mind/elarion_server_v2.py` → `BootAgent` → agents + `SharedResources` → `ElarionController.run_cycle`), HTTP surface, concurrency, and integration boundaries.  
**Not in scope:** Line-by-line security review, full threat model, or performance profiling.

---

## Executive summary

HaromaX6 is a **modular cognitive server**: Flask exposes a wide HTTP API; **BootAgent** constructs a large **`SharedResources`** graph with timed initialization and **CognitiveNull** fallbacks; **multiple daemon-style agents** (Input, TrueSelf, Background, Action, Personas) coordinate via a **thread-safe MessageBus**. Cognition is centered on **`ElarionController.run_cycle`** (large ordered pipeline) and optional **packed LLM** paths. **Embodiment** is modeled as **asynchronous environment JSON** (`agent_environment`, `robot_body`, `robot_bridge`) with an explicit **non–real-time** contract documented in [Robot cognitive / control split](robot-cognitive-control-split.md).

**Overall:** Strong **separation of concerns in documentation and contracts** (bridge, env validation, packed context); **operational complexity** is high due to **threaded Flask + many background tick loops + shared mutable state**; **default network exposure** assumes a **trusted network** unless additional controls are added.

---

## System shape (verified)

| Layer | Responsibility |
|-------|----------------|
| **HTTP** | `elarion_server_v2`: `/chat`, `/sensor`, `/agent/environment`, `/robot/bridge/feedback`, `/status`, async chat, etc. Werkzeug `run_simple(..., threaded=True)`. |
| **Boot** | `BootAgent.boot()` → `SharedResources.initialize()` → `MessageBus` → spawn agents → `start_all()`; `SensorPoller` with hardware adapters. |
| **Agents** | Input (buffer + tick), TrueSelf (executive), Background, Action (board), Personas; routing strategies on `MessageBus`. |
| **Cognition** | `ElarionController.run_cycle` drives perception → memory → gates → engines → action/outcome; step list in `mind.cycle_flow`. |
| **State** | `SharedResources` holds engines, memory, locks (`ConcurrencyCoordinator`), `agent_environment`, robot body merge state, metrics. |

---

## Strengths

1. **Explicit integration contracts** — `mind/robot_execution_contract.py`, `integrations/robot_http_bridge.py`, `mind/environment_context.py`, and `mind/packed_llm_context.py` give **stable JSON shapes** for hosts and bridges without entangling ROS inside the server.
2. **Graceful degradation at boot** — `SharedResources.initialize()` uses **per-module timeouts** and **CognitiveNull** substitution so a bad optional module does not always abort the whole boot.
3. **Observability hooks** — `mind/system_snapshot.build_http_status_payload`, `cognitive_metrics`, `agent_environment_status` (including **robot_bridge** summaries and **robot_bridge_metrics** counters), `status_build_notes` for partial failures.
4. **Multi-agent decomposition** — Clear roles (Input vs TrueSelf vs Background vs Persona) and **MessageBus** abstractions reduce ad-hoc coupling versus a single monolithic loop (though `SharedResources` remains a wide shared hub).
5. **Documentation** — Indexed guides ([architecture](architecture.md), [cognitive cycle](cognitive-cycle.md), [robot split](robot-cognitive-control-split.md), [bridge samples](../bridge/README.md)) align with the code paths for embodiment.

---

## Risks and limitations

### Concurrency and coupling

- **Threaded Flask** serves concurrent requests while agents run in **other threads**; correctness relies on **locks** on `SharedResources` and bus mailboxes. This is standard but **easy to get wrong** when adding new shared mutable fields without clear locking discipline.
- **`SharedResources` is a large façade** — many subsystems are reachable from one object, which **invites implicit dependencies** between features that should stay orthogonal.

### Cognitive pipeline complexity

- **`run_cycle` is long and stateful** — high cognitive load for maintainers; regressions may appear as subtle ordering or gating issues (`ProcessGate`, skipped steps). **Step tracing** env vars exist but operational debugging still depends on team familiarity.

### Network and trust

- **Default binding** is `0.0.0.0:8193` in `launch()` — suitable for LAN dev; **not** Internet-hardened without TLS, auth, and optional **rate limits** on selected POST routes (see [API Reference](api-reference.md)).
- **Authentication** is **optional** (`HAROMA_HTTP_BEARER_TOKEN` on configured paths). Chat and read-only routes stay open by default — treat unauthenticated surfaces as **sensory injection** into cognition.

### Platform and lifecycle

- **Port 8193 cleanup** uses **Windows-specific** `netstat`/`taskkill` in `launch()` — non-portable; Linux deployments need equivalent or disable.
- **Global `boot_agent`** in `elarion_server_v2` — simplifies handlers but complicates **multi-instance** or **tests** that need isolation.

### Real-time and safety

- **Not a motor controller** — already documented; repeating for audit: **functional safety** for actuators must remain **outside** Haroma (hardware ESTOP, vendor/ROS control, independent watchdogs).

### Documentation drift

- **Resolved for `/status`:** [API Reference](api-reference.md) examples target **`mind.system_snapshot.build_http_status_payload`**. If anything still disagrees, treat **live `GET /status`** or the Python builder as authoritative.

---

## Implemented upgrades

| Item | Details |
|------|---------|
| **Optional bearer auth** | ``HAROMA_HTTP_BEARER_TOKEN`` + ``HAROMA_HTTP_PROTECT_PATHS`` — :mod:`mind.http_server_guards` runs on selected routes (default: ``/agent/environment``, ``/robot/bridge/feedback``, ``/teach``, ``/save``). |
| **HTTP hardening & observability** | In-process **rate limiting** (`HAROMA_HTTP_RATE_LIMIT_PER_MIN`), **`429`** + **`Retry-After`**, **`X-Haroma-Request-Id`** on all responses, optional **stderr JSON access logs** (`HAROMA_STRUCTURED_LOG`, `event=http_access`). |
| **Lab / research** | Run manifest, experiment id threading, `/research/snapshot`, optional **`HAROMA_LAB_LOG`**. |

---

## Gaps (optional improvements)

| Priority | Item |
|----------|------|
| High | **TLS + reverse proxy** when exposing beyond localhost (bearer alone is not enough on untrusted networks). |
| High | **Integration tests** for HTTP bridge and env POSTs (Flask test client) to lock contract behavior. |
| Medium | **Narrower interfaces** — facades or read-only views of `SharedResources` per agent to reduce accidental coupling. |
| Medium | **Optional JSON logs** for bridge/env POST **bodies** (redact secrets) keyed by **`X-Haroma-Request-Id`** — correlation exists; per-route payload logging still optional. |
| Low | **Replace globals** in server module with an app factory / context object for testability. |
| Low | **Cross-platform** port binding and stale-process handling for non-Windows (Windows-only `taskkill` path in `launch()`; POSIX is a no-op). |

---

## Recommendations (ordered)

1. **Treat the server as a trusted-zone component** — firewall, VPN, or reverse proxy with TLS; use **`HAROMA_HTTP_BEARER_TOKEN`** for mutating routes when not on localhost-only.
2. **Keep motor torque and safety** on the robot stack; use Haroma only for **supervisory commands and fused state** per [robot-cognitive-control-split.md](robot-cognitive-control-split.md).
3. **When extending `SharedResources`**, document **which lock** protects new fields and add **tests** that exercise concurrent read/write paths if relevant.
4. **Keep** `docs/api-reference.md` aligned with `build_http_status_payload` when adding fields (CI or manual check on release).
5. **Run CI** with `pytest -m "not integration"` for fast feedback; reserve integration markers for real LLM or long-running paths.

---

## Related documents

- [Lab research](lab-research.md) — run manifest, experiment IDs
- [Architecture Overview](architecture.md)
- [API Reference](api-reference.md) — bearer auth section
- [The Cognitive Cycle](cognitive-cycle.md)
- [Robot cognitive / control split](robot-cognitive-control-split.md)
- [Bridge samples](../bridge/README.md)
- [Minded architecture](minded-architecture-metaphor.md)
- [`mind/http_server_guards.py`](../mind/http_server_guards.py) — optional bearer implementation

---

*This audit is descriptive, not a certification of safety or security.*
