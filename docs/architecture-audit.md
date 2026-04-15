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
- **One cognitive stack per process** — HTTP runtime state (`BootAgent`, sensor poller, async chat registry) is stored on the Flask app via :func:`mind.server_state.get_haroma_server_state` (``HAROMA_FLASK_EXTENSION_KEY``; see [`mind/server_state.py`](../mind/server_state.py), [`mind/elarion_server_v2.py`](../mind/elarion_server_v2.py)). There is still **one** booted graph per Werkzeug worker; tests should use ``get_haroma_server_state(app)`` or replace that state object rather than relying on removed module-level globals.

### Real-time and safety

- **Not a motor controller** — already documented; repeating for audit: **functional safety** for actuators must remain **outside** Haroma (hardware ESTOP, vendor/ROS control, independent watchdogs).

### Documentation drift

- **Resolved for `/status`:** [API Reference](api-reference.md) examples target **`mind.system_snapshot.build_http_status_payload`**. If anything still disagrees, treat **live `GET /status`** or the Python builder as authoritative.

---

## Implemented upgrades

| Item | Details |
|------|---------|
| **Optional bearer auth** | ``HAROMA_HTTP_BEARER_TOKEN`` + ``HAROMA_HTTP_PROTECT_PATHS`` — :mod:`mind.http_server_guards` runs on selected routes (default: ``/agent/environment``, ``/robot/bridge/feedback``, ``/teach``, ``/save``). |
| **Production checklist** | Ordered bind/TLS/auth/rate-limit/ops steps: [Production hardening](production-hardening.md). |
| **Client IP behind proxy** | [`mind/client_ip.py`](../mind/client_ip.py): optional `X-Forwarded-For` when the direct peer is trusted — aligns rate limits and structured logs with real clients. |
| **GET poll abuse** | Optional `HAROMA_HTTP_GET_RATE_LIMIT_PER_MIN` on `/chat/result`; save summary includes `persist_status` / `persist_error_keys` when components fail. |
| **Flask app state** | [`mind/server_state.py`](../mind/server_state.py): ``HAROMA_FLASK_EXTENSION_KEY`` + :func:`get_haroma_server_state` — `BootAgent`, sensor poller, async chat registry on the Flask app; :func:`mind.elarion_server_v2._haroma` uses :data:`flask.current_app` when an app context is active (else the module ``app``), so tests and future app factories see the right state. |
| **Idempotent boot** | :func:`mind.elarion_server_v2._init` holds a lock and skips if ``boot_agent`` is already set — avoids duplicate graphs if ``launch()`` were invoked more than once. |
| **HTTP readiness** | :func:`mind.elarion_server_v2._require_boot_with_input` / ``_require_boot_with_shared`` centralize 503 payloads so route handlers do not duplicate readiness checks. |
| **HTTP hardening & observability** | In-process **rate limiting** (`HAROMA_HTTP_RATE_LIMIT_PER_MIN`), **`429`** + **`Retry-After`**, **`X-Haroma-Request-Id`** on all responses, optional **stderr JSON access logs** (`HAROMA_STRUCTURED_LOG`, `event=http_access`). |
| **Lab / research** | Run manifest, experiment id threading, `/research/snapshot`, optional **`HAROMA_LAB_LOG`**. |
| **HTTP bridge/env tests** | [`tests/test_http_bridge_env_contract.py`](../tests/test_http_bridge_env_contract.py) — Flask test client for ``POST /agent/environment`` and ``POST /robot/bridge/feedback`` with mocked ``shared`` (no full boot). Async chat HTTP coverage in [`tests/test_elarion_chat_async_http.py`](../tests/test_elarion_chat_async_http.py). |
| **POSIX stale port** | Non-Windows: :func:`mind.elarion_server_v2._kill_port_posix` uses ``lsof`` (if installed) to SIGTERM other listeners on the configured HTTP port before bind — mirrors Windows ``taskkill`` behavior for local dev. |
| **Pytest import hygiene** | [`tests/_import_guard.py`](../tests/_import_guard.py): ``prepare_test_imports`` stubs optional ``sentence_transformers`` before heavy Haroma imports; ``torch_loads_in_subprocess`` / ``skip_unless_torch_imports`` use a **subprocess** ``import torch`` probe (cached) so broken PyTorch DLLs on Windows do not abort collection. ``HAROMA_SKIP_TORCH_TESTS=1`` skips torch-dependent tests without probing. Torch-heavy tests are marked ``@pytest.mark.torch`` (``pytest -m "not torch"`` deselects them). |

---

## Gaps (optional improvements)

| Priority | Item |
|----------|------|
| High | **TLS + reverse proxy** when exposing beyond localhost (bearer alone is not enough on untrusted networks). |
| Medium | **End-to-end HTTP** tests against a live booted server (optional; contract tests above cover route wiring with mocks). |
| Medium | **Narrower interfaces** — facades or read-only views of `SharedResources` per agent to reduce accidental coupling. |
| Medium | **Optional JSON logs** for bridge/env POST **bodies** (redact secrets) keyed by **`X-Haroma-Request-Id`** — correlation exists; per-route payload logging still optional. |
| Low | **Full app factory** (build Flask app + register all routes in one callable) — optional; runtime state is already per-app via :func:`mind.server_state.get_haroma_server_state` and :func:`mind.elarion_server_v2._haroma` resolves :data:`flask.current_app` when a context is active. |
| Low | **Port bind failures** on busy ports without `lsof` (POSIX) or when stale listeners ignore SIGTERM — rare in dev; production uses process managers. |

---

## Recommendations (ordered)

1. **Treat the server as a trusted-zone component** — firewall, VPN, or reverse proxy with TLS; use **`HAROMA_HTTP_BEARER_TOKEN`** for mutating routes when not on localhost-only.
2. **Keep motor torque and safety** on the robot stack; use Haroma only for **supervisory commands and fused state** per [robot-cognitive-control-split.md](robot-cognitive-control-split.md).
3. **When extending `SharedResources`**, document **which lock** protects new fields and add **tests** that exercise concurrent read/write paths if relevant.
4. **Keep** `docs/api-reference.md` aligned with `build_http_status_payload` when adding fields (CI or manual check on release).
5. **Run CI** with `pytest -m "not integration"` for fast feedback; add `and not torch` when the runner has no working PyTorch (see [`tests/_import_guard.py`](../tests/_import_guard.py) and the ``torch`` marker in [`pytest.ini`](../pytest.ini)). Reserve the **integration** marker for real LLM or long-running paths.

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
