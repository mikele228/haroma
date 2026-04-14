# Universal simulation backends

[<- Back to Index](index.md) | [Full training & integration reference](reference-training-integrations.md)

Haroma does not embed Unity, Isaac Sim, MuJoCo, or Gymnasium as mandatory dependencies. Instead, **`integrations.sim`** defines a small **protocol** and **registry** so *any* simulator can plug in:

- **Built-in** `null` (no I/O) and `http_json` (REST adapter).
- **Importable** `HAROMA_SIM_BACKEND=your.module:YourClass` for vendor SDKs (visual sims, ROS bridges, Gymnasium wrappers).

## Protocol

Implement `integrations.sim.protocol.SimulationBackend`:

| Method | Role |
|--------|------|
| `backend_id()` | Short string for logs |
| `capabilities()` | Optional modalities / schema hints |
| `reset(seed=..., **kwargs)` | New episode |
| `step(action: dict)` | Apply action, return observation bundle |
| `observe(**kwargs)` | Passive sense (optional) |
| `close()` | Disconnect |

Normalize host-specific payloads yourself; use [`merge_simulation_into_extensions`](../integrations/sim/universal.py) to attach the latest bundle under `extensions.simulation` for `agent_environment` flows.

## Registry (built-in names)

| Name | Factory behavior |
|------|-------------------|
| `null`, `none` | [`NullSimulationBackend`](../integrations/sim/backends/null_backend.py) — no network I/O. |
| `http_json`, `http` | [`HttpJsonSimulationBackend.from_env()`](../integrations/sim/backends/http_json_backend.py) — reads `HAROMA_SIM_HTTP_*` env vars. |

**Python API** ([`integrations/sim/registry.py`](../integrations/sim/registry.py)):

- `register_backend(name: str, factory: Callable[..., SimulationBackend])` — register a lowercase name.
- `create_backend(name: str, **kwargs)` — instantiate a registered backend.
- `load_backend_from_env()` — reads `HAROMA_SIM_BACKEND`; if it contains `:`, treats as `module:Class` and uses `HAROMA_SIM_BACKEND_KWARGS` JSON.

**Importable backend:** class must implement the protocol; `load_backend_from_env` constructs `Class(**kwargs)` from JSON.

## HTTP JSON backend — request shapes

Base URL: `HAROMA_SIM_HTTP_BASE_URL` (no trailing slash required; client strips).

| Call | Method | Path (default) | Body / query |
|------|--------|----------------|--------------|
| `reset(seed=…)` | POST | `/sim/reset` | JSON: merged `kwargs` + optional `seed`. |
| `step(action)` | POST | `/sim/step` | JSON: `{"action": { ... }}` (your action dict). |
| `observe(**kwargs)` | GET | `/sim/observe` | Query string from stringable kwargs. |

Responses should be JSON objects; the client attaches `http_status` when the payload is a dict. Empty `HAROMA_SIM_HTTP_RESET_PATH` or `HAROMA_SIM_HTTP_OBSERVE_PATH` skips that HTTP call and returns a minimal stub dict.

## Environment variables

| Variable | Meaning |
|----------|---------|
| `HAROMA_SIM_BACKEND` | `null` (default), `http_json`, or `package.module:ClassName` |
| `HAROMA_SIM_BACKEND_KWARGS` | JSON object passed to importable class `__init__` |
| `HAROMA_SIM_HTTP_BASE_URL` | Required for `http_json` (e.g. `http://127.0.0.1:9000`) |
| `HAROMA_SIM_HTTP_STEP_PATH` | Default `/sim/step` — POST body `{"action": {...}}` |
| `HAROMA_SIM_HTTP_RESET_PATH` | Default `/sim/reset` — POST body may include `seed` |
| `HAROMA_SIM_HTTP_OBSERVE_PATH` | Default `/sim/observe` — GET (optional query params) |
| `HAROMA_SIM_HTTP_TIMEOUT_SEC` | HTTP timeout (default `30`) |

Set `HAROMA_SIM_HTTP_RESET_PATH=` or `HAROMA_SIM_HTTP_OBSERVE_PATH=` empty to skip those HTTP calls.

## HTTP JSON contract (reference)

Your sim service can expose the three routes above and translate internally to Unity / Isaac / etc. Haroma stays a **client**; the sim host owns physics and rendering.

## Example (importable)

```python
# my_unity_bridge.py
class UnityHttpBackend:
    def __init__(self, port: int = 9000):
        ...
    def backend_id(self):
        return "unity"
    # ... implement protocol ...
```

```bash
set HAROMA_SIM_BACKEND=my_unity_bridge:UnityHttpBackend
set HAROMA_SIM_BACKEND_KWARGS={"port": 9000}
```

## Related

- [`environment/TextEnvironment.py`](../environment/TextEnvironment.py) — built-in **text** world (separate from this protocol; can wrap behind a `SimulationBackend` if desired).
- [`integrations/robot_http_bridge.py`](../integrations/robot_http_bridge.py) — robot feedback merging (complementary to sim observations).
- [`docs/gymnasium-bridge.md`](gymnasium-bridge.md) — RL / bandit JSONL (different from full sim loop).
