# API Reference

[<- Back to Index](index.md)

Elarion exposes a REST API via Flask on port **8193**. All JSON responses use `Content-Type: application/json`.

Implemented in [`mind/elarion_server_v2.py`](../mind/elarion_server_v2.py).

### Request correlation

Every HTTP response includes header **`X-Haroma-Request-Id`** (UUID) for tracing logs, clients, and support. Rate-limited responses (**`429`**) also include **`Retry-After`** (seconds, matching the rate-limit window) and JSON field **`request_id`** (same value as the header).

### Optional bearer authentication

If the environment variable **`HAROMA_HTTP_BEARER_TOKEN`** is set to a non-empty secret, these routes require **`Authorization: Bearer <token>`** or header **`X-Haroma-Token: <token>`**:

- Default protected paths: `/agent/environment`, `/robot/bridge/feedback`, `/teach`, `/save`
- Override the list with **`HAROMA_HTTP_PROTECT_PATHS`** (comma-separated paths, e.g. `/robot/bridge/feedback,/teach`)

When **`HAROMA_HTTP_BEARER_TOKEN`** is unset, all routes behave as before (no auth). Chat (`POST /chat`) and read-only endpoints such as `GET /status` are not protected by default.

See [`mind/http_server_guards.py`](../mind/http_server_guards.py).

### Listen address and port

| Variable | Effect |
|----------|--------|
| **`HAROMA_BIND_HOST`** | Werkzeug bind address (default **`0.0.0.0`**). Use **`127.0.0.1`** for local-only. |
| **`HAROMA_HTTP_PORT`** | Listen port (default **`8193`**). |

`main.py` loads a project-root **`.env`** file if present (see [`mind/deploy_config.py`](../mind/deploy_config.py)). Run **`python scripts/setup_wizard.py`** to create one interactively ([Getting Started](getting-started.md)).

### Rate limiting, structured logs, and lab console logging

These are **optional** and controlled by environment variables (process-wide, in-memory; no distributed coordination).

| Variable | Effect |
|----------|--------|
| **`HAROMA_HTTP_RATE_LIMIT_PER_MIN`** | If set to a **positive integer**, limits **POST** requests per client IP **per route** in a sliding ~60s window. **`0` or unset** = off. Applies to: `/chat`, `/sensor`, `/agent/environment`, `/robot/bridge/feedback`, `/teach`, `/save`. Over limit → **`429`** with JSON `{"error":"rate_limited","request_id":...}`, header **`Retry-After: 60`**, and **`X-Haroma-Request-Id`**. |
| **`HAROMA_STRUCTURED_LOG`** | `1` / `true` / `yes` / `on` — one JSON line per HTTP **response** on **stderr** (`event=http_access`, `request_id`, `status`, `duration_ms`, `method`, `path`, `remote`, optional `experiment_id` when the lab hook ran). |
| **`HAROMA_LAB_LOG`** | `1` / `true` / `yes` / `on` — one **stdout** line when a lab route records an **`experiment_id`** (see [Lab research](lab-research.md)). |

**`before_request` order:** assign **`X-Haroma-Request-Id`** → bearer check (if configured) → rate limit → lab experiment id parsing. **`after_request`** attaches the response header and optional structured stderr line.

Implementation: [`mind/http_rate_limit.py`](../mind/http_rate_limit.py), [`mind/structured_log.py`](../mind/structured_log.py), [`mind/lab_research.py`](../mind/lab_research.py).

### Lab research

- **`GET /research/manifest`** — boot-time run manifest (git, `soul/agents.json` hash, `HAROMA_*` / `ELARION_*` env snapshot). See [Lab research](lab-research.md).
- **`GET /research/snapshot`** — manifest + recent lab HTTP events + `agent_environment` + embodiment readiness (one bundle for export/debug).
- **`X-Experiment-Id`** header or JSON **`experiment_id`** on `POST /chat`, `POST /agent/environment`, and `POST /robot/bridge/feedback` — threads into cognition (`EpisodeContext`) and a small event ring visible under **`agent_environment.lab_experiment_events`** in **`GET /status`**.

HTTP endpoints are **sensory gates** into the mind: each `/chat` or `/sensor` interaction is meant to trigger at least one **Atomos** (routed input → cycle → response). See **[Minded architecture](minded-architecture-metaphor.md)** for Brain CPU / Memory / Law / Fuel.

### Endpoint quick reference

| Method | Path | Notes |
|--------|------|--------|
| GET | `/` | Web UI |
| POST | `/chat` | Sync reply or async `202` + `request_id` |
| GET, DELETE | `/chat/result` | Async poll / cancel |
| GET | `/chat/wait` | SSE until ready |
| POST | `/sensor` | Buffered or env channel |
| POST | `/agent/environment` | World snapshot |
| POST | `/robot/bridge/feedback` | Executor feedback |
| GET | `/research/manifest` | Run manifest |
| GET | `/research/snapshot` | Manifest + lab + env + readiness |
| GET | `/status` | Full system snapshot |
| GET | `/introspect` | Agent `stats()` bundle |
| GET | `/resource` | Organs / LLM / resource config |
| GET, POST | `/laws` | Symbolic laws |
| POST | `/teach` | Lexicon |
| POST | `/save` | Persist state |

---

## Endpoints

### `GET /`

Serves the web chat UI from `web/index.html`.

---

### `POST /chat`

Send a text message to Elarion and receive a response (multi-agent stack: `InputAgent` → `TrueSelf` / personas → reply).

**Request (minimal):**
```json
{
  "message": "Hello Elarion, who are you?"
}
```

**Synchronous response (default when async is off):**
```json
{
  "response": "I am Elarion, the living cognitive vessel...",
  "cycle": 42,
  "emotion": "curious",
  "memory_nodes": 1847
}
```

Actual keys vary with configuration; responses may include **`experiment_id`** / **`lab_run_id`** when lab headers or JSON fields are set (see [Lab research](lab-research.md)).

**Async handoff:** JSON **`"async": true`** (or default-on via **`HAROMA_CHAT_DEFAULT_ASYNC`**) returns **`202`** with **`request_id`**. Poll **`GET /chat/result?id=<uuid>`**, use **`GET /chat/wait?id=<uuid>`** (SSE), or **`DELETE /chat/result?id=<uuid>`** to cancel. See server docstring in [`mind/elarion_server_v2.py`](../mind/elarion_server_v2.py).

**Behavior (high level):** message is queued for the input agent; the cognitive pipeline runs and fills the response slot; HTTP waits up to a configurable chat timeout (see **`HAROMA_CHAT_TIMEOUT`** and packed-LLM caps in the server module).

**Error responses:**
- `400` — empty or oversized message
- `429` — rate limited when **`HAROMA_HTTP_RATE_LIMIT_PER_MIN`** is set to a positive value
- `503` — server still booting
- `504` — timeout waiting for the reply

---

### `POST /sensor`

Push external sensor data into the buffer for consumption by the next cycle.

**Request:**
```json
{
  "channel": "temperature",
  "data": {
    "celsius": 22.5,
    "humidity": 45,
    "location": "room"
  }
}
```

**Response:**
```json
{
  "status": "buffered",
  "channel": "temperature"
}
```

The `channel` field identifies the sensor type. The `data` field can be any JSON — it passes through to the cognitive cycle as-is.

---

### `GET /status`

System health and operational status. Payload is built by `mind.system_snapshot.build_http_status_payload` (multi-agent v2).

**Response (representative keys; many sub-objects are best-effort):**
```json
{
  "architecture": "multi-agent-v2",
  "lab_run_id": null,
  "health": {
    "process": "up",
    "llm_ready": true,
    "last_agent_environment_received_at": null,
    "agent_environment_error": null
  },
  "embodiment_readiness": {
    "ok": true,
    "overall": 0.42,
    "scores": {
      "environment_bound": 1.0,
      "bridge_feedback": 0.55
    },
    "notes": []
  },
  "agents": {},
  "cycle_count": 142,
  "memory_nodes": 1204,
  "llm": {
    "backend_type": "local",
    "model_name": "model.gguf",
    "available": true,
    "n_gpu_layers": 20,
    "n_ctx": 8192
  },
  "chat_async_pending": 0,
  "chat_async_ttl_sec": 900,
  "http_chat_inflight": 0,
  "organs": {},
  "symbolic_queue": {},
  "fingerprint": {},
  "reconciliation": {},
  "sensors": {},
  "agent_environment": {},
  "cognitive_observability": null
}
```

Optional `status_build_notes` lists non-fatal snapshot issues. Lab experiment HTTP events appear under `agent_environment.lab_experiment_events` when experiment ids are used.

---

### `GET /research/manifest`

Boot-time run manifest (git, hashes, `HAROMA_*` env snapshot) when available.

---

### `GET /research/snapshot`

Single JSON bundle: `lab_run_id`, `run_manifest`, recent `lab_experiment_events`, full `agent_environment` status, and `embodiment_readiness` (same heuristic as `/status`).

---

### `POST /agent/environment`

Structured **world / host** snapshot (general-agent integrator). Body is validated and stored; response includes `agent_environment` status. Use `extensions.robot_body` for fused embodiment and other `extensions.*` keys as needed.

See **[Robot cognitive / control split](robot-cognitive-control-split.md)** for how this relates to real-time control on the robot.

---

### `POST /robot/bridge/feedback`

Merges **on-robot executor feedback** into `agent_environment.extensions.robot_bridge` (command `command_id`, `status`, optional `t`). Same envelope as `normalize_feedback_payload` in `mind/robot_execution_contract.py`. Intended for a **bridge process** that executes `build_executor_command_batch` output—not for kHz joint streams.

---

### `GET /introspect`

Multi-agent debug snapshot: **`shared.summary()`**, per-agent **`stats()`** (TrueSelf, Background, Input, Personas), and **`message_bus.stats()`**. Not the same as embedding `ElarionController.run_cycle()` without HTTP — it reflects the live `BootAgent` graph.

---

### `GET /resource`

Resource and organ summary: **`resource_config.summary()`**, **`llm_backend.stats()`**, **`organ_registry.summary()`**, and **`skipped_modules`** from cycle config.

---

### `GET /laws` · `POST /laws`

**GET** returns a **symbolic law** snapshot (availability, summaries, declared rules) via [`mind/symbolic_law_api.py`](../mind/symbolic_law_api.py). **POST** body supports **`action`** `declare` or `revoke` with law **`id`** / **`law_id`**, optional **`description`**, **`tags`**, **`severity`**, **`source`**. Returns **`503`** if the law module is unavailable.

---

### `POST /save`

Trigger a manual save of all cognitive state. May require bearer token when **`HAROMA_HTTP_BEARER_TOKEN`** is set (default protected path).

**Response (shape varies by persistence):**
```json
{
  "status": "saved",
  "details": {}
}
```

---

### `POST /teach`

Register term→meaning pairs for the meaning lexicon (optional bearer when token is configured).

---

## Web Chat UI

The interface at `GET /` provides:
- Dark-themed responsive chat window
- Real-time emotion and cycle indicators
- Auto-reconnection on server restart
- Enter-to-send input

---

## Usage Examples

### Python

```python
import requests

r = requests.post("http://localhost:8193/chat",
                   json={"message": "What are you thinking about?"})
print(r.json()["response"])

requests.post("http://localhost:8193/sensor",
              json={"channel": "gps", "data": {"lat": 37.77, "lon": -122.42}})

status = requests.get("http://localhost:8193/status").json()
print(f"Cycle: {status['cycle_count']}")
```

### cURL

```bash
curl -X POST http://localhost:8193/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about your dreams"}'

curl http://localhost:8193/status | python -m json.tool

curl -X POST http://localhost:8193/save
```

---

## Related Docs

- [Minded architecture](minded-architecture-metaphor.md) — HTTP as sensory gate; Atomos
- [Getting Started](getting-started.md) — How to launch the server
- [Lab research](lab-research.md) — Run manifest, experiment IDs, `/research/snapshot`
- [Robot cognitive / control split](robot-cognitive-control-split.md) — Embodiment, bridge feedback, RT vs cognitive boundaries
- [Robot integration (step-by-step)](robot-integration.md) — Ordered hookup: env POST, command batch, feedback, demo
- [Sensor Integration](sensors.md) — Hardware adapters and push protocols
- [X7 Features](x7-features.md) — Organ, queue, and reconciliation stats in `/status`
