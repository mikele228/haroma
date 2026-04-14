# Lab research — reproducibility and experiment IDs

[<- Back to Index](index.md)

HaromaX6 includes lightweight hooks for **lab workflows**: a **run manifest** at boot (git, soul file hash, `HAROMA_*` / `ELARION_*` env snapshot) and **experiment id** threading from HTTP into `EpisodeContext`.

---

## Run manifest

At the end of `SharedResources.initialize()`, the server builds a **run manifest** and stores it on `shared.run_manifest`, with id **`shared.lab_run_id`**.

- **`GET /research/manifest`** — returns the same JSON (requires server booted).
- **`HAROMA_LAB_RUN_ID`** — optional fixed id for the process (default: random hex).
- **`HAROMA_LAB_RUN_DIR`** — if set, writes `run_manifest.json` under that directory at boot.
- **`HAROMA_LAB_SEED`** — recorded in the manifest only; you must still seed your own RNGs / frameworks if you need determinism.
- **`HAROMA_LAB_LOG`** — when enabled (`1` / `true` / `yes` / `on`), prints one **stdout** line per experiment-tagged request on the lab routes (path + `experiment_id`).

The manifest is **best-effort** (git may be unavailable; hashes may be null).

---

## Request pipeline

For each HTTP request: **request id** (header `X-Haroma-Request-Id` on the response) → **bearer guard** (if configured) → **rate limit** (if `HAROMA_HTTP_RATE_LIMIT_PER_MIN` is positive) → **lab experiment hook** (parses JSON once for `POST /chat`, `/agent/environment`, `/robot/bridge/feedback`, sets `g.lab_experiment_id`, records the ring event; optional **`HAROMA_LAB_LOG`** line) → route handler → **`after_request`** (structured **access** log to stderr if `HAROMA_STRUCTURED_LOG` is on). Handlers read `g.lab_experiment_id` instead of duplicating header/body parsing.

See [API Reference — rate limiting & logs](api-reference.md#rate-limiting-structured-logs-and-lab-console-logging) for env details.

---

## Research snapshot bundle

- **`GET /research/snapshot`** — single JSON object with **`lab_run_id`**, **`run_manifest`**, **`lab_experiment_events`** (recent ring rows), full **`agent_environment`** status, and **`embodiment_readiness`** (same heuristic as top-level `/status`). Useful for exporting a lab run bundle without merging multiple GETs.

---

## Experiment ID

Clients may send:

- Header **`X-Experiment-Id: <string>`**, and/or  
- JSON field **`"experiment_id": "<string>"`** on `POST /chat`, `POST /agent/environment`, and `POST /robot/bridge/feedback`.

Effects:

- **Chat turns:** id is forwarded through `InputAgent` into the persona cycle; **`EpisodeContext.experiment_id`** and **`EpisodeContext.lab_run_id`** are set for training exports / `to_payload()`.
- **All three routes:** a short entry is appended to a **ring buffer** exposed as **`lab_experiment_events`** inside **`GET /status` → `agent_environment`** (and nested status from env POST responses).

Successful JSON responses from those routes also **echo** **`experiment_id`** and **`lab_run_id`** when set, so scripts can correlate without parsing `/status` first.

**Async chat (`"async": true`):** metadata is stored with the `request_id`. **`GET /chat/result?id=...`** (complete body), **`200` pending** polls, and **SSE** `/chat/wait` completion payloads include the same **`experiment_id`** / **`lab_run_id`** even though the original HTTP request has ended.

---

## Status fields

Top-level **`GET /status`** payload includes **`lab_run_id`** (same id as in the run manifest) for quick scripting, plus **`health`** (e.g. `llm_ready`, last environment receive time) and **`embodiment_readiness`** (operator heuristic from `mind.robot_readiness` — not a safety certification).

Under **`agent_environment`** in `/status`:

- **`lab_run_id`** — current process run id (duplicate for nested convenience).
- **`lab_experiment_events`** — recent `{t, path, experiment_id}` rows.

---

## Related

- [Architecture audit](architecture-audit.md) — lab-oriented recommendations
- [`mind/lab_research.py`](../mind/lab_research.py) — implementation
