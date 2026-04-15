# Production hardening checklist

[<- Back to Index](index.md)

Use this when moving HaromaX6 / Elarion **off a trusted dev machine** or **onto any network you do not fully control**. The server is designed for **modular cognition**, not as an Internet-facing appliance without extra layers.

**Related:** [Architecture audit](architecture-audit.md) (trust boundaries, threading), [API Reference](api-reference.md) (HTTP surface, rate limits, auth).

---

## Threat model (one minute)

- **Sensory injection:** `POST /chat`, `POST /sensor`, and environment routes feed the cognitive loop — treat them like **privileged inputs**.
- **State mutation:** `/teach`, `/save`, `/agent/environment`, `/robot/bridge/feedback` change memory or embodiment-related state.
- **Read leakage:** `/status`, `/introspect`, `/resource`, `/research/*` can expose internals — restrict on untrusted networks.
- **Haroma is not a safety PLC** — robot/motor **ESTOP and real-time control** stay **outside** this process ([Robot cognitive / control split](robot-cognitive-control-split.md)).

---

## Ordered checklist

### 1. Bind address (first)

| Goal | Action |
|------|--------|
| Local-only | Set `HAROMA_BIND_HOST=127.0.0.1` and put a reverse proxy on the same host or use SSH tunnel for remote admins. |
| LAN | Bind to a **private** interface IP if you must listen on the network; avoid `0.0.0.0` on machines with multiple interfaces unless you understand exposure. |

Default in code is `0.0.0.0` ([`mind/deploy_config.py`](../mind/deploy_config.py)) — **explicitly override** for production.

### 2. TLS and edge

| Goal | Action |
|------|--------|
| HTTPS | Terminate TLS at **Caddy**, **nginx**, **Traefik**, or a cloud LB; talk to Haroma over **HTTP on loopback** (`127.0.0.1:8193`). |
| Certificates | Use ACME (Let’s Encrypt) or your org PKI; do not ship secrets in the repo. |

Haroma’s optional bearer auth **does not replace TLS** ([`mind/http_server_guards.py`](../mind/http_server_guards.py)).

### 3. Authentication on sensitive routes

| Goal | Action |
|------|--------|
| Enable | Set a long random `HAROMA_HTTP_BEARER_TOKEN` (store in env / secrets manager, not git). |
| Scope | Set `HAROMA_HTTP_PROTECT_PATHS` to a **comma-separated** list of paths that require the token. |

**Default protected paths** (when token is set and `HAROMA_HTTP_PROTECT_PATHS` is unset):  
`/agent/environment`, `/robot/bridge/feedback`, `/teach`, `/save`

**Stricter example for untrusted or wide networks** (adjust to your clients):

```text
HAROMA_HTTP_PROTECT_PATHS=/chat,/chat/result,/chat/wait,/sensor,/agent/environment,/robot/bridge/feedback,/teach,/save,/status,/introspect,/resource,/research/manifest,/research/snapshot
```

Clients send `Authorization: Bearer <token>` or `X-Haroma-Token: <token>`. If you protect `/chat`, **every** chat client must send the header.

### 4. Rate limiting

| Goal | Action |
|------|--------|
| Abuse mitigation | Set `HAROMA_HTTP_RATE_LIMIT_PER_MIN` to a **positive** integer (max POSTs per client IP per route per minute). `0` = off. |
| Behind a reverse proxy | By default limits key on the **direct TCP peer** (often `127.0.0.1`). Set `HAROMA_HTTP_USE_X_FORWARDED_FOR=1` and optionally `HAROMA_HTTP_TRUSTED_PROXIES` so the **real client** from `X-Forwarded-For` is used when the peer is trusted (see [`mind/client_ip.py`](../mind/client_ip.py)). |
| Async chat polling | Optional **`HAROMA_HTTP_GET_RATE_LIMIT_PER_MIN`** caps **`GET /chat/result`** per IP (does not affect `/status` health checks). |

See [API Reference](api-reference.md) for behavior (`429`, `Retry-After`).

### 5. Observability

| Goal | Action |
|------|--------|
| Request correlation | Every response includes `X-Haroma-Request-Id` — log it in your proxy and app logs. |
| Structured access logs | Set `HAROMA_STRUCTURED_LOG=1` for JSON lines on stderr (integrate with your log shipper). |

### 6. Secrets and data

| Goal | Action |
|------|--------|
| `.env` | Never commit `.env`; use `.env.example` as a template only. |
| Runtime data | `data/` and `models/` should stay **out of git**; encrypt backups if they contain conversations or weights. |

### 7. Process lifecycle

| Goal | Action |
|------|--------|
| Restarts | Run under **systemd**, **Docker**, or **Kubernetes** with `Restart=always` / equivalent. |
| Health | Use `GET /status` for liveness/readiness probes (tune timeouts for LLM cold start). |
| Resources | Set **memory and CPU limits** so one runaway cycle cannot take the whole host. |

### 8. Windows vs Linux notes

| Topic | Note |
|-------|------|
| Port cleanup | `launch()` may use Windows-specific helpers; Linux deployments should rely on **process managers** instead of ad-hoc `taskkill`. |
| Native deps | Optional stacks (e.g. some ML / VW installs) can fail on specific OS builds — **pin** production dependencies and validate in CI or a staging image. |

### 9. Embodiment and robots

| Goal | Action |
|------|--------|
| Auth | Always require bearer token on `/agent/environment` and `/robot/bridge/feedback` in any shared or hostile network. |
| Safety | Keep **hardware ESTOP** and vendor motion stacks **outside** Haroma; do not use Flask latency for safety-critical loops. |

---

## Minimal reverse proxy (Caddy, TLS → local Haroma)

Haroma listens on `127.0.0.1:8193` with `HAROMA_BIND_HOST=127.0.0.1`. Caddy terminates TLS and forwards:

```caddyfile
your.domain.example {
    reverse_proxy 127.0.0.1:8193
}
```

Add **authentication** at the edge (Caddy `basicauth`, OAuth2 plugin, or mTLS) if bearer tokens on Haroma routes are not enough for your org.

---

## Environment template

Copy [`.env.example`](../.env.example) to `.env` and set at least:

- `HAROMA_BIND_HOST`
- `HAROMA_HTTP_BEARER_TOKEN` + `HAROMA_HTTP_PROTECT_PATHS` (production)
- `HAROMA_HTTP_RATE_LIMIT_PER_MIN` (if exposed beyond localhost)

---

## Suggested review cadence

- **Quarterly:** Re-read this checklist after any change to HTTP routes or default bind behavior.
- **After incidents:** Extend `HAROMA_HTTP_PROTECT_PATHS` or proxy rules before expanding features on the same host.
