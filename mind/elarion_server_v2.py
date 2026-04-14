"""
Elarion Server v2 -- Multi-Agent Architecture (v6 TrueSelf).

Cooperating agents form Elarion's mind:
  - BootAgent      : initializes everything, supervises health
  - InputAgent     : collects HTTP/sensor inputs, forwards to TrueSelf
  - TrueSelfAgent  : executive consciousness -- delegate to personas or self
  - BackgroundAgent: dreams, reconciliation, training, persistence
  - PersonaAgent(s): specialist inner voices, activated by TrueSelf delegation

The Flask HTTP layer is unchanged from the user's perspective.

``GET /status`` JSON is built by :func:`mind.system_snapshot.build_http_status_payload`.
Background defer env is centralized in :mod:`mind.bg_training_env`; packed-LLM timeout
for status and HTTP wait margins uses :func:`mind.llm_context_timeout.llm_context_timeout_seconds`
(also re-exported on :mod:`mind.cognitive_contracts`). Non-fatal
snapshot issues append ``status_build_notes``; set ``HAROMA_STATUS_SNAPSHOT_DEBUG=1``
for traceback logging on those paths.

Env ``HAROMA_CHAT_LLM_PRIMARY`` (default ``1``): every conversant user turn
runs the packed-context LLM with soul snapshot, recalled memories, and KG
triples. Set to ``0`` to restore the lighter organic-only path on conversant
cycles.

Env ``HAROMA_LLM_CONTEXT_TIMEOUT_SEC`` (default ``600``): max seconds to wait
for one packed-context ``generate_chat`` (first local GGUF load can be very
slow). Use ``0`` or ``off`` for no limit. If the cap is hit, the cycle
continues with ``source=llm_timeout`` and the user sees a load message, not
a generic ``I don't know``.

Env ``HAROMA_CHAT_TIMEOUT`` (optional): seconds for HTTP ``/chat`` to wait on
the cognitive slot. If unset, the wait is ``max(540, LLM_cap+120)`` (or
``600`` when LLM timeout is unlimited).

Input latency: ``HAROMA_INPUT_TICK_INTERVAL_SEC`` overrides InputAgent tick
(default in ``soul/agents.json`` / merged config is ~0.12s). User HTTP messages
are queued with **priority** over other text. Legacy ``depth=fast`` in JSON is
accepted but treated as full pipeline (same as ``normal``). Set
``HAROMA_BG_DEFER_TRAINING_ON_HTTP_CHAT=0`` to keep background training during
``/chat`` waits (more neural lock contention).

Non-blocking chat: POST ``{"message": "...", "async": true}`` returns ``202`` with
``request_id``; poll ``GET /chat/result?id=<uuid>`` until ``status`` is absent
and the payload matches synchronous ``/chat``. Pending entries expire after
``HAROMA_CHAT_ASYNC_TTL_SEC`` (default ``900``). Single-process only.
``POST /robot/bridge/feedback`` merges on-robot executor results into
``agent_environment.extensions.robot_bridge`` (see :mod:`integrations.robot_http_bridge`).

``GET /status`` includes ``health`` (``process``, ``llm_ready``, ``last_agent_environment_received_at``,
optional ``agent_environment_error``), ``embodiment_readiness`` (heuristic scores from
``mind.robot_readiness``), ``chat_async_pending``, ``chat_async_ttl_sec``,
``http_chat_inflight``, ``bg_training_defer_enabled``, ``bg_training_deferred``,
optional ``bg_training_defer_cap_sec``, ``web_learn`` (crawler snapshot), and
``training_scheduler`` (module intervals / losses), ``runtime_signals``
(depth stack, ``last_background_training_at``, ``last_background_training_had_effect`` from
:class:`agents.runtime_signals.RuntimeSignals`),
``agent_environment`` (with ``robot_bridge`` summary and ``robot_bridge_metrics``
counters for executor feedback POSTs), and optionally ``status_build_notes`` when a
subsection could not be read.

``HAROMA_BG_DEFER_TRAINING_CAP_SEC`` (optional): when set with defer-on-chat,
background training still runs at least once per this many seconds even while
HTTP chat is in flight.

``HAROMA_WEB_LEARN_INJECT_MODE`` (``teaching_only`` | ``factual_heuristic`` | ``always``)
and ``HAROMA_WEB_LEARN_INJECT_MAX`` tune injection of ``web_learn`` memories into recall
(see ``core/chat_recall_policy``).

**HTTP hardening (optional):** set ``HAROMA_HTTP_BEARER_TOKEN`` to require
``Authorization: Bearer <token>`` or ``X-Haroma-Token`` on paths in
``HAROMA_HTTP_PROTECT_PATHS`` (comma-separated; defaults include
``/agent/environment``, ``/robot/bridge/feedback``, ``/teach``, ``/save``).
See :mod:`mind.http_server_guards`.

**Lab research:** ``HAROMA_LAB_RUN_ID``, ``HAROMA_LAB_RUN_DIR``, ``HAROMA_LAB_SEED``;
``GET /research/manifest``, ``GET /research/snapshot`` (manifest + lab events + env +
readiness); ``X-Experiment-Id`` / JSON ``experiment_id`` on chat and env routes. See
:mod:`mind.lab_research`. Optional ``HAROMA_LAB_LOG`` prints one line per tagged request.

**HTTP rate limit (optional):** ``HAROMA_HTTP_RATE_LIMIT_PER_MIN`` — max POSTs per client
IP per route per minute on ``/chat``, ``/sensor``, ``/agent/environment``,
``/robot/bridge/feedback``, ``/teach``, ``/save``; ``0`` (default) disables.

**Structured request logging (optional):** ``HAROMA_STRUCTURED_LOG=1`` emits one JSON line
per HTTP response to stderr (``event=http_access``, ``request_id``, ``status``, ``duration_ms``,
method, path, remote, optional ``experiment_id`` when lab flow ran). Every response includes
header ``X-Haroma-Request-Id`` for correlation.

Set ``HAROMA_CHAT_DEFAULT_ASYNC=1`` to use async handoff when the JSON body
omits ``"async"`` (explicit ``"async": false`` still disables). ``GET /chat/wait``
streams Server-Sent Events until the reply is ready (keepalive via
``HAROMA_CHAT_SSE_KEEPALIVE_SEC``, default ``15``). ``DELETE /chat/result?id=``
cancels a still-pending request.

Env ``HAROMA_LLM_LOG_PACKED_STATS`` / ``HAROMA_LLM_DUMMY_REPLY`` /
``HAROMA_LLM_CHAT_ONLY`` / ``HAROMA_LLM_PROMPT_INFO`` (see
``engine/LLMContextReasoner``): log packed prompt size, skip inference with a
probe reply, user-only chat, or attach ``prompt_info`` to the HTTP payload.

Env ``HAROMA_LLM_WARMUP`` (default ``1``): after a local GGUF loads, run a
short decode during boot so the first chat avoids cold-start latency. Set
``0`` to skip. ``HAROMA_LLM_WARMUP_MAX_TOKENS`` caps decode length (default ``24``).
"""

import math as _math
import os
import sys
import json
import time
import uuid
from typing import Any, Dict, Optional, Tuple
import atexit
import traceback as _tb

from flask import Flask, g, request, send_from_directory, Response, stream_with_context
from werkzeug.serving import run_simple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.boot_agent import BootAgent
from agents.chat_latency import trace_requested
from utils.coerce_bool import json_bool as _json_bool
from mind.cognitive_contracts import (
    llm_context_timeout_seconds,
    normalize_http_chat_response,
)
from mind.symbolic_law_api import handle_laws_request
from mind.chat_async_registry import ChatAsyncRegistry, truthy_async_flag
from mind.config_env import env_float
from mind.http_chat_timeouts import http_chat_wait_sec as _http_chat_wait_sec
from mind.http_server_guards import (
    configured_bearer_secret,
    protected_path_set,
    verify_http_request_bearer,
)
from mind.http_rate_limit import (
    RATE_LIMIT_WINDOW_SEC,
    check_rate_limit,
    rate_limit_per_minute,
)
from mind.lab_research import (
    apply_lab_experiment_to_request,
    lab_events_snapshot,
    merge_lab_context,
    merge_lab_context_values,
)
from mind.robot_readiness import embodiment_readiness_summary
from mind.structured_log import log_event, structured_log_enabled
from mind.system_snapshot import build_http_status_payload
from mind.user_identity import sanitize_user_id
from sensors.adapters import (
    SensorPoller,
    VisionAdapter,
    AudioAdapter,
    TouchAdapter,
    SmellAdapter,
    TasteAdapter,
    LidarAdapter,
    InfraredAdapter,
    ImuAdapter,
    GpsAdapter,
    warmup_neural_models,
)

_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


def _maybe_enable_fast_llm_for_gpu(shared) -> None:
    """If CUDA is available and local GGUF uses GPU layers, enable fast-path LLM."""
    if str(os.environ.get("HAROMA_FAST_LLM", "") or "").strip():
        return
    lb = getattr(shared, "llm_backend", None)
    if lb is None or not getattr(lb, "available", False):
        return
    if getattr(lb, "backend_type", "") != "local":
        return
    ngl = int(getattr(lb, "_n_gpu_layers", 0) or 0)
    if ngl == 0:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
    except Exception:
        return
    os.environ["HAROMA_FAST_LLM"] = "1"
    print(
        "[Elarion-v2] CUDA + local GGUF with GPU offload: set HAROMA_FAST_LLM=1",
        flush=True,
    )


def _env_truthy_flag(name: str) -> bool:
    return str(os.environ.get(name, "") or "").strip().lower() in ("1", "true", "yes", "on")


def _async_chat_resolve_ready(rid: str) -> Tuple[str, Any]:
    """Resolve one async chat slot. Caller must have verified ``boot_agent`` is ready.

    Returns ``(code, value)`` where ``code`` is
    ``no_registry`` | ``bad_id`` | ``not_found`` | ``pending`` | ``complete``.
    """
    reg = _chat_async_registry
    if reg is None:
        return ("no_registry", None)
    rid = (rid or "").strip()
    if not rid:
        return ("bad_id", None)
    ent = reg.get(rid)
    if ent is None:
        return ("not_found", None)
    slot = ent["slot"]
    if not slot["event"].is_set():
        return ("pending", rid)
    result = normalize_http_chat_response(slot["result"])
    if isinstance(result, dict):
        result = merge_lab_context_values(
            result,
            experiment_id=ent.get("experiment_id"),
            lab_run_id=ent.get("lab_run_id"),
        )
    reg.pop(rid)
    try:
        boot_agent.shared.http_chat_end()
    except Exception as _he:
        print(f"[Server-v2] chat async finalize http_chat_end error: {_he}", flush=True)
    try:
        boot_agent.input_agent.log_response(result)
    except Exception as _le:
        print(f"[Server-v2] chat async finalize log_response error: {_le}", flush=True)
    return ("complete", result)


# =====================================================================
# Globals
# =====================================================================

boot_agent: BootAgent = None  # type: ignore
sensor_poller: SensorPoller = None  # type: ignore
_chat_async_registry: Optional[ChatAsyncRegistry] = None


def _init():
    global boot_agent, sensor_poller, _chat_async_registry

    print("[Elarion-v2] Booting multi-agent architecture...", flush=True)
    t0 = time.time()

    # 1. Boot agent initializes SharedResources and spawns all agents
    boot_agent = BootAgent()
    shared = boot_agent.boot()
    _maybe_enable_fast_llm_for_gpu(shared)
    _chat_async_registry = ChatAsyncRegistry(
        shared,
        ttl_sec=env_float("HAROMA_CHAT_ASYNC_TTL_SEC", 900.0),
    )

    # 2. Wire boot_agent reference into agents that need it
    boot_agent.input_agent.set_boot_agent(boot_agent)
    boot_agent.trueself_agent.set_boot_agent(boot_agent)
    for persona in boot_agent.persona_agents:
        persona.set_boot_agent(boot_agent)

    # 3. Hardware sensor adapters (push into InputAgent's buffer)
    rc = shared.resource_config
    neural_perception = rc.sensor.get("neural_perception", True)
    if neural_perception:
        warmup_neural_models()

    sensor_poller = SensorPoller(
        boot_agent.input_agent,
        adapters=[
            VisionAdapter(),
            AudioAdapter(),
            TouchAdapter(),
            SmellAdapter(),
            TasteAdapter(),
            LidarAdapter(),
            InfraredAdapter(),
            ImuAdapter(),
            GpsAdapter(),
        ],
    )

    # 4. Start all agents
    boot_agent.start_all()
    sensor_poller.start()

    elapsed = time.time() - t0
    _llm_name = getattr(shared.llm_backend, "model_name", None) or "none"
    print(
        f"[Elarion-v2] Boot complete in {elapsed:.1f}s | "
        f"tier={rc.tier_name} | "
        f"trueself=active | "
        f"personas={len(boot_agent.persona_agents)} | "
        f"llm={_llm_name}",
        flush=True,
    )


def _shutdown_save():
    if boot_agent is None:
        return
    if sensor_poller:
        sensor_poller.stop()
    boot_agent.save_and_shutdown()


atexit.register(_shutdown_save)


# =====================================================================
# JSON helpers
# =====================================================================


def _safe(obj):
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [_safe(v) for v in sorted(obj, key=str)]
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, float) and not _math.isfinite(obj):
        return 0.0
    return obj


def _json(data, status=200):
    body = json.dumps(_safe(data), default=str, ensure_ascii=False)
    return Response(body, status=status, mimetype="application/json")


def _json_lab(data, status=200):
    """JSON response with optional ``experiment_id`` / ``lab_run_id`` for lab clients."""
    if isinstance(data, dict):
        sh = boot_agent.shared if boot_agent and boot_agent.shared else None
        data = merge_lab_context(data, sh, g)
    return _json(data, status)


# =====================================================================
# Flask
# =====================================================================

app = Flask(__name__, static_folder=None)


@app.before_request
def _haroma_request_context():
    """Per-request id and timing for ``X-Haroma-Request-Id`` and structured access logs."""
    g.haroma_request_id = str(uuid.uuid4())
    g._haroma_req_t0 = time.time()


@app.before_request
def _haroma_optional_bearer_guard():
    """Require bearer token on selected routes when ``HAROMA_HTTP_BEARER_TOKEN`` is set."""
    out = verify_http_request_bearer(request)
    if out is not None:
        return _json(out[0], out[1])


@app.before_request
def _haroma_rate_limit():
    """Optional sliding-window POST limit (see :mod:`mind.http_rate_limit`)."""
    out = check_rate_limit(request)
    if out is not None:
        body = dict(out[0])
        rid = getattr(g, "haroma_request_id", None)
        if rid:
            body["request_id"] = rid
        resp = _json(body, out[1])
        resp.headers["Retry-After"] = str(int(RATE_LIMIT_WINDOW_SEC))
        if rid:
            resp.headers["X-Haroma-Request-Id"] = rid
        return resp


@app.before_request
def _haroma_lab_request_flow():
    """Single place: parse ``X-Experiment-Id`` / JSON ``experiment_id`` → ``g.lab_experiment_id``."""
    apply_lab_experiment_to_request(request, g)


@app.after_request
def _haroma_after_request(response: Response):
    """Attach ``X-Haroma-Request-Id``; optional stderr JSON access line (status + duration)."""
    rid = getattr(g, "haroma_request_id", None)
    if rid:
        response.headers["X-Haroma-Request-Id"] = rid
    if structured_log_enabled():
        t0 = getattr(g, "_haroma_req_t0", None)
        dur_ms: Optional[float] = None
        if t0 is not None:
            dur_ms = round((time.time() - t0) * 1000.0, 3)
        extra: Dict[str, Any] = {}
        exp = getattr(g, "lab_experiment_id", None)
        if exp:
            extra["experiment_id"] = str(exp)[:200]
        acc: Dict[str, Any] = {
            "request_id": rid or "",
            "status": response.status_code,
            "method": getattr(request, "method", "") or "",
            "path": getattr(request, "path", "") or "",
            "remote": getattr(request, "remote_addr", None) or "",
        }
        if dur_ms is not None:
            acc["duration_ms"] = dur_ms
        log_event("http_access", **acc, **extra)
    return response


@app.errorhandler(Exception)
def handle_exception(e):
    _tb.print_exc()
    try:
        cycle = boot_agent.shared.cycle_count if boot_agent and boot_agent.shared else 0
    except Exception:
        cycle = 0
    return _json(
        {
            "response": "[internal server error]",
            "cycle": cycle,
            "affect": {},
            "strategy": "",
        },
        status=500,
    )


@app.route("/")
def serve_index():
    return send_from_directory(_WEB_DIR, "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    if boot_agent is None or getattr(boot_agent, "input_agent", None) is None:
        return _json(
            {
                "error": "server not ready",
                "response": "[server still booting — wait for console 'Chat API' line]",
            },
            503,
        )
    data = request.get_json(silent=True) or {}
    _exp_chat = getattr(g, "lab_experiment_id", None)
    _raw_env = data.get("agent_environment")
    if _raw_env is not None and boot_agent.shared is not None:
        try:
            boot_agent.shared.set_agent_environment(_raw_env)
        except Exception as _ae:
            print(f"[Server-v2] agent_environment on /chat: {_ae}", flush=True)
    message = data.get("message", "").strip()
    if not message:
        return _json_lab({"error": "empty message"}, 400)
    if len(message) > 4096:
        return _json_lab({"error": "message too long"}, 400)
    _raw_uid = data.get("user_id") if data.get("user_id") is not None else data.get("userId")
    _chat_user_id = sanitize_user_id(_raw_uid)
    _raw_dn = (
        data.get("display_name")
        if data.get("display_name") is not None
        else data.get("displayName")
    )
    _chat_display_name = str(_raw_dn).strip()[:160] if _raw_dn is not None and str(_raw_dn).strip() else None

    # Legacy clients may send depth=fast; it is normalized to normal (no fast path).
    _default_depth = str(os.environ.get("HAROMA_CHAT_DEFAULT_DEPTH", "normal") or "normal").lower()
    if _default_depth not in ("fast", "normal"):
        _default_depth = "normal"
    depth = str(data.get("depth") or _default_depth).lower()
    if depth not in ("fast", "normal"):
        depth = "normal"
    if depth == "fast":
        depth = "normal"
    debug_recall = _json_bool(data.get("debug_recall"), False)
    trace_latency = _json_bool(data.get("trace_latency"), False) or trace_requested()
    communication_debug = _json_bool(data.get("communication_debug"), False)
    if str(os.environ.get("HAROMA_COMMUNICATION_DEBUG", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        communication_debug = True
    deliberative = _json_bool(data.get("deliberative"), False)
    if str(os.environ.get("HAROMA_LLM_DELIBERATIVE", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        deliberative = True

    if "async" in data:
        _async_req = truthy_async_flag(data.get("async"))
    else:
        _async_req = _env_truthy_flag("HAROMA_CHAT_DEFAULT_ASYNC")

    input_agent = boot_agent.input_agent
    _chat_begun = False
    _async_handoff = False
    _prev = message[:80] + ("…" if len(message) > 80 else "")
    print(
        f"[Server-v2] chat POST accepted depth={depth} chars={len(message)} preview={_prev!r}",
        flush=True,
    )

    try:
        _chat_t0 = time.time()
        boot_agent.shared.http_chat_begin(depth=depth)
        _chat_begun = True
        slot = input_agent.push_text(
            message,
            source="user",
            depth=depth,
            debug_recall=debug_recall,
            trace_latency=trace_latency,
            communication_debug=communication_debug,
            deliberative=deliberative,
            user_id=_chat_user_id,
            display_name=_chat_display_name,
            experiment_id=_exp_chat,
        )
        _after_push = time.time() - _chat_t0
        _push_note = "message queued; persona cycle runs async"
        print(
            f"[Server-v2] chat push_text returned after {_after_push:.2f}s ({_push_note})",
            flush=True,
        )

        if _async_req:
            reg = _chat_async_registry
            if reg is None:
                return _json_lab({"error": "async chat not initialized"}, 503)
            rid = reg.register(
                slot,
                experiment_id=_exp_chat,
                lab_run_id=getattr(boot_agent.shared, "lab_run_id", None),
            )
            _async_handoff = True
            print(f"[Server-v2] chat async handoff request_id={rid}", flush=True)
            return _json_lab(
                {
                    "status": "pending",
                    "request_id": rid,
                    "poll": f"/chat/result?id={rid}",
                    "wait": f"/chat/wait?id={rid}",
                },
                202,
            )

        _chat_wait = _http_chat_wait_sec(depth)
        if slot["event"].wait(timeout=_chat_wait):
            result = normalize_http_chat_response(slot["result"])
            input_agent.log_response(result)
            print(
                f"[Server-v2] chat OK total={time.time() - _chat_t0:.2f}s",
                flush=True,
            )
            return _json_lab(result)

        print(
            f"[Server-v2] chat client wait timed out ({_chat_wait}s) "
            f"after push_text={_after_push:.2f}s",
            flush=True,
        )
        return _json_lab(
            {
                "response": "[Elarion is still thinking... try again shortly]",
                "cycle": boot_agent.shared.cycle_count,
                "affect": {},
                "strategy": "",
            }
        )
    except Exception as exc:
        print(f"[Server-v2] chat error: {exc}", flush=True)
        return _json_lab({"error": "internal error"}, 500)
    finally:
        if _chat_begun and not _async_handoff:
            try:
                boot_agent.shared.http_chat_end()
            except Exception as _he:
                print(f"[Server-v2] http_chat_end error: {_he}", flush=True)


@app.route("/chat/result", methods=["GET", "DELETE"])
def chat_result():
    """Poll or cancel the outcome of async ``POST /chat``."""
    if boot_agent is None or getattr(boot_agent, "input_agent", None) is None:
        return _json(
            {
                "error": "server not ready",
                "response": "[server still booting — wait for console 'Chat API' line]",
            },
            503,
        )
    rid = (request.args.get("id") or "").strip()
    if not rid:
        return _json({"error": "missing id query parameter"}, 400)

    if request.method == "DELETE":
        reg = _chat_async_registry
        if reg is None:
            return _json({"error": "async chat not initialized"}, 503)
        outcome = reg.cancel(rid)
        if outcome == "gone":
            return _json({"error": "unknown or expired request_id"}, 404)
        if outcome == "not_pending":
            return _json(
                {
                    "error": "request already complete; use GET /chat/result to fetch",
                },
                409,
            )
        return _json({"status": "cancelled", "request_id": rid})

    code, val = _async_chat_resolve_ready(rid)
    if code == "no_registry":
        return _json({"error": "async chat not initialized"}, 503)
    if code == "not_found":
        return _json({"error": "unknown or expired request_id"}, 404)
    if code == "pending":
        reg = _chat_async_registry
        ent = reg.get(rid) if reg is not None else None
        pend: Dict[str, Any] = {"status": "pending", "request_id": rid}
        if ent:
            if ent.get("experiment_id"):
                pend["experiment_id"] = ent["experiment_id"]
            if ent.get("lab_run_id"):
                pend["lab_run_id"] = str(ent["lab_run_id"])
        return _json(pend, 200)
    return _json(val)


def _chat_wait_sse_generator(rid: str, cap_sec: float, keepalive_sec: float):
    """Yield SSE ``data:`` lines until result, timeout, or error."""
    code, val = _async_chat_resolve_ready(rid)
    if code == "no_registry":
        yield f"data: {json.dumps(_safe({'error': 'async chat not initialized'}))}\n\n"
        return
    if code == "bad_id":
        yield f"data: {json.dumps(_safe({'error': 'missing id'}))}\n\n"
        return
    if code == "not_found":
        yield f"data: {json.dumps(_safe({'error': 'unknown or expired request_id'}))}\n\n"
        return
    if code == "complete":
        yield f"data: {json.dumps(_safe(val))}\n\n"
        return

    reg = _chat_async_registry
    deadline = time.time() + cap_sec
    ka = max(1.0, float(keepalive_sec))

    while time.time() < deadline:
        if reg is None:
            yield f"data: {json.dumps(_safe({'error': 'async chat not initialized'}))}\n\n"
            return
        ent = reg.get(rid)
        if ent is None:
            yield f"data: {json.dumps(_safe({'error': 'unknown or expired request_id'}))}\n\n"
            return
        slot = ent.get("slot") or {}
        ev = slot.get("event")
        if ev is None:
            break
        if ev.is_set():
            break
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        ev.wait(timeout=min(ka, remaining))
        if ev.is_set():
            break
        yield ": sse-keepalive\n\n"

    code2, val2 = _async_chat_resolve_ready(rid)
    if code2 == "complete":
        yield f"data: {json.dumps(_safe(val2))}\n\n"
        return
    if code2 == "pending":
        yield f"data: {json.dumps(_safe({'status': 'pending_timeout', 'request_id': rid}))}\n\n"
        return
    if code2 == "not_found":
        yield f"data: {json.dumps(_safe({'error': 'result already delivered or expired'}))}\n\n"
        return
    yield f"data: {json.dumps(_safe({'error': 'async chat not initialized'}))}\n\n"


@app.route("/chat/wait", methods=["GET"])
def chat_wait():
    """Server-Sent Events stream until async chat completes or ``max_wait_sec``."""
    if boot_agent is None or getattr(boot_agent, "input_agent", None) is None:
        return _json(
            {
                "error": "server not ready",
                "response": "[server still booting — wait for console 'Chat API' line]",
            },
            503,
        )
    rid = (request.args.get("id") or "").strip()
    if not rid:
        return _json({"error": "missing id query parameter"}, 400)

    raw_cap = (request.args.get("max_wait_sec") or "").strip()
    try:
        cap = float(raw_cap) if raw_cap else float(_http_chat_wait_sec("normal"))
    except (TypeError, ValueError):
        cap = float(_http_chat_wait_sec("normal"))
    cap = max(5.0, min(cap, 7200.0))
    keepalive = env_float("HAROMA_CHAT_SSE_KEEPALIVE_SEC", 15.0)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }

    @stream_with_context
    def _gen():
        yield from _chat_wait_sse_generator(rid, cap, keepalive)

    return Response(_gen(), mimetype="text/event-stream", headers=headers)


@app.route("/sensor", methods=["POST"])
def push_sensor():
    if boot_agent is None or boot_agent.shared is None:
        return _json({"error": "not booted"}, 503)
    data = request.get_json(silent=True) or {}
    channel = data.get("channel", "unknown")
    payload = data.get("data", data)
    if str(channel).lower() in ("agent_environment", "environment"):
        pe = payload if isinstance(payload, dict) else data
        result = boot_agent.shared.set_agent_environment(pe)
        return _json(
            {
                "status": "stored",
                "channel": channel,
                "result": result,
                "agent_environment": boot_agent.shared.agent_environment_status(),
            }
        )
    boot_agent.input_agent.push_sensor(channel, payload)
    return _json(
        {
            "status": "buffered",
            "channel": channel,
            "buffer": boot_agent.input_agent.buffer_stats(),
        }
    )


@app.route("/agent/environment", methods=["POST"])
def post_agent_environment():
    """Accept a structured environment snapshot (general-agent HTTP integrator)."""
    if boot_agent is None or boot_agent.shared is None:
        return _json({"error": "not booted"}, 503)
    raw = request.get_json(silent=True)
    if raw is None or not isinstance(raw, dict):
        return _json({"error": "expected JSON object"}, 400)
    result = boot_agent.shared.set_agent_environment(raw)
    return _json_lab(
        {
            "status": "ok" if result.get("ok") else "error",
            "result": result,
            "agent_environment": boot_agent.shared.agent_environment_status(),
        },
        status=200 if result.get("ok") else 400,
    )


@app.route("/robot/bridge/feedback", methods=["POST"])
def post_robot_bridge_feedback():
    """Merge executor command results into ``agent_environment.extensions.robot_bridge``.

    Body matches :func:`mind.robot_execution_contract.normalize_feedback_payload`
    (``bridge_schema_version``, ``correlation_id``, ``results``).
    """
    if boot_agent is None or boot_agent.shared is None:
        return _json({"error": "not booted"}, 503)
    raw = request.get_json(silent=True)
    if raw is None or not isinstance(raw, dict):
        return _json({"error": "expected JSON object"}, 400)
    result = boot_agent.shared.merge_robot_bridge_feedback(raw)
    return _json_lab(
        {
            "status": "ok" if result.get("ok") else "error",
            "result": result,
            "agent_environment": boot_agent.shared.agent_environment_status(),
        },
        status=200 if result.get("ok") else 400,
    )


@app.route("/research/manifest", methods=["GET"])
def research_manifest():
    """Return the boot-time run manifest (git, soul file hash, HAROMA_* env snapshot)."""
    if boot_agent is None or boot_agent.shared is None:
        return _json({"error": "not booted"}, 503)
    m = getattr(boot_agent.shared, "run_manifest", None)
    if not isinstance(m, dict) or not m:
        return _json({"error": "manifest_unavailable"}, 503)
    return _json(m)


@app.route("/research/snapshot", methods=["GET"])
def research_snapshot():
    """Bundle run manifest, recent lab HTTP events, ``agent_environment``, and readiness summary."""
    if boot_agent is None or boot_agent.shared is None:
        return _json({"error": "not booted"}, 503)
    s = boot_agent.shared
    lr = getattr(s, "lab_run_id", None)
    m = getattr(s, "run_manifest", None)
    return _json(
        {
            "lab_run_id": str(lr) if lr else None,
            "run_manifest": m if isinstance(m, dict) else None,
            "lab_experiment_events": lab_events_snapshot(48),
            "agent_environment": s.agent_environment_status(),
            "embodiment_readiness": embodiment_readiness_summary(s),
        }
    )


@app.route("/status", methods=["GET"])
def status():
    if boot_agent is None or boot_agent.shared is None:
        return _json({"error": "not booted"}, 503)
    return _json(
        build_http_status_payload(boot_agent, sensor_poller, _chat_async_registry)
    )


@app.route("/introspect", methods=["GET"])
def introspect():
    if boot_agent is None or boot_agent.shared is None:
        return _json({"error": "not booted"}, 503)
    s = boot_agent.shared
    personas = {p.agent_id: p.stats() for p in boot_agent.persona_agents}
    return _json(
        {
            "shared": s.summary(),
            "trueself": boot_agent.trueself_agent.stats(),
            "personas": personas,
            "background": boot_agent.background_agent.stats(),
            "input": boot_agent.input_agent.stats(),
            "message_bus": boot_agent.bus.stats(),
        }
    )


@app.route("/resource", methods=["GET"])
def resource_info():
    if boot_agent is None or boot_agent.shared is None:
        return _json({"error": "not booted"}, 503)
    rc = boot_agent.shared.resource_config
    return _json(
        {
            "resource_config": rc.summary(),
            "llm_backend": boot_agent.shared.llm_backend.stats(),
            "active_modules": boot_agent.shared.organ_registry.summary(),
            "skipped_modules": rc.cycle.get("skip_modules", []),
        }
    )


@app.route("/save", methods=["POST"])
def save_state():
    if boot_agent is None or boot_agent.shared is None:
        return _json({"error": "not booted"}, 503)
    try:
        result = boot_agent.shared.persistence.save(boot_agent.shared)
        return _json({"status": "saved", "details": result})
    except Exception as e:
        _tb.print_exc()
        return _json({"status": "error", "details": "save failed"}, 500)


@app.route("/teach", methods=["POST"])
def teach_meaning():
    """Register term→meaning glosses (JSON: {"term","meaning"} or {"items":[...]})."""
    if boot_agent is None or boot_agent.shared is None:
        return _json({"error": "not booted"}, 503)
    data = request.get_json(silent=True) or {}
    lex = boot_agent.shared.meaning_lexicon
    items = data.get("items")
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                lex.register(it.get("term", ""), it.get("meaning", ""))
    else:
        lex.register(data.get("term", ""), data.get("meaning", ""))
    return _json(
        {
            "status": "ok",
            "terms_stored": len(lex),
        }
    )


@app.route("/laws", methods=["GET", "POST"])
def laws_http():
    """Symbolic law snapshot (GET) or declare/revoke (POST). See :func:`mind.symbolic_law_api.handle_laws_request`."""
    if boot_agent is None or boot_agent.shared is None:
        return _json({"error": "not booted"}, 503)
    return handle_laws_request(getattr(boot_agent.shared, "law", None), _json)


# =====================================================================
# Launch
# =====================================================================


def _kill_port(port: int):
    """Best-effort free *port* before bind. Windows: netstat/taskkill; other OS: no-op (add lsof/fuser later)."""
    if os.name != "nt":
        return
    import subprocess

    try:
        out = subprocess.check_output(
            f'netstat -ano | findstr ":{port}" | findstr "LISTENING"',
            shell=True,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        pids = set()
        for line in out.strip().splitlines():
            parts = line.split()
            if parts:
                pids.add(parts[-1])
        my_pid = str(os.getpid())
        for pid in pids:
            if pid != my_pid and pid != "0":
                subprocess.call(
                    f"taskkill /PID {pid} /F",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(
                    f"[Elarion-v2] Killed stale process {pid} on port {port}",
                    flush=True,
                )
    except Exception as _e:
        print(f"[Server-v2] kill_port error: {_e}", flush=True)


def launch():
    from mind.deploy_config import display_base_url, http_listen_host, http_listen_port

    _bind_host = http_listen_host()
    _port = http_listen_port()
    _base = display_base_url()
    _kill_port(_port)
    time.sleep(1)
    _init()
    print("[Elarion-v2] -----------------------------------------", flush=True)
    print("[Elarion-v2]  Architecture : Multi-Agent v6 (TrueSelf)", flush=True)
    print(f"[Elarion-v2]  Listen       : {_bind_host}:{_port}  (open: {_base})", flush=True)
    print(f"[Elarion-v2]  Web UI       : {_base}", flush=True)
    print(f"[Elarion-v2]  Chat API     : POST {_base}/chat", flush=True)
    print(f"[Elarion-v2]  Sensor       : POST {_base}/sensor", flush=True)
    print(
        f"[Elarion-v2]  Agent env    : POST {_base}/agent/environment",
        flush=True,
    )
    print(
        f"[Elarion-v2]  Robot bridge : POST {_base}/robot/bridge/feedback",
        flush=True,
    )
    print(
        f"[Elarion-v2]  Lab manifest : GET  {_base}/research/manifest",
        flush=True,
    )
    print(
        f"[Elarion-v2]  Lab snapshot : GET  {_base}/research/snapshot",
        flush=True,
    )
    print(f"[Elarion-v2]  Status       : GET  {_base}/status", flush=True)
    print(f"[Elarion-v2]  Laws         : GET/POST {_base}/laws", flush=True)
    print(f"[Elarion-v2]  Teach        : POST {_base}/teach", flush=True)
    _llm_pri = str(os.environ.get("HAROMA_CHAT_LLM_PRIMARY", "1") or "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    print(
        "[Elarion-v2]  Chat LLM     : "
        f"{'packed context (soul+memories+KG) per turn' if _llm_pri else 'organic path only (set HAROMA_CHAT_LLM_PRIMARY=1 to enable)'}",
        flush=True,
    )
    _cap = llm_context_timeout_seconds()
    _cap_s = "unlimited" if _cap is None else f"{int(_cap)}s"
    _w_chat = _http_chat_wait_sec()
    print(
        f"[Elarion-v2]  Chat HTTP wait: {_w_chat}s | "
        f"packed LLM cap (global): {_cap_s}",
        flush=True,
    )
    if configured_bearer_secret():
        ps = ", ".join(sorted(protected_path_set()))
        print(f"[Elarion-v2]  HTTP bearer   : ON (protected: {ps})", flush=True)
    else:
        print("[Elarion-v2]  HTTP bearer   : off (set HAROMA_HTTP_BEARER_TOKEN to require)", flush=True)
    _rlpm = rate_limit_per_minute()
    if _rlpm > 0:
        print(
            f"[Elarion-v2]  HTTP rate limit: { _rlpm } POST/min per IP (chat, sensor, env, bridge, teach, save)",
            flush=True,
        )
    else:
        print(
            "[Elarion-v2]  HTTP rate limit: off (set HAROMA_HTTP_RATE_LIMIT_PER_MIN)",
            flush=True,
        )
    if _env_truthy_flag("HAROMA_STRUCTURED_LOG"):
        print("[Elarion-v2]  Structured log: ON (stderr JSON per response, status+duration)", flush=True)
    print("[Elarion-v2] -----------------------------------------", flush=True)
    # Avoid Flask's Click/colorama server banner — on some Windows consoles it
    # raises OSError (WinError 6: invalid handle) when writing to stdout.
    os.environ.setdefault("NO_COLOR", "1")
    run_simple(
        _bind_host,
        _port,
        app,
        threaded=True,
        use_reloader=False,
        use_debugger=False,
    )


if __name__ == "__main__":
    launch()
