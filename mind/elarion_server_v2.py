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

Env ``HAROMA_INPUT_PIPELINE_LOG=1`` (legacy: ``HAROMA_CHAT_PIPELINE_LOG``): print
``[InputPipeline]`` lines with a per-turn ``trace=`` id (see :mod:`mind.chat_pipeline_log`)
so you can see the last stage reached before a stall. Use ``HAROMA_INPUT_PIPELINE_TIMING=1``
(legacy: ``HAROMA_CHAT_PIPELINE_TIMING=1``) for ``seg=…ms cum=…ms`` on each line, or set
``HAROMA_INPUT_PIPELINE_LOG=full`` for logs + timing together.

``HAROMA_LLM_DUMMY_REPLY=1`` skips real ``generate_chat`` only inside packed-LLM
inference; the full persona stack (including ``persona_neural_section`` / shared
neural read lock and encoder) still runs — use that to measure **pipeline** latency
without decode. ``HAROMA_LLM_DUMMY_FULL_PACK``
controls whether :mod:`engine.LLMContextReasoner` runs the expensive ``build_messages``
path while dummy (pack-size profiling).

``HAROMA_CHAT_INPUT_PRIORITY`` (default ``1``): while the input pipeline is busy
(HTTP ``/chat`` and/or InputAgent queues), defer non-critical persona/TrueSelf/background
work so input completes first
(see :mod:`mind.chat_priority`). Set to ``0`` to interleave as before.

``HAROMA_TRUESELF_HTTP_CHAT_UNPHASED`` (default ``1``): when ``HAROMA_PERSONA_PHASED_CYCLE=1``,
still run **HTTP-traced** TrueSelf conversant turns in **one** synchronous pass (no
multi-tick phased scheduling), so async ``/chat`` is not stretched across ticks.
Set to ``0`` to use phased mode for TrueSelf like other agents.

After each persona cognitive cycle, if **HTTP /chat or text queues** are still
active (see :func:`mind.chat_priority.input_pipeline_yield_busy`; sensor-only
backlog does not trigger this), agents sleep ``HAROMA_POST_CYCLE_INPUT_BUSY_SLEEP_SEC``
(fallback: ``HAROMA_POST_CYCLE_CHAT_BUSY_SLEEP_SEC``, default ``0.05``), up to
``HAROMA_POST_CYCLE_INPUT_BUSY_MAX_RETRIES`` (fallback:
``HAROMA_POST_CYCLE_CHAT_BUSY_MAX_RETRIES``, default ``1``; ``0`` = retry until
idle). See :meth:`agents.persona_agent.PersonaAgent._yield_after_cycle_if_input_pipeline_busy`.
Full :func:`mind.chat_priority.input_pipeline_busy` still includes sensor queues
for deferral and status snapshots.

Input latency: ``HAROMA_INPUT_TICK_INTERVAL_SEC`` overrides InputAgent tick
(default in ``soul/agents.json`` / merged config is ~0.12s). User HTTP messages
are queued with **priority** over other text. For sub-2s turns on CPU, typical
``.env`` tuning includes ``HAROMA_INPUT_CHAT_SKIP_ENCODER=1`` (skip SBERT in
InputAgent), ``HAROMA_TRUESELF_USER_CHAT_FAST_RECALL=1`` (FAISS-only recall),
``HAROMA_TRUESELF_USER_CHAT_BUDGET_SEC`` / ``HAROMA_TRUESELF_USER_CHAT_SKIP_PERSONA_ENCODE``,
and ``HAROMA_POST_CYCLE_INPUT_BUSY_SLEEP_SEC=0`` — see :mod:`agents.persona_agent` and
:mod:`agents.input_agent`. Optional JSON ``depth`` is accepted for
compatibility and normalized to ``normal``. Set
``HAROMA_BG_DEFER_TRAINING_ON_INPUT_PIPELINE=0`` (or legacy
``HAROMA_BG_DEFER_TRAINING_ON_HTTP_CHAT=0``) to keep background training while the
input pipeline is active (more neural lock contention).

Non-blocking chat: POST ``{"message": "...", "async": true}`` returns ``202`` with
``request_id``; poll ``GET /chat/result?id=<uuid>`` until ``status`` is absent
and the payload matches synchronous ``/chat``. Pending entries expire after
``HAROMA_CHAT_ASYNC_TTL_SEC`` (default ``900``). Single-process only.
``POST /robot/bridge/feedback`` merges on-robot executor results into
``agent_environment.extensions.robot_bridge`` (see :mod:`integrations.robot_http_bridge`).

``POST /chat`` may include optional ``sensor_data`` (alias ``sensors``): an object whose
keys are channel names (e.g. ``lidar``, ``gps``) and values are a reading or list of
readings; they are merged into the same turn as ``message`` (see :mod:`agents.input_agent`).
The cognitive path also receives ``senses_numpy``: per-modality ``float32`` arrays
(empty if missing), including ``text_embedding``, from :mod:`mind.sense_numpy_bundle`.

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

``HAROMA_BG_DEFER_TRAINING_CAP_SEC`` (optional): when set with defer-on-input-pipeline,
background training still runs at least once per this many seconds even while the
input pipeline is busy.

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
**GET poll limit (optional):** ``HAROMA_HTTP_GET_RATE_LIMIT_PER_MIN`` — max GETs per IP per minute on ``/chat/result``; ``0`` (default) disables.

**Client IP behind a proxy (optional):** ``HAROMA_HTTP_USE_X_FORWARDED_FOR`` and
``HAROMA_HTTP_TRUSTED_PROXIES`` — :mod:`mind.client_ip` uses ``X-Forwarded-For`` only when
the direct peer is trusted (loopback or listed CIDRs).

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
Optional ``HAROMA_CHAT_TRACE=1`` or ``{"trace_latency": true}`` adds ``latency_trace``
(see :mod:`agents.chat_latency`). With ``HAROMA_LLM_DUMMY_REPLY=1``, ``latency_trace``
is attached automatically; set ``HAROMA_LLM_DUMMY_NO_LATENCY_TRACE=1`` to disable.

Env ``HAROMA_LLM_WARMUP`` (default ``1``): after a local GGUF loads, run a
short decode during boot so the first chat avoids cold-start latency. Set
``0`` to skip. ``HAROMA_LLM_WARMUP_MAX_TOKENS`` caps decode length (default ``24``).

Local GGUF loads at **boot** by default (including Windows). Set ``HAROMA_LLM_LAZY_LOCAL=1``
to defer loading until first chat if llama-cpp crashes during init. With lazy load,
**background prefetch** (``HAROMA_LLM_LAZY_PREFETCH``, default ``1``) loads the model
in a daemon thread after init so the first request is still faster than cold load on-path.
"""

import math as _math
import os
import sys
import json
import time
import uuid
from typing import Any, Dict, Optional, Tuple
import atexit
import re
import threading
import traceback as _tb

from flask import (
    Flask,
    abort,
    current_app,
    g,
    has_app_context,
    make_response,
    redirect,
    request,
    send_from_directory,
    Response,
    stream_with_context,
)
from werkzeug.serving import run_simple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensors.domains import resolve_channel_to_domain

from agents.boot_agent import BootAgent
from agents.input_agent import normalize_chat_inline_sensor_data
from agents.chat_latency import (
    packed_llm_dummy_probe_active,
    packed_llm_dummy_reply_raw,
    trace_requested,
)
from utils.coerce_bool import json_bool as _json_bool
from mind.cognitive_contracts import (
    llm_context_timeout_seconds,
    normalize_http_chat_response,
)
from mind.symbolic_law_api import handle_laws_request
from mind.chat_async_registry import ChatAsyncRegistry, truthy_async_flag
from mind.chat_pipeline_log import log_input_pipeline, pipeline_trace_end, trace_id_from_slot
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
    get_rate_limit_per_minute,
    rate_limit_per_minute,
)
from mind.lab_research import (
    apply_lab_experiment_to_request,
    lab_events_snapshot,
    merge_lab_context,
    merge_lab_context_values,
)
from mind.robot_readiness import embodiment_readiness_summary
from mind.client_ip import get_effective_client_ip
from mind.structured_log import log_event, structured_log_enabled
from mind.system_snapshot import build_http_status_payload
from mind.user_identity import sanitize_user_id
from mind.server_state import HaromaServerState, get_haroma_server_state
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
    register_http_chat_busy_checker,
    warmup_neural_models,
)

_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
# Flat files only under web/ (no path traversal) for logo / favicon assets.
_SAFE_WEB_ASSET = re.compile(r"^[A-Za-z0-9._-]+\.(svg|png|ico|webp)$")

app = Flask(__name__, static_folder=None)
get_haroma_server_state(app)


def _haroma() -> HaromaServerState:
    """Haroma state for the active Flask app, or the module ``app`` outside any context.

    Uses :data:`current_app` when an application context is active (requests, tests,
    ``app.app_context()``); otherwise falls back to the process default ``app`` (e.g.
    :func:`_init` during :func:`launch`).
    """
    if has_app_context():
        return get_haroma_server_state(current_app)
    return get_haroma_server_state(app)


def _boot() -> Optional[BootAgent]:
    """The booted multi-agent graph, or ``None`` before :func:`_init` completes."""
    return _haroma().boot_agent


def _http_server_not_ready_chat() -> Any:
    """503 when :class:`InputAgent` is not available (user-facing chat routes)."""
    return _json(
        {
            "error": "server not ready",
            "response": "[server still booting — wait for console 'Chat API' line]",
        },
        503,
    )


def _http_not_booted() -> Any:
    """503 when ``SharedResources`` is not attached (most API routes)."""
    return _json({"error": "not booted"}, 503)


def _require_boot_with_input() -> Tuple[Optional[BootAgent], Any]:
    """``(boot_agent, None)`` or ``(None, error_response)`` for chat/poll routes.

    Requires ``input_agent`` and ``shared`` — synchronous and async chat both use
    ``shared`` (e.g. ``http_chat_begin`` / environment merge).
    """
    ba = _boot()
    if ba is None or getattr(ba, "input_agent", None) is None:
        return None, _http_server_not_ready_chat()
    if ba.shared is None:
        return None, _http_not_booted()
    return ba, None


def _require_boot_with_shared() -> Tuple[Optional[BootAgent], Any]:
    """``(boot_agent, None)`` or ``(None, error_response)`` when ``shared`` is required."""
    ba = _boot()
    if ba is None or ba.shared is None:
        return None, _http_not_booted()
    return ba, None


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

    Uses atomic ``pop`` so concurrent polls cannot double-finalize the same slot.
    """
    st = _haroma()
    reg = st.chat_async_registry
    if reg is None:
        return ("no_registry", None)
    rid = (rid or "").strip()
    if not rid:
        return ("bad_id", None)
    # Peek first (cheap, does not remove).
    ent = reg.get(rid)
    if ent is None:
        cached = reg.get_completed(rid)
        if cached is not None:
            return ("complete", cached)
        return ("not_found", None)
    slot = ent["slot"]
    if not slot["event"].is_set():
        return ("pending", rid)
    # Atomically consume: only the winner proceeds with finalization.
    ent = reg.pop(rid)
    if ent is None:
        # Lost a race with another GET: winner may not have called ``remember_completed`` yet.
        cached = reg.get_completed(rid)
        if cached is not None:
            return ("complete", cached)
        for _ in range(40):
            time.sleep(0.003)
            cached = reg.get_completed(rid)
            if cached is not None:
                return ("complete", cached)
        return ("not_found", None)
    result = normalize_http_chat_response(ent["slot"]["result"])
    if isinstance(result, dict):
        result = merge_lab_context_values(
            result,
            experiment_id=ent.get("experiment_id"),
            lab_run_id=ent.get("lab_run_id"),
        )
    ba = st.boot_agent
    if ba is not None:
        try:
            ba.shared.http_chat_end()
        except Exception as _he:
            print(f"[Server-v2] chat async finalize http_chat_end error: {_he}", flush=True)
        try:
            ba.input_agent.log_response(result)
        except Exception as _le:
            print(f"[Server-v2] chat async finalize log_response error: {_le}", flush=True)
    try:
        reg.remember_completed(rid, result)
    except Exception as _rc:
        print(f"[Server-v2] chat async remember_completed error: {_rc}", flush=True)
    _atid = trace_id_from_slot(ent.get("slot"))
    log_input_pipeline(
        "http.async_slot_ready",
        trace_id=_atid,
        detail=f"rid={rid}",
    )
    pipeline_trace_end(_atid)
    return ("complete", result)


# =====================================================================
# Boot / shutdown (state on Flask app — :func:`mind.server_state.get_haroma_server_state`)
# =====================================================================

_INIT_LOCK = threading.Lock()


def _init():
    """Construct agents once per process. Thread-safe; second call is a no-op."""
    with _INIT_LOCK:
        st = _haroma()
        if st.boot_agent is not None:
            print(
                "[Elarion-v2] Boot skipped: already initialized (use one launch() per process).",
                flush=True,
            )
            return

        print("[Elarion-v2] Booting multi-agent architecture...", flush=True)
        t0 = time.time()

        # 1. Boot agent initializes SharedResources and spawns all agents
        ba = BootAgent()
        shared = ba.boot()
        _maybe_enable_fast_llm_for_gpu(shared)
        st.chat_async_registry = ChatAsyncRegistry(
            shared,
            ttl_sec=env_float("HAROMA_CHAT_ASYNC_TTL_SEC", 900.0),
        )

        # 2. Wire boot_agent reference into agents that need it
        ba.input_agent.set_boot_agent(ba)
        ba.trueself_agent.set_boot_agent(ba)
        for persona in ba.persona_agents:
            persona.set_boot_agent(ba)

        # 3. Hardware sensor adapters (push into InputAgent's buffer)
        rc = shared.resource_config
        neural_perception = rc.sensor.get("neural_perception", True)
        if neural_perception:
            warmup_neural_models()

        st.sensor_poller = SensorPoller(
            ba.input_agent,
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

        st.boot_agent = ba
        register_http_chat_busy_checker(
            lambda: int(getattr(shared, "http_chat_inflight", 0) or 0) > 0
        )

        # 4. Start all agents
        ba.start_all()
        st.sensor_poller.start()

        elapsed = time.time() - t0
        _llm_name = getattr(shared.llm_backend, "model_name", None) or "none"
        print(
            f"[Elarion-v2] Boot complete in {elapsed:.1f}s | "
            f"tier={rc.tier_name} | "
            f"trueself=active | "
            f"personas={len(ba.persona_agents)} | "
            f"llm={_llm_name}",
            flush=True,
        )


def _shutdown_save():
    st = _haroma()
    if st.boot_agent is None:
        return
    if st.sensor_poller:
        st.sensor_poller.stop()
    st.boot_agent.save_and_shutdown()


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
        ba = _boot()
        sh = ba.shared if ba and ba.shared else None
        data = merge_lab_context(data, sh, g)
    return _json(data, status)


def _http_async_chat_not_initialized() -> Any:
    """503 when async chat registry was never wired (should not happen after full boot)."""
    return _json_lab({"error": "async chat not initialized"}, 503)


# =====================================================================
# Flask routes (``app`` is created above with ``HaromaServerState``)
# =====================================================================


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
    """Optional sliding-window rate limit for POST and (if configured) GET poll routes."""
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
            "remote": get_effective_client_ip(request),
        }
        if dur_ms is not None:
            acc["duration_ms"] = dur_ms
        log_event("http_access", **acc, **extra)
    return response


@app.errorhandler(Exception)
def handle_exception(e):
    _tb.print_exc()
    try:
        ba = _boot()
        cycle = ba.shared.cycle_count if ba and ba.shared else 0
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
    resp = make_response(send_from_directory(_WEB_DIR, "index.html"))
    resp.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
    return resp


@app.route("/assets/<path:filename>")
def serve_web_asset(filename: str):
    """Serve small static files from ``web/`` (logo, icons)."""
    base = os.path.basename(filename)
    if base != filename or not _SAFE_WEB_ASSET.match(base):
        abort(404)
    path = os.path.join(_WEB_DIR, base)
    if not os.path.isfile(path):
        abort(404)
    resp = make_response(send_from_directory(_WEB_DIR, base))
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.route("/favicon.ico")
def favicon():
    """Browsers request /favicon.ico by default; point at our SVG mark."""
    path = os.path.join(_WEB_DIR, "elarion-mark.svg")
    if not os.path.isfile(path):
        abort(404)
    return redirect("/assets/elarion-mark.svg", code=302)


@app.route("/chat", methods=["POST"])
def chat():
    ba, _boot_err = _require_boot_with_input()
    if _boot_err is not None:
        return _boot_err
    data = request.get_json(silent=True) or {}
    _exp_chat = getattr(g, "lab_experiment_id", None)
    _raw_env = data.get("agent_environment")
    if _raw_env is not None and ba.shared is not None:
        try:
            ba.shared.set_agent_environment(_raw_env)
        except Exception as _ae:
            print(f"[Server-v2] agent_environment on /chat: {_ae}", flush=True)
    message = data.get("message", "").strip()
    if not message:
        return _json_lab({"error": "empty message"}, 400)
    if len(message) > 4096:
        return _json_lab({"error": "message too long"}, 400)
    _inline_sensors = normalize_chat_inline_sensor_data(
        data.get("sensor_data")
        if data.get("sensor_data") is not None
        else data.get("sensors"),
    )
    _raw_uid = data.get("user_id") if data.get("user_id") is not None else data.get("userId")
    _chat_user_id = sanitize_user_id(_raw_uid)
    _raw_dn = (
        data.get("display_name")
        if data.get("display_name") is not None
        else data.get("displayName")
    )
    _chat_display_name = str(_raw_dn).strip()[:160] if _raw_dn is not None and str(_raw_dn).strip() else None

    _default_depth = str(os.environ.get("HAROMA_CHAT_DEFAULT_DEPTH", "normal") or "normal").lower()
    if _default_depth != "normal":
        _default_depth = "normal"
    depth = str(data.get("depth") or _default_depth).lower()
    if depth != "normal":
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

    input_agent = ba.input_agent
    _chat_begun = False
    _async_handoff = False
    _pipeline_tid: Optional[str] = None
    _prev = message[:80] + ("…" if len(message) > 80 else "")
    print(
        f"[Server-v2] chat POST accepted depth={depth} chars={len(message)} preview={_prev!r}",
        flush=True,
    )

    try:
        _chat_t0 = time.time()
        ba.shared.http_chat_begin(depth=depth)
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
            inline_sensor_data=_inline_sensors,
        )
        _ptid = trace_id_from_slot(slot)
        _pipeline_tid = _ptid
        log_input_pipeline("http.after_push_text", trace_id=_ptid)
        _after_push = time.time() - _chat_t0
        _push_note = "message queued; persona cycle runs async"
        print(
            f"[Server-v2] chat push_text returned after {_after_push:.2f}s ({_push_note})",
            flush=True,
        )

        if _async_req:
            reg = _haroma().chat_async_registry
            if reg is None:
                return _http_async_chat_not_initialized()
            rid = reg.register(
                slot,
                experiment_id=_exp_chat,
                lab_run_id=getattr(ba.shared, "lab_run_id", None),
            )
            _async_handoff = True
            print(f"[Server-v2] chat async handoff request_id={rid}", flush=True)
            log_input_pipeline(
                "http.async_202",
                trace_id=_ptid,
                detail=f"request_id={rid}",
            )
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
        log_input_pipeline(
            "http.sync_wait_start",
            trace_id=_ptid,
            detail=f"timeout_sec={_chat_wait}",
        )
        if slot["event"].wait(timeout=_chat_wait):
            log_input_pipeline("http.sync_wait_done", trace_id=_ptid)
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
        log_input_pipeline(
            "http.sync_wait_timeout",
            trace_id=_ptid,
            detail=f"timeout_sec={_chat_wait}",
        )
        return _json_lab(
            {
                "response": "[Elarion is still thinking... try again shortly]",
                "cycle": ba.shared.cycle_count,
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
                ba.shared.http_chat_end()
            except Exception as _he:
                print(f"[Server-v2] http_chat_end error: {_he}", flush=True)
        if not _async_handoff:
            pipeline_trace_end(_pipeline_tid)


@app.route("/chat/result", methods=["GET", "DELETE"])
def chat_result():
    """Poll or cancel the outcome of async ``POST /chat``."""
    ba, _boot_err = _require_boot_with_input()
    if _boot_err is not None:
        return _boot_err
    rid = (request.args.get("id") or "").strip()
    if not rid:
        return _json({"error": "missing id query parameter"}, 400)

    if request.method == "DELETE":
        reg = _haroma().chat_async_registry
        if reg is None:
            return _http_async_chat_not_initialized()
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
        return _http_async_chat_not_initialized()
    if code == "not_found":
        return _json({"error": "unknown or expired request_id"}, 404)
    if code == "pending":
        reg = _haroma().chat_async_registry
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

    reg = _haroma().chat_async_registry
    deadline = time.time() + cap_sec
    ka = max(1.0, float(keepalive_sec))

    while time.time() < deadline:
        if reg is None:
            yield f"data: {json.dumps(_safe({'error': 'async chat not initialized'}))}\n\n"
            return
        ent = reg.get(rid)
        if ent is None:
            code_mid, val_mid = _async_chat_resolve_ready(rid)
            if code_mid == "complete":
                yield f"data: {json.dumps(_safe(val_mid))}\n\n"
                return
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
    _ba, _boot_err = _require_boot_with_input()
    if _boot_err is not None:
        return _boot_err
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
    ba, _boot_err = _require_boot_with_shared()
    if _boot_err is not None:
        return _boot_err
    data = request.get_json(silent=True) or {}
    channel = data.get("channel", "unknown")
    payload = data.get("data", data)
    if str(channel).lower() in ("agent_environment", "environment"):
        pe = payload if isinstance(payload, dict) else data
        result = ba.shared.set_agent_environment(pe)
        _sd = resolve_channel_to_domain(channel)
        return _json(
            {
                "status": "stored",
                "channel": channel,
                "sense_domain": _sd.value,
                "result": result,
                "agent_environment": ba.shared.agent_environment_status(),
            }
        )
    ba.input_agent.push_sensor(channel, payload)
    _sd = resolve_channel_to_domain(channel)
    return _json(
        {
            "status": "buffered",
            "channel": channel,
            "sense_domain": _sd.value,
            "buffer": ba.input_agent.buffer_stats(),
        }
    )


@app.route("/agent/environment", methods=["POST"])
def post_agent_environment():
    """Accept a structured environment snapshot (general-agent HTTP integrator)."""
    ba, _boot_err = _require_boot_with_shared()
    if _boot_err is not None:
        return _boot_err
    raw = request.get_json(silent=True)
    if raw is None or not isinstance(raw, dict):
        return _json({"error": "expected JSON object"}, 400)
    result = ba.shared.set_agent_environment(raw)
    return _json_lab(
        {
            "status": "ok" if result.get("ok") else "error",
            "result": result,
            "agent_environment": ba.shared.agent_environment_status(),
        },
        status=200 if result.get("ok") else 400,
    )


@app.route("/robot/bridge/feedback", methods=["POST"])
def post_robot_bridge_feedback():
    """Merge executor command results into ``agent_environment.extensions.robot_bridge``.

    Body matches :func:`mind.robot_execution_contract.normalize_feedback_payload`
    (``bridge_schema_version``, ``correlation_id``, ``results``).
    """
    ba, _boot_err = _require_boot_with_shared()
    if _boot_err is not None:
        return _boot_err
    raw = request.get_json(silent=True)
    if raw is None or not isinstance(raw, dict):
        return _json({"error": "expected JSON object"}, 400)
    result = ba.shared.merge_robot_bridge_feedback(raw)
    return _json_lab(
        {
            "status": "ok" if result.get("ok") else "error",
            "result": result,
            "agent_environment": ba.shared.agent_environment_status(),
        },
        status=200 if result.get("ok") else 400,
    )


@app.route("/research/manifest", methods=["GET"])
def research_manifest():
    """Return the boot-time run manifest (git, soul file hash, HAROMA_* env snapshot)."""
    ba, _boot_err = _require_boot_with_shared()
    if _boot_err is not None:
        return _boot_err
    m = getattr(ba.shared, "run_manifest", None)
    if not isinstance(m, dict) or not m:
        return _json({"error": "manifest_unavailable"}, 503)
    return _json(m)


@app.route("/research/snapshot", methods=["GET"])
def research_snapshot():
    """Bundle run manifest, recent lab HTTP events, ``agent_environment``, and readiness summary."""
    ba, _boot_err = _require_boot_with_shared()
    if _boot_err is not None:
        return _boot_err
    s = ba.shared
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
    ba, _boot_err = _require_boot_with_shared()
    if _boot_err is not None:
        return _boot_err
    st = _haroma()
    return _json(
        build_http_status_payload(ba, st.sensor_poller, st.chat_async_registry)
    )


@app.route("/introspect", methods=["GET"])
def introspect():
    ba, _boot_err = _require_boot_with_shared()
    if _boot_err is not None:
        return _boot_err
    s = ba.shared
    personas = {p.agent_id: p.stats() for p in ba.persona_agents}
    return _json(
        {
            "shared": s.summary(),
            "trueself": ba.trueself_agent.stats(),
            "personas": personas,
            "background": ba.background_agent.stats(),
            "input": ba.input_agent.stats(),
            "message_bus": ba.bus.stats(),
        }
    )


@app.route("/resource", methods=["GET"])
def resource_info():
    ba, _boot_err = _require_boot_with_shared()
    if _boot_err is not None:
        return _boot_err
    rc = ba.shared.resource_config
    return _json(
        {
            "resource_config": rc.summary(),
            "llm_backend": ba.shared.llm_backend.stats(),
            "active_modules": ba.shared.organ_registry.summary(),
            "skipped_modules": rc.cycle.get("skip_modules", []),
        }
    )


@app.route("/save", methods=["POST"])
def save_state():
    ba, _boot_err = _require_boot_with_shared()
    if _boot_err is not None:
        return _boot_err
    try:
        result = ba.shared.persistence.save(ba.shared)
        return _json({"status": "saved", "details": result})
    except Exception as e:
        _tb.print_exc()
        return _json({"status": "error", "details": "save failed"}, 500)


@app.route("/teach", methods=["POST"])
def teach_meaning():
    """Register term→meaning glosses (JSON: {"term","meaning"} or {"items":[...]})."""
    ba, _boot_err = _require_boot_with_shared()
    if _boot_err is not None:
        return _boot_err
    data = request.get_json(silent=True) or {}
    lex = ba.shared.meaning_lexicon
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
    ba, _boot_err = _require_boot_with_shared()
    if _boot_err is not None:
        return _boot_err
    return handle_laws_request(getattr(ba.shared, "law", None), _json)


# =====================================================================
# Launch
# =====================================================================


def _kill_port_windows(port: int) -> None:
    import subprocess

    # Best-effort: free the listen port before bind. No output on "nothing to kill".
    # CMD pipelines can report returncode 0 even when findstr finds no lines; rely on stdout.
    try:
        proc = subprocess.run(
            f'netstat -ano | findstr ":{port}" | findstr "LISTENING"',
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        out = (proc.stdout or "").strip()
        if not out:
            return
        pids = set()
        for line in out.splitlines():
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
    except Exception:
        pass


def _kill_port_posix(port: int) -> None:
    """Best-effort: terminate other processes listening on *port* (requires ``lsof``)."""
    import shutil
    import signal
    import subprocess

    lsof = shutil.which("lsof")
    if not lsof:
        return
    try:
        out = subprocess.check_output(
            [lsof, "-ti", f":{port}", "-sTCP:LISTEN"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=8,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return
    my_pid = os.getpid()
    for line in out.strip().splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid == my_pid or pid <= 0:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            print(
                f"[Elarion-v2] Sent SIGTERM to stale pid {pid} on port {port}",
                flush=True,
            )
        except ProcessLookupError:
            pass
        except OSError as _e:
            print(f"[Server-v2] kill_port (posix) error for pid {pid}: {_e}", flush=True)


def _kill_port(port: int) -> None:
    """Best-effort free *port* before bind (dev convenience; not a security boundary)."""
    if os.name == "nt":
        _kill_port_windows(port)
    else:
        _kill_port_posix(port)


def launch(debug: bool = False):
    """Start the HTTP server. With ``debug=True``, Werkzeug blocks the main thread (no live robot UI).

    With ``debug=False`` (default), the server runs in a daemon thread so the main thread can
    host an optional Tk robot popup on Windows / macOS / X11 / Wayland, or ASCII status on stderr.

    Loads ``.env`` from the project root (see :func:`mind.deploy_config.load_dotenv`) so variables
    like ``HAROMA_LLM_DUMMY_REPLY`` apply when using ``python -m mind.elarion_server_v2`` as well
    as ``python main.py``.
    """
    from mind.deploy_config import display_base_url, http_listen_host, http_listen_port, load_dotenv

    _env_n = load_dotenv()
    if _env_n:
        print(
            f"[Elarion-v2] Applied {_env_n} variable(s) from .env in project root",
            flush=True,
        )

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
    print(
        "[Elarion-v2]  (Browse that URL while this process runs — not the GitHub repo page.)",
        flush=True,
    )
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
    _dr = packed_llm_dummy_reply_raw()
    print(
        "[Elarion-v2]  Packed LLM dummy: "
        f"{'ON (native generate_chat skipped; pipeline only)' if packed_llm_dummy_probe_active() else 'OFF (native decode)'} "
        f"| HAROMA_LLM_DUMMY_REPLY={_dr!r}",
        flush=True,
    )
    if configured_bearer_secret():
        ps = ", ".join(sorted(protected_path_set()))
        print(f"[Elarion-v2]  HTTP bearer   : ON (protected: {ps})", flush=True)
    else:
        print("[Elarion-v2]  HTTP bearer   : off (set HAROMA_HTTP_BEARER_TOKEN to require)", flush=True)
    _rlpm = rate_limit_per_minute()
    _getlm = get_rate_limit_per_minute()
    if _rlpm > 0 or _getlm > 0:
        _parts = []
        if _rlpm > 0:
            _parts.append(
                f"{_rlpm} POST/min (chat, sensor, env, bridge, teach, save)"
            )
        if _getlm > 0:
            _parts.append(f"{_getlm} GET/min (chat/result poll)")
        print(f"[Elarion-v2]  HTTP rate limit: {'; '.join(_parts)}", flush=True)
    else:
        print(
            "[Elarion-v2]  HTTP rate limit: off (HAROMA_HTTP_RATE_LIMIT_PER_MIN / HAROMA_HTTP_GET_RATE_LIMIT_PER_MIN)",
            flush=True,
        )
    if _env_truthy_flag("HAROMA_STRUCTURED_LOG"):
        print("[Elarion-v2]  Structured log: ON (stderr JSON per response, status+duration)", flush=True)
    if debug:
        print("[Elarion-v2]  Mode         : --debug (server on main thread; live robot head off)", flush=True)
    print("[Elarion-v2] -----------------------------------------", flush=True)
    # Avoid Flask's Click/colorama server banner — on some Windows consoles it
    # raises OSError (WinError 6: invalid handle) when writing to stdout.
    os.environ.setdefault("NO_COLOR", "1")

    def _run_http() -> None:
        run_simple(
            _bind_host,
            _port,
            app,
            threaded=True,
            use_reloader=False,
            use_debugger=False,
        )

    if debug:
        _run_http()
        return

    _http_thread = threading.Thread(target=_run_http, name="haroma-werkzeug", daemon=True)
    _http_thread.start()

    try:
        from mind.live_robot_head import attach_live_robot_head

        attach_live_robot_head(get_boot=_boot, debug=False)
    except Exception as _lrh:
        print(f"[Elarion-v2] Live robot head attach skipped: {_lrh}", flush=True)

    try:
        _http_thread.join()
    except KeyboardInterrupt:
        print("\n[Elarion-v2] Shutdown requested.", flush=True)


if __name__ == "__main__":
    launch()
