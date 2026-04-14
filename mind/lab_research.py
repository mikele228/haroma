"""
Lab research helpers: run manifests, experiment id threading, optional run directory export.

Env:

* ``HAROMA_LAB_RUN_ID`` — fixed run id (default: new uuid per process).
* ``HAROMA_LAB_RUN_DIR`` — if set, write ``run_manifest.json`` here at boot.
* ``HAROMA_LAB_SEED`` — recorded in manifest for reproducibility (caller must apply).
* ``HAROMA_LAB_LOG`` — when ``1`` / ``true`` / ``yes`` / ``on``, print one console line per
  experiment-tagged lab HTTP request (path + ``experiment_id``).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional

_MANIFEST_VERSION = 1
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LAB_EVENTS_LOCK = threading.Lock()
_LAB_EVENTS: deque = deque(maxlen=48)


def _file_sha256(path: str) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:64]
    except OSError:
        return None


def _git_probe(root: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"commit": None, "dirty": None, "error": None}
    try:
        commit = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        out["commit"] = commit or None
        rc = subprocess.call(
            ["git", "-C", root, "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            timeout=5,
        )
        out["dirty"] = rc != 0
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        out["error"] = str(e)[:200]
    return out


def _haroma_env_snapshot() -> Dict[str, str]:
    snap: Dict[str, str] = {}
    for k, v in os.environ.items():
        if k.startswith("HAROMA_") or k.startswith("ELARION_"):
            snap[k] = v[:2000]
    return dict(sorted(snap.items()))


def build_run_manifest(*, shared: Any = None) -> Dict[str, Any]:
    """Single JSON-serializable blob for reproducibility (best-effort)."""
    import platform

    agents_path = os.path.join(_PROJECT_ROOT, "soul", "agents.json")
    manifest: Dict[str, Any] = {
        "manifest_version": _MANIFEST_VERSION,
        "created_at_epoch": time.time(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cwd": os.getcwd(),
        "project_root": _PROJECT_ROOT,
        "git": _git_probe(_PROJECT_ROOT),
        "soul_agents_json_sha256": _file_sha256(agents_path),
        "env_haroma_elarion": _haroma_env_snapshot(),
        "haroma_lab_seed": str(os.environ.get("HAROMA_LAB_SEED", "") or "").strip() or None,
    }
    if shared is not None:
        try:
            rc = getattr(shared, "resource_config", None)
            if rc is not None:
                manifest["resource_tier"] = getattr(rc, "tier_name", None)
                manifest["resource_tier_id"] = getattr(rc, "tier", None)
            lb = getattr(shared, "llm_backend", None)
            if lb is not None:
                manifest["llm_backend"] = {
                    "backend_type": getattr(lb, "backend_type", None),
                    "model_name": getattr(lb, "model_name", None),
                    "available": bool(getattr(lb, "available", False)),
                }
        except Exception as exc:
            manifest["shared_probe_error"] = str(exc)[:200]
    return manifest


def maybe_write_run_manifest(lab_run_id: str, manifest: Dict[str, Any]) -> Optional[str]:
    """Write ``run_manifest.json`` under ``HAROMA_LAB_RUN_DIR`` if set. Returns path or None."""
    base = str(os.environ.get("HAROMA_LAB_RUN_DIR", "") or "").strip()
    if not base:
        return None
    try:
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, "run_manifest.json")
        payload = dict(manifest)
        payload["lab_run_id"] = lab_run_id
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        print(f"[lab] wrote run manifest: {path}", flush=True)
        return path
    except OSError as e:
        print(f"[lab] could not write run manifest: {e}", flush=True)
        return None


def init_lab_run(shared: Any) -> str:
    """Set ``shared.lab_run_id``, ``shared.run_manifest``, optional disk write. Returns lab_run_id."""
    raw = str(os.environ.get("HAROMA_LAB_RUN_ID", "") or "").strip()
    lab_run_id = raw if raw else uuid.uuid4().hex
    manifest = build_run_manifest(shared=shared)
    manifest["lab_run_id"] = lab_run_id
    try:
        manifest["shared_boot_time_sec"] = float(getattr(shared, "boot_time", 0.0) or 0.0)
    except (TypeError, ValueError):
        manifest["shared_boot_time_sec"] = None
    setattr(shared, "lab_run_id", lab_run_id)
    setattr(shared, "run_manifest", manifest)
    maybe_write_run_manifest(lab_run_id, manifest)
    return lab_run_id


def record_lab_http_event(path: str, experiment_id: Optional[str]) -> None:
    """Ring buffer of recent experiment-tagged HTTP events (for /status)."""
    if not experiment_id:
        return
    row = {"t": time.time(), "path": path[:120], "experiment_id": experiment_id[:160]}
    with _LAB_EVENTS_LOCK:
        _LAB_EVENTS.append(row)


def lab_events_snapshot(max_n: int = 16) -> List[Dict[str, Any]]:
    with _LAB_EVENTS_LOCK:
        return list(_LAB_EVENTS)[-max_n:]


def _lab_log_enabled() -> bool:
    return str(os.environ.get("HAROMA_LAB_LOG", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def clear_lab_events_for_tests() -> None:
    """Reset the in-memory experiment event ring (call from tests only)."""
    with _LAB_EVENTS_LOCK:
        _LAB_EVENTS.clear()


def merge_lab_context_values(
    payload: Dict[str, Any],
    *,
    experiment_id: Optional[str] = None,
    lab_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Copy *payload* and set lab trace fields when non-empty."""
    out = dict(payload)
    if experiment_id:
        out["experiment_id"] = str(experiment_id)[:200]
    if lab_run_id:
        out["lab_run_id"] = str(lab_run_id)[:120]
    return out


def merge_lab_context(
    payload: Dict[str, Any],
    shared: Any,
    flask_g: Any,
) -> Dict[str, Any]:
    """Copy *payload* and add ``experiment_id`` / ``lab_run_id`` from Flask ``g`` and *shared*."""
    exp = getattr(flask_g, "lab_experiment_id", None)
    lr = getattr(shared, "lab_run_id", None) if shared is not None else None
    exp_s: Optional[str] = None
    if exp is not None:
        _es = str(exp).strip()[:200]
        if _es:
            exp_s = _es
    lr_s: Optional[str] = None
    if lr is not None:
        _ls = str(lr).strip()[:120]
        if _ls:
            lr_s = _ls
    return merge_lab_context_values(payload, experiment_id=exp_s, lab_run_id=lr_s)


# HTTP routes that accept ``X-Experiment-Id`` / JSON ``experiment_id`` (POST).
LAB_EXPERIMENT_PATHS = frozenset(
    {
        "/chat",
        "/agent/environment",
        "/robot/bridge/feedback",
    }
)


def apply_lab_experiment_to_request(request: Any, flask_g: Any) -> None:
    """Set ``g.lab_experiment_id`` and append to the lab event ring.

    Call once per request from Flask ``before_request`` (after auth guards).
    Uses one JSON parse per POST to these paths (Flask caches the body).
    """
    flask_g.lab_experiment_id = None
    if getattr(request, "method", None) != "POST":
        return
    path = getattr(request, "path", "") or ""
    if path not in LAB_EXPERIMENT_PATHS:
        return
    data: Dict[str, Any] = {}
    if hasattr(request, "get_json"):
        try:
            raw = request.get_json(silent=True)
            if isinstance(raw, dict):
                data = raw
        except Exception:
            data = {}
    exp = parse_experiment_id(headers=getattr(request, "headers", None), body=data)
    flask_g.lab_experiment_id = exp
    if exp:
        record_lab_http_event(path, exp)
        if _lab_log_enabled():
            print(
                f"[HAROMA lab] {path} experiment_id={str(exp)[:120]}",
                flush=True,
            )


def parse_experiment_id(
    *,
    headers: Any,
    body: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Resolve ``X-Experiment-Id`` / JSON ``experiment_id`` (trimmed, capped)."""
    exp: Optional[str] = None
    if headers is not None and hasattr(headers, "get"):
        exp = headers.get("X-Experiment-Id") or headers.get("X-Experiment-ID")
    if not exp and isinstance(body, dict):
        raw = body.get("experiment_id")
        if raw is not None:
            exp = str(raw)
    if not exp:
        return None
    s = str(exp).strip()
    if not s:
        return None
    return s[:200]
