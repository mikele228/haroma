"""HTTP helpers for symbolic LawEngine state and declare/revoke."""

from __future__ import annotations

from typing import Any, Callable, Dict

from flask import request

from core.cognitive_null import is_cognitive_null
from core.engine.LawEngine import LAW_SOURCE_EXTERNAL, LawEngine

_DEFAULT_SOURCE = LAW_SOURCE_EXTERNAL


def laws_snapshot(law) -> Dict[str, Any]:
    if law is None or is_cognitive_null(law):
        return {"available": False, "laws": {}, "summary": {}}
    summary = {}
    try:
        summary = law.summarize()
    except Exception:
        summary = {}
    detail: Dict[str, Any] = {}
    eng = getattr(law, "engine", None)
    raw = getattr(eng, "laws", None) if eng else None
    if isinstance(raw, dict):
        for lid, row in raw.items():
            if isinstance(row, dict):
                detail[lid] = {
                    "description": row.get("description", ""),
                    "tags": list(row.get("tags") or []),
                    "severity": row.get("severity", 1.0),
                    "source": row.get("source", LAW_SOURCE_EXTERNAL),
                }
    return {"available": True, "summary": summary, "laws": detail}


def handle_laws_request(
    law,
    json_response: Callable[[Dict[str, Any], int], Any],
) -> Any:
    """GET returns snapshot; POST body: declare | revoke."""
    if request.method == "GET":
        return json_response(laws_snapshot(law), 200)

    if law is None or is_cognitive_null(law):
        return json_response({"error": "law module unavailable"}, 503)

    data = request.get_json(silent=True) or {}
    action = (data.get("action") or "declare").lower()

    if action == "revoke":
        law_id = data.get("id") or data.get("law_id")
        if not law_id:
            return json_response({"error": "missing id for revoke"}, 400)
        try:
            law.revoke(str(law_id))
        except Exception as exc:
            return json_response({"error": str(exc)}, 500)
        return json_response({"ok": True, "revoked": str(law_id)}, 200)

    if action == "declare":
        law_id = data.get("id") or data.get("law_id")
        if not law_id:
            return json_response({"error": "missing id for declare"}, 400)
        description = str(data.get("description", ""))
        tags = data.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        elif not isinstance(tags, list):
            tags = []
        severity = float(data.get("severity", 1.0))
        src = data.get("source", _DEFAULT_SOURCE)
        try:
            src_n = LawEngine.normalize_source(str(src))
        except Exception:
            src_n = _DEFAULT_SOURCE
        try:
            law.declare(
                str(law_id),
                description,
                [str(t) for t in tags],
                severity,
                source=src_n,
            )
        except Exception as exc:
            return json_response({"error": str(exc)}, 500)
        return json_response(
            {"ok": True, "declared": str(law_id), "snapshot": laws_snapshot(law)},
            200,
        )

    return json_response({"error": f"unknown action: {action}"}, 400)
