"""Derive myth, value reinforcement, concept fusion, and dream symbols from
perception when callers do not pass explicit ``value`` / ``myth`` / ``fusion_targets``.
Keeps the cognitive loop from leaving those managers idle on typical turns.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from core.cognitive_null import is_cognitive_null

_SIDECAR_DEBUG = os.environ.get("ELARION_SIDECAR_DEBUG", "0") == "1"


def _sc_warn(label: str, exc: Exception) -> None:
    if _SIDECAR_DEBUG:
        print(f"  [sidecar] {label}: {type(exc).__name__}: {exc}", flush=True)


def _module_real(obj) -> bool:
    return obj is not None and not is_cognitive_null(obj)


def apply_derived_symbolic_sidecar(
    *,
    value,
    myth,
    fusion,
    dream_mgr,
    episode,
    symbolic_input: Dict[str, Any],
    explicit: Dict[str, Any],
    role: str,
    has_external: bool,
    skip_derived_value_myth_fusion: bool = False,
) -> None:
    tags: List[str] = [str(t) for t in (symbolic_input.get("tags") or []) if t is not None]

    if _module_real(dream_mgr) and tags:
        try:
            dream_mgr.fuse_symbols(tags[:12])
        except Exception as _e:
            _sc_warn("dream_mgr.fuse_symbols", _e)

    if skip_derived_value_myth_fusion or not has_external:
        return

    if _module_real(value) and "value" not in explicit:
        dom = episode.affect.get("dominant_emotion", "neutral")
        inten = float(episode.affect.get("intensity", 0.0))
        if dom and dom != "neutral" and inten > 0.35:
            try:
                value.reinforce_value(
                    f"affect::{dom}",
                    weight=round(0.1 + min(0.35, inten * 0.15), 3),
                )
            except Exception as _e:
                _sc_warn("value.reinforce_value", _e)

    if _module_real(myth) and "myth" not in explicit and len(tags) >= 2:
        narr = (episode.narrative_context or "").strip()[:96]
        t1, t2 = tags[0], tags[1]
        myth_text = f"themes:{t1}×{t2}"
        if narr:
            myth_text = f"{myth_text} | {narr}"
        try:
            myth.bind(
                myth_text,
                anchor=role,
                meta={"derived": True, "tags": tags[:5]},
            )
        except Exception as _e:
            _sc_warn("myth.bind", _e)

    if _module_real(fusion) and "fusion_targets" not in explicit and len(tags) >= 2:
        try:
            fusion.fuse(
                {"concept": tags[0], "layer": "perception", "role": role},
                {"concept": tags[1], "layer": "perception", "role": role},
                mode="merge",
            )
        except Exception as _e:
            _sc_warn("fusion.fuse", _e)
