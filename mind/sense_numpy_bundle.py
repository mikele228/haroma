"""Canonical NumPy vectors for multimodal input (one array per sense bucket).

Each key is a :data:`SENSE_NUMPY_KEYS` entry: ``float32`` 1-D arrays, **empty**
(``shape == (0,)``) when that modality had no usable numeric samples this turn.
``text_embedding`` holds the language encoder vector when present, else empty.

Channel names from ``sensor_data`` (including POST ``/chat`` ``sensor_data``) are
mapped via :func:`sensors.domains.resolve_channel_to_domain` into the nine
first-class buckets; unmapped readings go to ``other``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sensors.domains import SenseDomain, resolve_channel_to_domain

__all__ = [
    "SENSE_NUMPY_KEYS",
    "build_senses_numpy_bundle",
]

# Matches ``sensors.domains.CANONICAL_CHANNELS`` — fixed order for stable APIs.
SENSE_NUMPY_ORDER: Tuple[str, ...] = (
    "vision",
    "audio",
    "touch",
    "smell",
    "taste",
    "lidar",
    "infrared",
    "proprioception",
    "gps",
)

SENSE_NUMPY_KEYS: Tuple[str, ...] = SENSE_NUMPY_ORDER + ("other", "text_embedding")

_PER_BUCKET_CAP = 8192


def _domain_to_bucket(domain: SenseDomain) -> str:
    if domain is SenseDomain.UNKNOWN:
        return "other"
    if domain is SenseDomain.EMBODIMENT_CONTEXT:
        return "other"
    m: Dict[SenseDomain, str] = {
        SenseDomain.VISION: "vision",
        SenseDomain.AUDITION: "audio",
        SenseDomain.TOUCH: "touch",
        SenseDomain.OLFACTION: "smell",
        SenseDomain.GUSTATION: "taste",
        SenseDomain.THERMAL_RADIANCE: "infrared",
        SenseDomain.SPATIAL_RANGE: "lidar",
        SenseDomain.SPATIAL_GLOBAL: "gps",
        SenseDomain.PROPRIOCEPTION: "proprioception",
    }
    return m.get(domain, "other")


def _collect_floats(x: Any, out: List[float], cap: int) -> None:
    if len(out) >= cap:
        return
    if x is None:
        return
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        out.append(float(x))
        return
    if isinstance(x, np.ndarray):
        flat = x.astype(np.float64, copy=False).ravel()
        for v in flat:
            if len(out) >= cap:
                return
            out.append(float(v))
        return
    if isinstance(x, (list, tuple)):
        for v in x:
            _collect_floats(v, out, cap)
        return
    if isinstance(x, dict):
        preferred = (
            "ranges",
            "range",
            "embedding",
            "values",
            "data",
            "vector",
            "lat",
            "lon",
            "alt",
            "heading",
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
        )
        for k in preferred:
            if k in x:
                _collect_floats(x[k], out, cap)
                if len(out) >= cap:
                    return
        for k, v in x.items():
            if k in preferred:
                continue
            _collect_floats(v, out, cap)
            if len(out) >= cap:
                return
        return
    if isinstance(x, (bytes, bytearray, memoryview)):
        try:
            arr = np.frombuffer(x, dtype=np.uint8)
            for v in arr.flat[: min(len(arr), cap - len(out))]:
                out.append(float(v) / 255.0)
        except Exception:
            return


def _readings_to_floats(readings: Any, cap: int) -> List[float]:
    out: List[float] = []
    if readings is None:
        return out
    if isinstance(readings, list):
        for r in readings:
            _collect_floats(r, out, cap)
            if len(out) >= cap:
                break
    else:
        _collect_floats(readings, out, cap)
    return out


def build_senses_numpy_bundle(
    sensor_data: Optional[Dict[str, Any]],
    *,
    text_embedding: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Build ``dict`` of sense name → ``float32`` 1-D array (empty if unavailable)."""
    buckets: Dict[str, List[float]] = {k: [] for k in SENSE_NUMPY_ORDER}
    buckets["other"] = []

    if isinstance(sensor_data, dict):
        for ch, readings in sensor_data.items():
            ch_name = str(ch).strip().lower()
            if ch_name in ("", "chat"):
                continue
            dom = resolve_channel_to_domain(ch_name)
            bucket = _domain_to_bucket(dom)
            tgt = buckets.setdefault(bucket, [])
            chunk = _readings_to_floats(readings, _PER_BUCKET_CAP - len(tgt))
            for v in chunk:
                if len(tgt) >= _PER_BUCKET_CAP:
                    break
                tgt.append(v)

    out: Dict[str, np.ndarray] = {}
    for k in SENSE_NUMPY_ORDER:
        vals = buckets.get(k) or []
        out[k] = np.asarray(vals, dtype=np.float32) if vals else np.empty(0, dtype=np.float32)
    oth = buckets.get("other") or []
    out["other"] = np.asarray(oth, dtype=np.float32) if oth else np.empty(0, dtype=np.float32)

    if text_embedding is not None:
        try:
            te = np.asarray(text_embedding, dtype=np.float32).ravel()
        except Exception:
            te = np.empty(0, dtype=np.float32)
    else:
        te = np.empty(0, dtype=np.float32)
    out["text_embedding"] = te

    return out
