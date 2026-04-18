"""Human-readable ``text_translation`` for each sensor reading (multimodal input).

Every dict in ``sensor_data[channel]`` lists gets a ``text_translation`` string so
personas see both raw fields and a unified text line alongside ``chat`` text.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def _fmt_num(x: Any, nd: int = 3) -> str:
    try:
        if isinstance(x, bool):
            return str(x)
        v = float(x)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.{nd}f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(x)[:120]


def reading_text_translation(channel: str, reading: Dict[str, Any]) -> str:
    """One-line description for a single sensor reading dict."""
    ch = str(channel or "").strip().lower()
    parts: list[str] = []

    # HTTP / unified chat sensor (same schema as user text merged into sensor_data)
    if ch == "chat":
        t = str(reading.get("text") or "").strip()
        src = str(reading.get("source") or "user").strip()
        if t:
            return f"chat ({src}): {t}"
        return "chat: (empty)"

    if ch == "vision":
        if reading.get("scene_classification"):
            sc = reading["scene_classification"]
            if isinstance(sc, dict):
                top = max(sc.items(), key=lambda kv: float(kv[1] or 0), default=(None, None))
                if top[0]:
                    parts.append(f"scene≈{top[0]} ({_fmt_num(top[1], 2)})")
            else:
                parts.append(f"scene={sc}")
        res = reading.get("resolution")
        if isinstance(res, (list, tuple)) and len(res) >= 2:
            parts.append(f"{res[0]}×{res[1]}px")
        if reading.get("brightness") is not None:
            parts.append(f"brightness {_fmt_num(reading['brightness'], 2)}")
        if reading.get("edge_density") is not None:
            parts.append(f"edges {_fmt_num(reading['edge_density'], 3)}")
        if reading.get("has_motion"):
            parts.append("motion")
        return "vision: " + ("; ".join(parts) if parts else "frame (no summary)")

    if ch == "audio":
        if reading.get("transcription"):
            tr = reading["transcription"]
            if isinstance(tr, dict) and tr.get("text"):
                parts.append(f"heard “{str(tr['text'])[:200]}”")
            elif isinstance(tr, str):
                parts.append(f"heard “{tr[:200]}”")
        if reading.get("audio_events"):
            ev = reading["audio_events"]
            if isinstance(ev, list) and ev:
                bits = []
                for e in ev[:5]:
                    if isinstance(e, dict) and e.get("label"):
                        bits.append(f"{e['label']}({_fmt_num(e.get('score', 0), 2)})")
                if bits:
                    parts.append("events: " + ", ".join(bits))
        if reading.get("rms_level") is not None:
            parts.append(f"rms {_fmt_num(reading['rms_level'], 4)}")
        if reading.get("spectral_centroid") is not None:
            parts.append(f"centroid {_fmt_num(reading['spectral_centroid'], 1)}")
        return "audio: " + ("; ".join(parts) if parts else "sample (no summary)")

    if ch == "taste":
        if reading.get("ph") is not None:
            parts.append(f"pH {_fmt_num(reading['ph'], 2)}")
        if reading.get("conductivity") is not None:
            parts.append(f"conductivity {_fmt_num(reading['conductivity'], 2)}")
        if reading.get("tds") is not None:
            parts.append(f"TDS {_fmt_num(reading['tds'], 1)}")
        return "taste: " + ("; ".join(parts) if parts else "liquid sample")

    if ch == "smell":
        for key, label in (
            ("voc_index", "VOC"),
            ("gas_resistance", "gas Ω"),
            ("humidity", "RH%"),
            ("temperature", "°C"),
        ):
            if reading.get(key) is not None:
                parts.append(f"{label} {_fmt_num(reading[key], 2)}")
        return "smell: " + ("; ".join(parts) if parts else "air sample")

    if ch == "touch":
        if reading.get("pressure") is not None:
            parts.append(f"pressure {_fmt_num(reading['pressure'], 3)}")
        for k in ("capacitance", "force", "zones"):
            if reading.get(k) is not None:
                parts.append(f"{k} {reading[k]}")
        return "touch: " + ("; ".join(parts) if parts else "contact")

    if ch == "lidar":
        if reading.get("distance_m") is not None:
            parts.append(f"distance {_fmt_num(reading['distance_m'], 3)} m")
        if reading.get("ranges") is not None:
            parts.append("scan ranges present")
        return "lidar: " + ("; ".join(parts) if parts else "range data")

    if ch == "infrared":
        if reading.get("temperature_c") is not None:
            parts.append(f"temp {_fmt_num(reading['temperature_c'], 1)} °C")
        if reading.get("thermal_grid") is not None:
            parts.append("thermal grid")
        return "infrared: " + ("; ".join(parts) if parts else "thermal")

    if ch == "gps":
        lat, lon = reading.get("lat"), reading.get("lon")
        if lat is not None and lon is not None:
            parts.append(f"lat {_fmt_num(lat, 5)}, lon {_fmt_num(lon, 5)}")
        if reading.get("alt_m") is not None:
            parts.append(f"alt {_fmt_num(reading['alt_m'], 1)} m")
        if reading.get("heading") is not None:
            parts.append(f"heading {_fmt_num(reading['heading'], 1)}°")
        return "gps: " + ("; ".join(parts) if parts else "fix")

    if ch == "proprioception":
        for axis in ("accel", "gyro", "orientation"):
            if reading.get(axis) is not None:
                parts.append(f"{axis} {reading[axis]}")
        return "proprioception: " + ("; ".join(parts) if parts else "imu")

    # Generic: stable, short JSON subset (skip huge blobs)
    slim = {k: v for k, v in reading.items() if k not in ("clip_embedding", "embedding") and not str(k).startswith("_")}
    try:
        s = json.dumps(slim, default=str, ensure_ascii=False)[:400]
    except Exception:
        s = str(slim)[:400]
    return f"{ch}: {s}"


def enrich_sensor_data(sensor_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Mutate each reading dict under ``sensor_data[channel]`` with ``text_translation``."""
    if not sensor_data:
        return sensor_data
    for ch, readings in list(sensor_data.items()):
        if not isinstance(readings, list):
            continue
        for r in readings:
            if not isinstance(r, dict):
                continue
            existing = r.get("text_translation")
            if isinstance(existing, str) and existing.strip():
                continue
            try:
                r["text_translation"] = reading_text_translation(str(ch), r)
            except Exception:
                r["text_translation"] = f"{ch}: (translation error)"
    return sensor_data


def sensor_text_translation_digest(sensor_data: Optional[Dict[str, Any]], max_chars: int = 1200) -> str:
    """Single string of non-chat translations for prompts (optional)."""
    if not sensor_data:
        return ""
    lines: list[str] = []
    for ch in sorted(sensor_data.keys(), key=str):
        for r in sensor_data[ch] or []:
            if not isinstance(r, dict):
                continue
            tt = r.get("text_translation")
            if not tt:
                continue
            if str(ch).lower() == "chat":
                continue
            lines.append(str(tt))
    out = " | ".join(lines)
    return out[:max_chars]
