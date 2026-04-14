"""
Robot state merged from repeated ``agent_environment`` snapshots — **layered** model
similar to typical stacks (ROS: ``robot_description`` / joint states / ``tf`` / localization):

* **hardware** — quasi-static: approximate size, limb topology, end-effectors, calibration id.
* **proprioception** — dynamic: posture, joint summary, contacts, IMU summary.
* **localization** — pose in a frame, GPS when valid, map / odometry hints.
* **scene** — optional compact world summary (objects, people, obstacles) for LLM context.
* **control** — motion / manipulation mode: controller state, grasp, gaze, velocity caps (optional).
* **safety** — interlocks: estop, protective stop, zone flags, human proximity (optional).
* **perception** — fused tracks, attention / gaze target, semantic scene label (optional).

Integrators send updates under ``extensions.robot_body`` (see schema below). Readings may be
**flat** (backward compatible) or **nested** under these layer keys.

.. code-block:: json

   {
     "schema_version": 4,
     "readings": {
       "hardware": {
         "approx_height_m": 1.65,
         "calibration_id": "arm_v2",
         "limb_roles": ["arm_l", "arm_r", "leg_l", "leg_r"],
         "end_effectors": ["gripper_l", "gripper_r"]
       },
       "proprioception": {
         "posture": "standing",
         "base_motion": "slow_walk"
       },
       "localization": {
         "frame_id": "map",
         "pose": { "x": 1.0, "y": 2.0, "theta": 0.5, "valid": true },
         "gps": { "lat": 48.85, "lon": 2.35, "fix_quality": 2, "valid": false }
       },
       "scene": {
         "obstacles_near": 2,
         "people_tracks": 1
       },
       "control": { "motion_mode": "nav2_follow", "velocity_cap_m_s": 0.6 },
       "safety": { "estop": false, "protective_stop": false, "human_in_zone": true },
       "perception": { "attention_target": "door_3", "fused_objects": [{"id": "cup"}], "semantic_scene": "kitchen" },
       "height_m": 1.65
     },
     "sensor_available": {
       "hardware.approx_height_m": true,
       "proprioception.posture": false,
       "localization.gps": false,
       "height_m": true
     }
   }

* ``sensor_available`` may use **dot paths** for nested fields, or disable a **whole subtree**
  (e.g. ``"localization": false``) so nothing under ``localization`` updates.
* If a sensor is unavailable, **last known** subtree/leaf is kept.
* If **no** measurement was ever accepted, ``body_defined`` is false (undefined physique).

Env ``HAROMA_ROBOT_FIELD_MAX_AGE_SEC`` (default ``0``): if ``>0``, merged output lists
``stale_fields`` where ``now - last_update > max_age`` (value retained, still marked stale).

Env ``HAROMA_ROBOT_BODY_SANITY=1`` (default): clamps ``height_m`` / ``*.approx_height_m`` to ``[0.05, 5.0]``.
"""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

LAYER_KEYS = (
    "hardware",
    "proprioception",
    "localization",
    "scene",
    "control",
    "safety",
    "perception",
)


def robot_body_prompt_block(
    robot_body: Optional[Dict[str, Any]],
    *,
    max_chars: int = 4000,
) -> str:
    """Compact JSON for packed LLM ``[ROBOT BODY STATE]`` (from ``extensions.robot_body``)."""
    if not isinstance(robot_body, dict) or not robot_body:
        return ""
    slim: Dict[str, Any] = {}
    for k in (
        "architecture",
        "schema_version",
        "body_defined",
        "layers_present",
        "cognitive_summary",
        "coordination_hint",
        "operational_mode",
        "risk_posture",
        "perception_digest",
        "interpretation",
        "field_stale_max_age_sec",
    ):
        if k in robot_body:
            slim[k] = robot_body[k]
    r = robot_body.get("readings")
    if isinstance(r, dict) and r:
        slim["readings"] = r
    sf = robot_body.get("stale_fields")
    if isinstance(sf, list) and sf:
        slim["stale_fields"] = sf[:96]
    try:
        txt = json.dumps(slim, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        txt = "{}"
    if len(txt) > max_chars:
        txt = txt[: max_chars - 24] + "\n…[truncated]"
    return txt


def _max_field_age_sec() -> float:
    raw = str(os.environ.get("HAROMA_ROBOT_FIELD_MAX_AGE_SEC", "") or "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _is_empty(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and not str(val).strip():
        return True
    return False


def _sensor_ok(avail: Dict[str, Any], path: str, short_key: str) -> bool:
    if path in avail:
        return bool(avail[path])
    if short_key in avail:
        return bool(avail[short_key])
    for layer in LAYER_KEYS:
        if path == layer or path.startswith(layer + "."):
            if layer in avail and not bool(avail[layer]):
                return False
    return True


def _clamp_height(val: Any) -> Any:
    if str(os.environ.get("HAROMA_ROBOT_BODY_SANITY", "1") or "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return val
    try:
        f = float(val)
    except (TypeError, ValueError):
        return val
    if 0.05 <= f <= 5.0:
        return round(f, 4)
    return val


def _maybe_clamp(path: str, val: Any) -> Any:
    pl = path.lower()
    if pl.endswith("height_m") or pl.endswith("approx_height_m"):
        return _clamp_height(val)
    return val


def merge_robot_body_readings(
    last_effective: Dict[str, Any],
    ever_observed: bool,
    incoming_block: Optional[Dict[str, Any]],
    *,
    now: Optional[float] = None,
    last_stamps: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, Any], bool, Dict[str, float]]:
    """Merge ``extensions.robot_body`` into nested-or-flat state + per-leaf timestamps."""
    t = time.time() if now is None else float(now)
    stamps: Dict[str, float] = dict(last_stamps) if last_stamps else {}
    out: Dict[str, Any] = deepcopy(last_effective) if last_effective else {}
    ev = ever_observed

    if not isinstance(incoming_block, dict) or not incoming_block:
        return out, ev, stamps

    readings = incoming_block.get("readings")
    if not isinstance(readings, dict):
        readings = {}

    avail = incoming_block.get("sensor_available")
    if not isinstance(avail, dict):
        avail = {}

    def merge_value(path: str, key: str, last_val: Any, inc_val: Any) -> Tuple[Any, bool]:
        nonlocal ev
        if not _sensor_ok(avail, path, key):
            return last_val, False
        if isinstance(inc_val, dict):
            lv = last_val if isinstance(last_val, dict) else {}
            changed = False
            acc: Dict[str, Any] = deepcopy(lv)
            for ck, cv in inc_val.items():
                cp = f"{path}.{ck}" if path else str(ck)
                nv, ch = merge_value(cp, str(ck), acc.get(ck), cv)
                if ch:
                    acc[ck] = nv
                    changed = True
            if changed:
                ev = True
            return acc, changed
        if _is_empty(inc_val):
            return last_val, False
        nv = _maybe_clamp(path, inc_val)
        if last_val != nv:
            stamps[path] = t
            ev = True
            return nv, True
        return last_val, False

    for key, raw in readings.items():
        sk = str(key)
        path = sk
        merged, _did = merge_value(path, sk, out.get(sk), raw)
        if _did or (isinstance(merged, dict) and merged):
            out[sk] = merged

    return out, ev, stamps


def _collect_leaf_paths(obj: Any, prefix: str = "") -> List[str]:
    paths: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                paths.extend(_collect_leaf_paths(v, p))
            else:
                paths.append(p)
    return paths


def _coordination_hint(readings: Dict[str, Any]) -> str:
    """Short hint for planners: primary TF frame + pose validity."""
    loc = readings.get("localization")
    if not isinstance(loc, dict):
        return ""
    parts: List[str] = []
    fid = str(loc.get("frame_id") or "").strip()
    if fid:
        parts.append(f"frame={fid}")
    pose = loc.get("pose")
    if isinstance(pose, dict):
        parts.append("pose_valid" if pose.get("valid") else "pose_invalid")
    return " | ".join(parts)[:200]


def _operational_mode(readings: Dict[str, Any]) -> str:
    """High-level run mode from control.* or safety fault."""
    c = readings.get("control")
    if isinstance(c, dict):
        for k in ("operational_mode", "mode", "mission_state", "behavior"):
            v = c.get(k)
            if v is not None and str(v).strip():
                return str(v).strip().lower()[:48]
    s = readings.get("safety")
    if isinstance(s, dict) and s.get("estop") is True:
        return "fault_estop"
    return "nominal"


def _risk_posture(readings: Dict[str, Any]) -> str:
    """Coarse risk for planners / LLM (not a replacement for full safety layer)."""
    s = readings.get("safety")
    if not isinstance(s, dict):
        return "normal"
    if s.get("estop") is True:
        return "critical"
    if s.get("protective_stop") is True:
        return "high"
    if s.get("human_in_zone") is True:
        return "elevated"
    return "normal"


def _perception_digest(readings: Dict[str, Any]) -> str:
    """One-line perception summary for extension JSON."""
    p = readings.get("perception")
    if not isinstance(p, dict):
        return ""
    parts: List[str] = []
    for k in ("attention_target", "gaze_target", "focus"):
        v = p.get(k)
        if v is not None and str(v).strip():
            parts.append(f"attn={str(v)[:40]}")
            break
    fo = p.get("fused_objects")
    if isinstance(fo, list) and fo:
        parts.append(f"obj≈{len(fo)}")
    tr = p.get("tracks")
    if isinstance(tr, list) and tr:
        parts.append(f"trk≈{len(tr)}")
    sem = p.get("semantic_scene")
    if isinstance(sem, str) and sem.strip():
        parts.append(f"sem={sem.strip()[:32]}")
    return " | ".join(parts)[:220]


def _derive_cognitive_summary(
    readings: Dict[str, Any],
    layers_present: Dict[str, bool],
    body_defined: bool,
) -> str:
    """Dense one-line summary for packed LLM (not a substitute for full readings)."""
    if not body_defined:
        return "No embodiment data; morphology unknown."
    parts: List[str] = []
    parts.append(f"mode={_operational_mode(readings)}|risk={_risk_posture(readings)}")
    active = [k for k, v in layers_present.items() if v]
    if active:
        parts.append("layers:" + "+".join(active[:12]))
    prop = readings.get("proprioception")
    if isinstance(prop, dict):
        for key in ("posture", "base_motion", "gait", "manipulator_mode"):
            v = prop.get(key)
            if v is not None and str(v).strip():
                parts.append(f"{key}={str(v)[:48]}")
                break
    else:
        for key in ("posture", "base_motion"):
            v = readings.get(key)
            if v is not None and str(v).strip():
                parts.append(f"{key}={str(v)[:48]}")
                break
    hw = readings.get("hardware")
    if isinstance(hw, dict):
        ee = hw.get("end_effectors")
        if isinstance(ee, list) and ee:
            parts.append(f"eef_count={len(ee)}")
    loc = readings.get("localization")
    if isinstance(loc, dict):
        pose = loc.get("pose")
        if isinstance(pose, dict):
            if pose.get("valid"):
                parts.append("localized_ok")
            try:
                pc = pose.get("confidence")
                if pc is not None:
                    pf = max(0.0, min(1.0, float(pc)))
                    parts.append(f"pose_conf≈{pf:.2f}")
            except (TypeError, ValueError):
                pass
    per = readings.get("perception")
    if isinstance(per, dict):
        for k in ("attention_target", "gaze_target"):
            v = per.get(k)
            if v is not None and str(v).strip():
                parts.append(f"attn={str(v)[:40]}")
                break
        fo = per.get("fused_objects")
        if isinstance(fo, list) and fo:
            parts.append(f"fuse≈{len(fo)}")
        sem = per.get("semantic_scene")
        if isinstance(sem, str) and sem.strip():
            parts.append(f"sem={sem.strip()[:32]}")
    scene = readings.get("scene")
    if isinstance(scene, dict):
        on = scene.get("obstructions_near", scene.get("obstacles_near"))
        pt = scene.get("people_tracks")
        if on is not None or pt is not None:
            parts.append(f"scene(obs≈{on},people≈{pt})")
    ctrl = readings.get("control")
    if isinstance(ctrl, dict):
        mm = str(ctrl.get("motion_mode") or ctrl.get("controller_state") or "").strip()
        if mm:
            parts.append(f"ctrl={mm[:40]}")
    safe = readings.get("safety")
    if isinstance(safe, dict):
        if safe.get("estop") is True:
            parts.append("ESTOP")
        elif safe.get("protective_stop") is True:
            parts.append("protective_stop")
        if safe.get("human_in_zone") is True:
            parts.append("human_in_zone")
    s = " | ".join(parts) if parts else "Embodiment present; see readings JSON."
    return s[:720]


def _stale_paths(
    stamps: Dict[str, float],
    effective: Dict[str, Any],
    now: float,
    max_age: float,
) -> List[str]:
    if max_age <= 0:
        return []
    stale: List[str] = []
    paths = set(stamps.keys()) | set(_collect_leaf_paths(effective))
    for p in paths:
        ts = stamps.get(p)
        if ts is None:
            continue
        if now - ts > max_age:
            stale.append(p)
    return sorted(stale)


def build_robot_body_extension(
    effective_readings: Dict[str, Any],
    ever_observed: bool,
    field_stamps: Optional[Dict[str, float]] = None,
    *,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Payload for ``agent_environment.extensions.robot_body`` (cognition / LLM)."""
    t = time.time() if now is None else float(now)
    stamps = field_stamps or {}
    body_defined = bool(ever_observed and effective_readings)
    max_age = _max_field_age_sec()

    layers: Dict[str, bool] = {}
    for lk in LAYER_KEYS:
        layers[lk] = lk in effective_readings and isinstance(effective_readings.get(lk), dict)

    ext: Dict[str, Any] = {
        "architecture": "layered_v4",
        "schema_version": 4,
        "body_defined": body_defined,
        "readings": dict(effective_readings) if effective_readings else {},
        "layers_present": layers,
        "field_stale_max_age_sec": max_age if max_age > 0 else None,
    }
    if body_defined:
        er = dict(effective_readings) if effective_readings else {}
        ext["cognitive_summary"] = _derive_cognitive_summary(er, layers, body_defined)
        ext["operational_mode"] = _operational_mode(er)
        ext["risk_posture"] = _risk_posture(er)
        pd = _perception_digest(er)
        if pd:
            ext["perception_digest"] = pd
        ch = _coordination_hint(er)
        if ch:
            ext["coordination_hint"] = ch
    if max_age > 0 and body_defined:
        ext["stale_fields"] = _stale_paths(stamps, effective_readings, t, max_age)
    else:
        ext["stale_fields"] = []

    if stamps:
        ext["field_updated_epoch"] = {k: stamps[k] for k in sorted(stamps.keys())[:96]}

    if not body_defined:
        ext["interpretation"] = (
            "No physical profile has ever been inferred from sensors; "
            "all embodiment layers (including perception, control, safety) are undefined."
        )
    else:
        ext["interpretation"] = (
            "Layered embodiment v4: hardware, proprioception, localization, scene, control, safety, perception. "
            "operational_mode / risk_posture summarize run state; cognitive_summary and perception_digest aid the LLM. "
            "Unavailable sensors retain last known values; stale_fields marks outdated paths when max-age is set."
        )
    return ext
