"""
Canonical sense-domain taxonomy for Haroma sensor channels.

Maps the classical five senses plus embodied / spatial modalities used by
``sensors/adapters.py`` and arbitrary ``POST /sensor`` channel names (aliases).

Use :func:`resolve_channel_to_domain` to attach a stable domain label to any
incoming channel string for fusion, memory tagging, or analytics.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet, Optional


class SenseDomain(str, Enum):
    """High-level sensory modality (not necessarily one physical device)."""

    # Classical five (audition = hearing)
    VISION = "vision"
    AUDITION = "audition"
    TOUCH = "touch"
    OLFACTION = "olfaction"
    GUSTATION = "gustation"

    # Beyond the five: spatial / embodied
    THERMAL_RADIANCE = "thermal_radiance"  # IR / temperature-facing sensors
    SPATIAL_RANGE = "spatial_range"  # LiDAR, ToF, depth geometry
    SPATIAL_GLOBAL = "spatial_global"  # GNSS / lat-lon
    PROPRIOCEPTION = "proprioception"  # IMU — accel/gyro; vestibular cues included

    # Aggregate HTTP / environment snapshots
    EMBODIMENT_CONTEXT = "embodiment_context"

    UNKNOWN = "unknown"


# First-class channels from SensorAdapter subclasses in adapters.py
CANONICAL_CHANNELS: FrozenSet[str] = frozenset(
    {
        "vision",
        "audio",
        "touch",
        "smell",
        "taste",
        "lidar",
        "infrared",
        "proprioception",
        "gps",
    }
)

# Normalized alias -> SenseDomain (lowercase keys only)
_ALIAS_TO_DOMAIN: Dict[str, SenseDomain] = {
    # Vision
    "vision": SenseDomain.VISION,
    "camera": SenseDomain.VISION,
    "cam": SenseDomain.VISION,
    "webcam": SenseDomain.VISION,
    "rgb": SenseDomain.VISION,
    "video": SenseDomain.VISION,
    # Audition
    "audio": SenseDomain.AUDITION,
    "sound": SenseDomain.AUDITION,
    "mic": SenseDomain.AUDITION,
    "microphone": SenseDomain.AUDITION,
    "hearing": SenseDomain.AUDITION,
    # Touch / somatosensation
    "touch": SenseDomain.TOUCH,
    "tactile": SenseDomain.TOUCH,
    "pressure": SenseDomain.TOUCH,
    "skin": SenseDomain.TOUCH,
    # Chemo — air
    "smell": SenseDomain.OLFACTION,
    "olfaction": SenseDomain.OLFACTION,
    "odor": SenseDomain.OLFACTION,
    "gas": SenseDomain.OLFACTION,
    "volatile": SenseDomain.OLFACTION,
    # Chemo — liquid
    "taste": SenseDomain.GUSTATION,
    "gustation": SenseDomain.GUSTATION,
    "chemical_liquid": SenseDomain.GUSTATION,
    # Thermal / IR
    "infrared": SenseDomain.THERMAL_RADIANCE,
    "ir": SenseDomain.THERMAL_RADIANCE,
    "thermal": SenseDomain.THERMAL_RADIANCE,
    "radiometry": SenseDomain.THERMAL_RADIANCE,
    "temperature_surface": SenseDomain.THERMAL_RADIANCE,
    # Range / geometry
    "lidar": SenseDomain.SPATIAL_RANGE,
    "depth": SenseDomain.SPATIAL_RANGE,
    "tof": SenseDomain.SPATIAL_RANGE,
    "range": SenseDomain.SPATIAL_RANGE,
    "pointcloud": SenseDomain.SPATIAL_RANGE,
    # Global position
    "gps": SenseDomain.SPATIAL_GLOBAL,
    "gnss": SenseDomain.SPATIAL_GLOBAL,
    "position": SenseDomain.SPATIAL_GLOBAL,
    "location": SenseDomain.SPATIAL_GLOBAL,
    # IMU / body motion / balance (vestibular layered under proprioception)
    "proprioception": SenseDomain.PROPRIOCEPTION,
    "imu": SenseDomain.PROPRIOCEPTION,
    "gyro": SenseDomain.PROPRIOCEPTION,
    "accelerometer": SenseDomain.PROPRIOCEPTION,
    "accel": SenseDomain.PROPRIOCEPTION,
    "vestibular": SenseDomain.PROPRIOCEPTION,
    "balance": SenseDomain.PROPRIOCEPTION,
    # Embodiment / world snapshot (see elarion_server_v2 push_sensor)
    "agent_environment": SenseDomain.EMBODIMENT_CONTEXT,
    "environment": SenseDomain.EMBODIMENT_CONTEXT,
}


def resolve_channel_to_domain(channel: Optional[object]) -> SenseDomain:
    """Map a channel name (or alias) to :class:`SenseDomain`."""
    if channel is None:
        return SenseDomain.UNKNOWN
    key = str(channel).strip().lower()
    if not key:
        return SenseDomain.UNKNOWN
    return _ALIAS_TO_DOMAIN.get(key, SenseDomain.UNKNOWN)


def domain_display_name(domain: SenseDomain) -> str:
    """Short human-readable label for logs and UI."""
    _labels = {
        SenseDomain.VISION: "Vision",
        SenseDomain.AUDITION: "Audition (hearing)",
        SenseDomain.TOUCH: "Touch",
        SenseDomain.OLFACTION: "Olfaction (smell)",
        SenseDomain.GUSTATION: "Gustation (taste)",
        SenseDomain.THERMAL_RADIANCE: "Thermal / IR",
        SenseDomain.SPATIAL_RANGE: "Spatial range (LiDAR / depth)",
        SenseDomain.SPATIAL_GLOBAL: "Spatial global (GNSS)",
        SenseDomain.PROPRIOCEPTION: "Proprioception / IMU (incl. vestibular)",
        SenseDomain.EMBODIMENT_CONTEXT: "Embodiment context",
        SenseDomain.UNKNOWN: "Unknown",
    }
    return _labels.get(domain, domain.value)


def all_sense_domains() -> tuple[SenseDomain, ...]:
    """All defined domains except UNKNOWN (for iteration / docs)."""
    return tuple(d for d in SenseDomain if d is not SenseDomain.UNKNOWN)
