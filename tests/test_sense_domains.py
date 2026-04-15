"""Tests for sensors.domains taxonomy and channel resolution."""

from sensors.domains import (
    CANONICAL_CHANNELS,
    SenseDomain,
    all_sense_domains,
    domain_display_name,
    resolve_channel_to_domain,
)


def test_canonical_channels_cover_adapters():
    assert "vision" in CANONICAL_CHANNELS
    assert "audio" in CANONICAL_CHANNELS
    assert "proprioception" in CANONICAL_CHANNELS


def test_resolve_aliases():
    assert resolve_channel_to_domain("cam") is SenseDomain.VISION
    assert resolve_channel_to_domain("MIC") is SenseDomain.AUDITION
    assert resolve_channel_to_domain("imu") is SenseDomain.PROPRIOCEPTION
    assert resolve_channel_to_domain("vestibular") is SenseDomain.PROPRIOCEPTION
    assert resolve_channel_to_domain("lidar") is SenseDomain.SPATIAL_RANGE
    assert resolve_channel_to_domain("gps") is SenseDomain.SPATIAL_GLOBAL
    assert resolve_channel_to_domain("agent_environment") is SenseDomain.EMBODIMENT_CONTEXT


def test_unknown_channel():
    assert resolve_channel_to_domain("custom_vendor_xyz") is SenseDomain.UNKNOWN
    assert resolve_channel_to_domain(None) is SenseDomain.UNKNOWN
    assert resolve_channel_to_domain("") is SenseDomain.UNKNOWN


def test_all_sense_domains_excludes_unknown():
    assert SenseDomain.UNKNOWN not in all_sense_domains()
    assert SenseDomain.VISION in all_sense_domains()


def test_domain_display_name_smoke():
    assert "Vision" in domain_display_name(SenseDomain.VISION)
    assert domain_display_name(SenseDomain.UNKNOWN) == "Unknown"
