"""Haroma sensor package: hardware adapters (:mod:`sensors.adapters`) and sense-domain taxonomy (:mod:`sensors.domains`)."""

from sensors.domains import (
    CANONICAL_CHANNELS,
    SenseDomain,
    all_sense_domains,
    domain_display_name,
    resolve_channel_to_domain,
)

__all__ = [
    "CANONICAL_CHANNELS",
    "SenseDomain",
    "all_sense_domains",
    "domain_display_name",
    "resolve_channel_to_domain",
]
