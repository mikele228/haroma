"""
Effective client IP for rate limiting and access logs behind reverse proxies.

Without configuration, :func:`get_effective_client_ip` matches Werkzeug's
``request.remote_addr`` (direct TCP peer).

When ``HAROMA_HTTP_USE_X_FORWARDED_FOR`` is enabled, the **leftmost** address in
``X-Forwarded-For`` is used **only if** the direct peer is trusted (loopback
and/or ``HAROMA_HTTP_TRUSTED_PROXIES``). This prevents arbitrary clients from
spoofing the header when Haroma is exposed on a routable interface.

See ``docs/production-hardening.md``.
"""

from __future__ import annotations

import ipaddress
import os
import re
from typing import Any, List, Optional, Tuple, Union

_USE_XFF = ("1", "true", "yes", "on")

_Net = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]

# Cached parsed networks (immutable after first parse)
_trusted_nets: Optional[Tuple[_Net, ...]] = None


def _env_use_xff() -> bool:
    return str(os.environ.get("HAROMA_HTTP_USE_X_FORWARDED_FOR", "") or "").strip().lower() in _USE_XFF


def _parse_trusted_proxy_env() -> Tuple[_Net, ...]:
    global _trusted_nets
    if _trusted_nets is not None:
        return _trusted_nets
    raw = str(os.environ.get("HAROMA_HTTP_TRUSTED_PROXIES", "") or "").strip()
    nets: List[_Net] = []
    if raw:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                if "/" in part:
                    nets.append(ipaddress.ip_network(part, strict=False))
                else:
                    host = ipaddress.ip_address(part)
                    if isinstance(host, ipaddress.IPv4Address):
                        nets.append(ipaddress.ip_network(f"{host}/32"))
                    else:
                        nets.append(ipaddress.ip_network(f"{host}/128"))
            except ValueError:
                continue
    _trusted_nets = tuple(nets)
    return _trusted_nets


def _peer_is_loopback(addr: str) -> bool:
    if not addr:
        return False
    try:
        a = ipaddress.ip_address(addr.split("%")[0])
    except ValueError:
        return False
    return bool(a.is_loopback)


def _peer_in_trusted_nets(addr: str, nets: Tuple[_Net, ...]) -> bool:
    if not addr or not nets:
        return False
    try:
        a = ipaddress.ip_address(addr.split("%")[0])
    except ValueError:
        return False
    for n in nets:
        if a in n:
            return True
    return False


_IP_RE = re.compile(r"^[0-9a-fA-F:.]+$")


def _first_valid_xff_ip(xff: str) -> Optional[str]:
    """Return the first (leftmost) plausible client IP from X-Forwarded-For."""
    if not xff or not isinstance(xff, str):
        return None
    for part in xff.split(","):
        token = part.strip()
        if not token:
            continue
        # Strip port if present (uncommon in XFF but defensive)
        if token.startswith("["):
            end = token.find("]")
            if end > 0:
                token = token[1:end]
        elif token.count(":") > 1:
            pass  # IPv6 as-is (no port form without brackets)
        else:
            if ":" in token and not token.startswith("::"):
                # Could be IPv4:port
                if token.rfind(":") == token.find(":"):
                    host, _, maybe_port = token.rpartition(":")
                    if maybe_port.isdigit():
                        token = host
        if not _IP_RE.match(token):
            continue
        try:
            ipaddress.ip_address(token.split("%")[0])
        except ValueError:
            continue
        return token.split("%")[0]
    return None


def direct_peer_trusted_for_xff(remote_addr: str) -> bool:
    """Whether we may read ``X-Forwarded-For`` for this direct peer."""
    if _peer_is_loopback(remote_addr):
        return True
    nets = _parse_trusted_proxy_env()
    return _peer_in_trusted_nets(remote_addr, nets)


def get_effective_client_ip(request: Any) -> str:
    """
    Client IP for rate limiting and structured logs.

    * Default: ``request.remote_addr`` (or ``unknown``).
    * If ``HAROMA_HTTP_USE_X_FORWARDED_FOR`` is set **and** the direct peer is
      trusted, returns the first valid IP from ``X-Forwarded-For`` when present.
    """
    remote = getattr(request, "remote_addr", None) or ""
    if not _env_use_xff():
        return remote if remote else "unknown"
    if not direct_peer_trusted_for_xff(remote):
        return remote if remote else "unknown"
    xff = ""
    try:
        hdrs = getattr(request, "headers", None) or {}
        xff = str(hdrs.get("X-Forwarded-For", "") or "")
    except Exception:
        xff = ""
    first = _first_valid_xff_ip(xff)
    if first:
        return first
    return remote if remote else "unknown"


def clear_trusted_proxy_cache_for_tests() -> None:
    """Reset parsed CIDR cache (tests only)."""
    global _trusted_nets
    _trusted_nets = None
