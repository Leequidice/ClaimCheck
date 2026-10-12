"""Transparent DNS fallback for environments where local ISP DNS fails on cloud domains."""

from __future__ import annotations

import json
import logging
import socket
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_ORIG_GETADDRINFO = socket.getaddrinfo
_INSTALLED = False


def fallback_getaddrinfo(host: Any, port: Any, family: int = 0, type: int = 0, proto: int = 0, flags: int = 0):
    try:
        return _ORIG_GETADDRINFO(host, port, family, type, proto, flags)
    except socket.gaierror:
        # If host lookup fails locally, resolve via Google DNS over HTTPS (DoH)
        if isinstance(host, str) and not host.replace(".", "").isdigit():
            try:
                url = f"https://dns.google/resolve?name={host}&type=A"
                req = urllib.request.Request(url, headers={"User-Agent": "ClaimCheck/1.0"})
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                    for ans in payload.get("Answer", []):
                        if ans.get("type") == 1 and "data" in ans:
                            ip = ans["data"]
                            return _ORIG_GETADDRINFO(ip, port, family, type, proto, flags)
            except Exception as e:
                logger.debug(f"DNS over HTTPS fallback failed for {host}: {e}")
        raise


def install_dns_fallback(force: bool = False):
    """Install DNS over HTTPS fallback if enabled via ENABLE_DOH_FALLBACK env var or force=True."""
    global _INSTALLED
    import os

    opt_in = force or os.getenv("ENABLE_DOH_FALLBACK", "").lower() in ("true", "1", "yes")
    if opt_in and not _INSTALLED:
        socket.getaddrinfo = fallback_getaddrinfo
        _INSTALLED = True
        logger.info("DoH DNS fallback installed.")
