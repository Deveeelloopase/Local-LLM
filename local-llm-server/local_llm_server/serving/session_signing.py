"""session_signing.py — sign and verify session IDs bound to an API key.

sessions were previously bare UUIDs, stored in a cookie
or supplied via the ``X-Session-ID`` header, with no binding to the API key
that created them. Any authenticated caller who learned another caller's
session_id could read and continue that conversation by sending the same
value — no ownership check existed at any point.

This module fixes that by making the session a *capability*: the token
contains the session_id and an HMAC over ``(session_id || ":" || key_id)``.
A token issued to key A cannot be used by key B, and cannot be forged
without the server's signing secret.

Wire format
-----------
    <urlsafe_b64(session_id)> "." <urlsafe_b64(hmac_sha256(secret, sid || ":" || key_id))[:32]>

The two parts are joined by a single ``.``. Both are base64url-encoded so
the whole token is cookie-safe and header-safe.

The signing secret is derived from ``api.master_key`` (which
``startup_validation`` already requires to be present). Deriving via HMAC
(rather than using the master key directly) means the two usages are
cryptographically separated — leaking a session token cannot be used to
recover the master key, and rotating the master key invalidates all
sessions (an acceptable, expected consequence).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import re
from typing import Optional
from urllib.parse import quote as _urlquote
from urllib.parse import unquote as _urlunquote

log = logging.getLogger(__name__)


# A signed token is exactly two base64url chunks separated by one dot.
# Anything that does not match this shape is either a bare UUID (legacy
# client) or malformed — both are rejected by verify_session().
_SIGNED_RE = re.compile(r"^[A-Za-z0-9_\-]+=*\.[A-Za-z0-9_\-]+=*$")


def _b64u_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64u_decode(s: str) -> bytes:
    # Restore padding
    pad = (-len(s)) % 4
    return base64.urlsafe_b64decode(s + ("=" * pad))


def _session_secret(master_key: str) -> bytes:
    """HKDF-Extract style: HMAC(master_key, info) → 32-byte secret.

    Deliberately NOT the master key itself. Even if a session token leaks,
    an attacker cannot invert the HMAC to recover the master key. If the
    master key is rotated, all existing sessions become invalid (which is
    the desired behaviour for a key rotation).
    """
    return hmac.new(
        master_key.encode("utf-8"),
        b"local-llm-session-v1",
        hashlib.sha256,
    ).digest()


def _session_mac(secret: bytes, session_id: str, key_id: str) -> bytes:
    """HMAC-SHA256 over (session_id || ':' || key_id), truncated to 32 bytes.

    Truncation is safe here: 256 bits of HMAC output, using the first 32
    bytes = 256 bits. (We truncate for cookie compactness; this is not a
    security reduction.)
    """
    msg = f"{session_id}:{key_id}".encode("utf-8")
    return hmac.new(secret, msg, hashlib.sha256).digest()[:32]


def sign_session(session_id: str, key_id: str, secret: bytes) -> str:
    """Return the signed token for ``session_id`` bound to ``key_id``."""
    sid_b = _b64u_encode(session_id.encode("utf-8"))
    mac_b = _b64u_encode(_session_mac(secret, session_id, key_id))
    return f"{sid_b}.{mac_b}"


def verify_session(token: str, key_id: str, secret: bytes) -> Optional[str]:
    """Verify ``token`` against ``key_id`` and return the session_id.

    Returns None if:
      * the token is empty
      * the token is not in the signed format (legacy bare UUID → rejected
        so a bare session_id cannot be forged across keys)
      * the MAC does not match
      * the MAC is not valid for the *current* key_id
    """
    if not token or not isinstance(token, str):
        return None
    token = token.strip()
    if not _SIGNED_RE.match(token):
        # Bare session_id or malformed — reject. Legacy unsigned headers
        # are explicitly ignored by the caller (get_session_id) so this
        # does not fall back to "trust the raw value".
        return None
    try:
        sid_part, mac_part = token.split(".", 1)
        session_id = _b64u_decode(sid_part).decode("utf-8")
        provided_mac = _b64u_decode(mac_part)
    except (ValueError, UnicodeDecodeError, base64.binascii.Error):
        return None
    expected_mac = _session_mac(secret, session_id, key_id)
    if not hmac.compare_digest(provided_mac, expected_mac):
        return None
    return session_id


def is_secure_request(request) -> bool:
    """Decide whether the ``Secure`` cookie flag should be set.

    Returns True when:
      * the request itself arrived over HTTPS, or
      * the request came from loopback AND the reverse proxy set
        ``X-Forwarded-Proto: https`` (typical local-deployment pattern:
        nginx/Caddy terminates TLS, forwards to 127.0.0.1).

    Returns False otherwise. Fixes a pre-existing bug where the cookie was
    always sent with ``Secure=True``, meaning plain-HTTP deployments never
    received it back — every request appeared to start a fresh session.
    """
    try:
        if request.url.scheme == "https":
            return True
        client_host = request.client.host if request.client else ""
        if client_host in ("127.0.0.1", "::1", "localhost"):
            proto = (request.headers.get("X-Forwarded-Proto") or "").lower().strip()
            if proto == "https":
                return True
    except Exception:
        pass
    return False


__all__ = [
    "sign_session",
    "verify_session",
    "is_secure_request",
    "_session_secret",  # exposed for api_server.get_session_id
]
