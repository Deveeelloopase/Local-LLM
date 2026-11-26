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

``require_signed_sessions: false`` reopens the cross-key session-hijacking
risk that the signing mechanism exists to close (plan §S7): any authenticated
caller who learns another caller's session_id can use it. Set to ``false``
ONLY in single-tenant deployments where every caller shares one API key.
``api_server.startup_validation`` logs a WARNING if it is ``false``
AND ``key_store_enabled: true`` (multi-key) — the default config here has
``require_signed_sessions: true`` and no warning is emitted.

version = 1.0.0
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import re
from typing import Optional, Tuple
from urllib.parse import quote as _urlquote
from urllib.parse import unquote as _urlunquote

log = logging.getLogger(__name__)


# A signed token is exactly two base64url chunks separated by one dot.
# Anything that does not match this shape is either a bare UUID (legacy
# client) or malformed — both are rejected by verify_session().
_SIGNED_RE = re.compile(r"^[A-Za-z0-9_\-]+=*\.[A-Za-z0-9_\-]+=*$")

# Raw (unsigned) session id shape. Matches:
#   - web UI's "c<timestamp>" (14 chars, starts with 'c')
#   - server-generated UUID4 without dashes (32 hex chars)
#   - anything reasonably UUID-shaped or slug-shaped
# Rejects: path traversal (../), newlines (\r\n header injection), null
# bytes, spaces, empty, oversized (>128), and anything with unusual chars.
# 8 is a small enough floor that the web UI's "c<13-digit-ts>" always fits.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")

# A signed token is ~60-100 chars. 512 gives 5x headroom for future
# format changes without opening a DoS hole. Enforced in verify_session
# BEFORE the regex scan so a 10 MB header cannot force O(n) regex work.
_MAX_SIGNED_TOKEN_LEN = 512


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


def _is_valid_session_id(sid: str) -> bool:
    """Validate the shape of a raw (unsigned) session id.

    Accepts ``[A-Za-z0-9_-]{8,128}``. Used in two places:

      1. ``resolve_session()`` before accepting an unsigned header value
         (prevents path traversal, header injection, oversized input
         from becoming a session id when ``require_signed_sessions: false``).

      2. ``verify_session()`` on the DECODED session_id after HMAC
         verification (defense in depth: a token whose MAC happens to be
         valid for a malicious sid still gets rejected).

    Legitimate ids:
      * ``c1731000000000`` — web UI (14 chars, matches)
      * ``5f9e1c0b-a2d4-4e8a-9b6c-8d7e6f5a4b3c`` — UUID with dashes (36 chars, matches)
      * ``5f9e1c0ba2d44e8a9b6c8d7e6f5a4b3c`` — UUID stripped (32 chars, matches)

    Rejected:
      * ``../../etc/passwd`` — contains ``.`` and ``/`` → regex rejects
      * ``abc`` — too short (< 8) → regex rejects
      * ``<500 chars>`` — too long (> 128) → regex rejects
      * ``  `` — empty after strip → regex rejects
      * ``"abc\r\nX-Evil: 1"`` — header injection → regex rejects
    """
    return bool(_SESSION_ID_RE.match(sid or ""))


def sign_session(session_id: str, key_id: str, secret: bytes) -> str:
    """Return the signed token for ``session_id`` bound to ``key_id``."""
    if not _is_valid_session_id(session_id):
        raise ValueError(
            f"sign_session: invalid session_id shape {session_id!r} "
            f"(must match [A-Za-z0-9_-]{{8,128}})"
        )
    sid_b = _b64u_encode(session_id.encode("utf-8"))
    mac_b = _b64u_encode(_session_mac(secret, session_id, key_id))
    return f"{sid_b}.{mac_b}"


def verify_session(token: str, key_id: str, secret: bytes) -> Optional[str]:
    """Verify ``token`` against ``key_id`` and return the session_id.

    Returns None if:
      * the token is empty
      * the token exceeds ``_MAX_SIGNED_TOKEN_LEN`` (DoS guard)
      * the token is not in the signed format (legacy bare UUID → rejected
        so a bare session_id cannot be forged across keys)
      * the MAC does not match
      * the MAC is not valid for the *current* key_id
      * the decoded session_id fails ``_is_valid_session_id`` (defense in
        depth: MACs are unforgeable, but a *valid* MAC over a malicious
        sid should still be rejected — the secret could leak, and we
        would rather fail closed)
    """
    if not token or not isinstance(token, str):
        return None
    token = token.strip()
    if len(token) > _MAX_SIGNED_TOKEN_LEN:
        log.warning(
            "verify_session: token too long (%d > %d) — rejected",
            len(token), _MAX_SIGNED_TOKEN_LEN,
        )
        return None
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
    # Defense in depth: reject a valid MAC for a malicious session_id.
    if not _is_valid_session_id(session_id):
        log.warning(
            "verify_session: MAC-valid token has invalid session_id shape "
            "— rejected (sid prefix=%r)",
            session_id[:32],
        )
        return None
    expected_mac = _session_mac(secret, session_id, key_id)
    if not hmac.compare_digest(provided_mac, expected_mac):
        return None
    return session_id


def resolve_session(
    *,
    header_value: str,
    cookie_value: str,
    key_id: str,
    secret: bytes,
    require_signed: bool,
) -> Tuple[Optional[str], str]:
    """single decision point for session identification.

    Returns ``(session_id, source)`` where ``source`` is one of:

      ``"header"``    — signed X-Session-ID header, MAC verified
      ``"cookie"``    — signed llm_session cookie, MAC verified
      ``"unsigned"``  — raw header or cookie accepted under
                        ``require_signed=False`` (shape validated)
      ``"none"``      — no session found; caller generates a fresh one

    Precedence (identical whether or not signed sessions are required):

      1. Signed X-Session-ID header
      2. Signed llm_session cookie
      3. (only if require_signed=False) Unsigned header or cookie,
         subject to ``_is_valid_session_id`` shape validation

    Why header > cookie: the header is set explicitly per-request by the
    client, so it wins ties against the ambient cookie. This preserves the
    previous precedence and matches the docstring of the old inline logic
    in ``get_session_id``.

    When ``require_signed=True`` and the header is unsigned but a valid
    signed cookie exists, this returns ``(cookie_sid, "cookie")`` —
    silently. The web UI's unsigned header is *expected* on this
    deployment and the cookie is the authoritative source; no log line is
    emitted for the silent fallback. The caller logs a DEBUG line only
    when it receives ``(None, "none")`` and generates a fresh session.

    When ``require_signed=False`` and the header is unsigned but
    shape-valid, this returns ``(header_value, "unsigned")``. The caller
    logs at DEBUG: ``"accepting unsigned X-Session-ID as conv_id (source=unsigned)"``.
    This is the opt-in path for single-tenant deployments.

    Args:
        header_value:  The raw ``X-Session-ID`` header (may be empty or
                       whitespace). Already stripped by caller is fine;
                       this function strips again for safety.
        cookie_value:  The raw ``llm_session`` cookie (may be empty).
        key_id:        The caller's API key id (from ``verify_api_key``).
                       ``"anonymous"`` if the deployment has no per-key
                       identity (single static key).
        secret:        The derived session secret, from
                       ``_session_secret(master_key)``.
        require_signed: When True, unsigned header/cookie are rejected.
                        When False, unsigned but shape-valid header/cookie
                        is accepted as the session_id.

    Returns:
        ``(session_id, source)`` where session_id is ``None`` if source
        is ``"none"``.
    """
    header_value = (header_value or "").strip()
    cookie_value = (cookie_value or "").strip()

    # 1. Signed header — highest priority
    if header_value:
        sid = verify_session(header_value, key_id, secret)
        if sid is not None:
            return sid, "header"

    # 2. Signed cookie — second priority
    if cookie_value:
        sid = verify_session(cookie_value, key_id, secret)
        if sid is not None:
            return sid, "cookie"

    # 3. Unsigned acceptance — only when explicitly opted out of signing
    if not require_signed:
        candidate = header_value or cookie_value
        if candidate:
            if _is_valid_session_id(candidate):
                return candidate, "unsigned"
            log.warning(
                "Rejecting unsigned session id with invalid shape "
                "(prefix=%r, len=%d) — falling through to fresh session",
                candidate[:32], len(candidate),
            )

    return None, "none"

def derive_conversation_session(secret: bytes, key_id: str, conversation_id: str) -> str:
    """Stable session id for a client-chosen conversation label.

    The id is an HMAC over ``(key_id, conversation_id)``, so:
      * the same browser conversation maps to the same server-side session
        after a reload (resume / checkpoints keep working);
      * a different API key can never address it (the key id is in the MAC);
      * the label itself is never trusted as a storage key.

    Only the holder of the API key can reach its own conversations; clients
    that share one key share its conversations (use one key per user/tenant).
    """
    if not _is_valid_session_id(conversation_id):
        raise ValueError("invalid conversation id shape")
    msg = f"conv:{key_id}:{conversation_id}".encode("utf-8")
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()[:48]

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
    "resolve_session",     # centralised entry point
    "derive_conversation_session",
    "is_secure_request",
    "_session_secret",     # exposed for api_server.get_session_id
    "_is_valid_session_id",  # exposed for tests / admin introspection
]
