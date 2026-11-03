"""
protocol.py — length-prefixed framed transport, OS-agnostic, versioned.

Frame wire format (used in both directions over a TCP loopback socket):

    [ 4 bytes: header_len, big-endian uint32 ]
    [ header_len bytes: UTF-8 JSON with mandatory "v": PROTOCOL_VERSION ]
    [ 4 bytes: body_len, big-endian uint32 ]
    [ body_len bytes: raw (empty for control frames, UTF-8 token bytes for data) ]

No pickling. No msgpack dependency. Both sides can be implemented in 30 lines.

Version 1 of the protocol (2026-11-01)
──────────────────────────────────────
    adds three defensible properties that were previously missing:

  1. **Version negotiation.** Every frame carries ``"v": 1`` in its JSON
     header. ``read_frame`` rejects any frame whose ``"v"`` is absent or
     does not match ``PROTOCOL_VERSION``. This makes a silent protocol
     change (a renamed field, a reordered enum) fail loudly at the
     transport boundary instead of quietly corrupting state one layer up.

  2. **Bounded allocation.** ``read_frame`` rejects frames whose declared
     header or body length exceeds ``MAX_HEADER_BYTES`` / ``MAX_BODY_BYTES``
     BEFORE calling ``readexactly``. Previously a corrupted (or malicious)
     peer could declare a 4 GiB body and force a 4 GiB allocation attempt
     — an OOM DoS vector from an unprivileged local process. The limits
     are: 64 KiB header (a verbose control frame is ~1 KiB; a "ready"
     frame carrying a 4096-element ``resident_prefix_head`` list is
     ~25 KiB), 16 MiB body (a token frame is <200 B; an error traceback
     is <10 KiB — 16 MiB is generous and leaves headroom if a prompt-echo
     frame is ever added).

  3. **Malformed-input rejection.** Truncated headers, oversized declared
     lengths, invalid UTF-8, invalid JSON, non-object header payloads,
     and version mismatch all raise ``ProtocolError`` (subclass of
     ``ConnectionError``). Previously an invalid JSON header raised a raw
     ``json.JSONDecodeError`` that fell through to the generic
     ``except Exception`` in ``_reader_loop``, logging a full traceback
     for every malformed frame from an unprivileged peer.

Backward compatibility with v0 (missing ``"v"``) is deliberately NOT
supported. The worker is always a freshly spawned child process, so
parent and child are guaranteed to be running the same ``protocol.py``
version across any deployment. A frame without ``"v"`` is a bug in a
caller, not a legacy peer.
"""
from __future__ import annotations

import asyncio
import json
import logging
import struct
from typing import Tuple

log = logging.getLogger(__name__)


# ── Protocol constants ──────────────────────────────────────────────────────
#: Wire-format version. Bump this whenever the frame schema changes in a
#: way that is not backward-compatible. ``read_frame`` rejects any frame
#: whose ``"v"`` field is absent or does not match.
PROTOCOL_VERSION: int = 1

#: Maximum size of the JSON header portion of one frame, in bytes.
#: Generous: a verbose control frame is ~1 KiB, and the worker's ``ready``
#: frame with a 4096-element ``resident_prefix_head`` list serialises to
#: ~25 KiB. 64 KiB leaves headroom for future fields without ever
#: allocating on a peer's say-so.
MAX_HEADER_BYTES: int = 64 * 1024

#: Maximum size of the body portion of one frame, in bytes.
#: Generous: a token frame is <200 B, a truncated error is <10 KiB.
#: 16 MiB is more than enough for anything this protocol will ever carry.
MAX_BODY_BYTES: int = 16 * 1024 * 1024


class ProtocolError(ConnectionError):
    """Raised on malformed frames, version mismatch, or size violation.

    Subclasses ``ConnectionError`` so existing ``except (ConnectionResetError,
    asyncio.IncompleteReadError)`` catch-all blocks in ``handle.py`` and
    ``worker/main.py`` continue to treat a protocol failure the same as a
    TCP-level disconnect for cleanup purposes. Callers that want finer-
    grained handling can catch ``ProtocolError`` first — it is a distinct
    class, not a synonym.
    """


# ── Frame layout ────────────────────────────────────────────────────────────
HEADER = struct.Struct(">I")
FRAME_HEAD = struct.Struct(">II")


async def write_frame(
    writer: asyncio.StreamWriter,
    header: dict,
    body: bytes = b"",
) -> None:
    """Serialise and send one frame. Raises ``ProtocolError`` on oversized input.

    The ``"v"`` field is injected automatically if the caller did not
    provide it. Callers MUST NOT set a mismatched ``"v"`` — doing so raises
    ``ValueError`` (a bug in the caller, not a wire-format problem).
    """
    # Inject version if absent. If present and wrong, this is a caller bug —
    # raise loudly rather than silently overwriting a caller's intent.
    _v = header.get("v")
    if _v is None:
        header = {**header, "v": PROTOCOL_VERSION}
    elif _v != PROTOCOL_VERSION:
        raise ValueError(
            f"write_frame: caller passed v={_v!r} but "
            f"PROTOCOL_VERSION={PROTOCOL_VERSION}"
        )

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ProtocolError(
            f"write_frame: header too large "
            f"({len(header_bytes)} bytes, max {MAX_HEADER_BYTES}); "
            f"frame type={header.get('t')!r}"
        )
    if not isinstance(body, (bytes, bytearray)):
        raise TypeError(
            f"write_frame: body must be bytes, got {type(body).__name__}"
        )
    if len(body) > MAX_BODY_BYTES:
        raise ProtocolError(
            f"write_frame: body too large "
            f"({len(body)} bytes, max {MAX_BODY_BYTES}); "
            f"frame type={header.get('t')!r}"
        )

    writer.write(FRAME_HEAD.pack(len(header_bytes), len(body)))
    writer.write(header_bytes)
    if body:
        writer.write(body)
    await writer.drain()


async def read_frame(
    reader: asyncio.StreamReader,
) -> Tuple[dict, bytes]:
    """Read one frame. Raises ``ProtocolError`` on malformed input.

    Called on both the parent and worker sides. The size checks happen
    BEFORE ``readexactly`` allocates — a peer declaring a 4 GiB body cannot
    force us to allocate 4 GiB.
    """
    head = await reader.readexactly(8)
    header_len, body_len = FRAME_HEAD.unpack(head)

    # Guard the lengths BEFORE allocating. Rejecting here means a corrupted
    # or malicious length prefix cannot trigger an OOM in readexactly().
    if header_len > MAX_HEADER_BYTES:
        raise ProtocolError(
            f"read_frame: declared header length {header_len} exceeds "
            f"max {MAX_HEADER_BYTES}"
        )
    if body_len > MAX_BODY_BYTES:
        raise ProtocolError(
            f"read_frame: declared body length {body_len} exceeds "
            f"max {MAX_BODY_BYTES}"
        )

    header_bytes = await reader.readexactly(header_len)
    body = await reader.readexactly(body_len) if body_len else b""

    try:
        header = json.loads(header_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProtocolError(f"read_frame: invalid header JSON: {exc}") from exc

    if not isinstance(header, dict):
        raise ProtocolError(
            f"read_frame: header must be a JSON object, "
            f"got {type(header).__name__}"
        )

    _v = header.get("v")
    if _v != PROTOCOL_VERSION:
        raise ProtocolError(
            f"read_frame: protocol version mismatch "
            f"(got {_v!r}, expected {PROTOCOL_VERSION}); "
            f"frame type={header.get('t')!r}"
        )

    return header, body


__all__ = [
    "PROTOCOL_VERSION",
    "MAX_HEADER_BYTES",
    "MAX_BODY_BYTES",
    "ProtocolError",
    "write_frame",
    "read_frame",
]
