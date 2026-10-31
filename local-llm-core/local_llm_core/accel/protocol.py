"""
protocol.py — length-prefixed framed transport, OS-agnostic.

Frame wire format (used in both directions over a TCP loopback socket):

    [ 4 bytes: header_len, big-endian uint32 ]
    [ header_len bytes: UTF-8 JSON ]
    [ 4 bytes: body_len, big-endian uint32 ]
    [ body_len bytes: raw (empty for control frames, UTF-8 token bytes for data) ]

No pickling. No msgpack dependency. Both sides can be implemented in 30 lines.
"""
from __future__ import annotations

import asyncio
import json
import struct
from typing import Any, Tuple

HEADER = struct.Struct(">I")
FRAME_HEAD = struct.Struct(">II")


async def write_frame(
    writer: asyncio.StreamWriter,
    header: dict,
    body: bytes = b"",
) -> None:
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    writer.write(FRAME_HEAD.pack(len(header_bytes), len(body)))
    writer.write(header_bytes)
    if body:
        writer.write(body)
    await writer.drain()


async def read_frame(
    reader: asyncio.StreamReader,
) -> Tuple[dict, bytes]:
    head = await reader.readexactly(8)
    header_len, body_len = FRAME_HEAD.unpack(head)
    header_bytes = await reader.readexactly(header_len)
    body = await reader.readexactly(body_len) if body_len else b""
    header = json.loads(header_bytes)
    return header, body
