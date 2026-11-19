"""sd_notify.py — minimal systemd notification socket helper.

No dependencies, no systemd-python package required. If NOTIFY_SOCKET is
not set (i.e. not running under systemd), every call is a safe no-op that
returns False.

Implements the subset of
https://www.freedesktop.org/software/systemd/man/sd_notify.html that this
project needs:

    READY=1      — daemon finished initialising and can accept traffic.
                   Only meaningful with Type=notify. Once sent, systemd
                   stops blocking on this unit's start and lets
                   After=local-llm-cloud.service downstream units start.

    WATCHDOG=1   — heartbeat. Must be sent at least once per WatchdogSec/2.
                   If the process stops sending it (frozen event loop,
                   stuck syscall, deadlock), systemd kills and restarts
                   the unit. This is what makes a HANG visible and
                   recoverable — the whole reason the P0 deadlock went
                   unnoticed was that systemd had no way to detect it.

    STOPPING=1   — optional; sent at the start of a clean shutdown.

Abstract-namespace sockets (NOTIFY_SOCKET starting with '@') are handled
per sd_notify(3): the '@' is replaced with a NUL byte before connect().
"""
from __future__ import annotations

import logging
import os
import socket

log = logging.getLogger(__name__)


def sd_notify(message: str) -> bool:
    """Send a message to the systemd notification socket.

    Returns True if delivered, False if NOTIFY_SOCKET is unset (not
    running under systemd) or the send failed. Never raises — a
    notification failure must never take the service down.
    """
    addr = os.environ.get("NOTIFY_SOCKET", "").strip()
    if not addr:
        return False

    # Abstract-namespace UNIX socket: leading '@' → NUL byte per sd_notify(3)
    if addr[0] == "@":
        addr = "\x00" + addr[1:]

    try:
        with socket.socket(
            socket.AF_UNIX,
            socket.SOCK_DGRAM | getattr(socket, "SOCK_CLOEXEC", 0),
        ) as s:
            s.connect(addr)
            s.sendall(message.encode("utf-8"))
        return True
    except OSError as exc:
        # Common outside systemd, or during shutdown races. Debug only,
        # so a heartbeat failure does not spam the log.
        log.debug("sd_notify(%r) failed: %s", message, exc)
        return False
