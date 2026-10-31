"""
local_llm_core/hal/suspend_inhibitor.py — prevents the host OS from
suspending/hibernating while the server is actively serving, using
systemd-logind's standard inhibitor-lock mechanism.

v1.0.0 – Production‑ready, safe no‑op fallback.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Optional

log = logging.getLogger(__name__)


class SuspendInhibitor:
    """
    Holds a systemd-logind sleep/idle inhibitor for as long as `start()` has
    been called and `stop()` hasn't. Safe no-op on non-systemd systems or if
    `systemd-inhibit` isn't on PATH — logs a warning once and does nothing,
    rather than failing server startup over a missing OS integration.
    """

    def __init__(self, who: str = "local-llm-server",
                 why: str = "Serving inference requests",
                 what: str = "sleep:idle:handle-lid-switch:handle-suspend-key:handle-hibernate-key:handle-power-key") -> None:
        self._who = who
        self._why = why
        self._what = what
        self._proc: Optional[subprocess.Popen] = None

    @property
    def active(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> bool:
        if self.active:
            return True
        binary = shutil.which("systemd-inhibit")
        if not binary:
            log.warning(
                "SuspendInhibitor: systemd-inhibit not found on PATH — "
                "cannot prevent host suspend. If this host suspends while "
                "serving, requests will fail/hang for the duration. "
                "Install systemd, or disable suspend on this host manually."
            )
            return False
        try:
            self._proc = subprocess.Popen(
                [binary, f"--what={self._what}", f"--who={self._who}",
                 f"--why={self._why}", "--mode=block",
                 "sleep", "infinity"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            log.info("SuspendInhibitor: active (%s) — host will not "
                      "suspend/idle-sleep while this process runs.", self._what)
            return True
        except Exception as exc:
            log.warning("SuspendInhibitor: failed to start: %s", exc)
            self._proc = None
            return False

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            log.info("SuspendInhibitor: released.")
            self._proc = None
