"""Helpers shared by the command-line tools.

Version: 1.0.0
"""
from __future__ import annotations

import os
import re
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Optional

__version__ = "1.0.0"

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def find_config(explicit: Optional[str] = None) -> Optional[Path]:
    """Explicit path, ``$LLM_CONFIG_PATH``, then the usual locations."""
    for cand in (explicit, os.environ.get("LLM_CONFIG_PATH")):
        if cand and Path(cand).is_file():
            return Path(cand)
    for cand in (
        Path.cwd() / "config.yaml",
        Path.home() / "config.yaml",
        Path("/etc/local-llm/config.yaml"),
    ):
        if cand.is_file():
            return cand
    return None


def _expand(value: Any) -> Any:
    """Expand ``${VAR}`` and ``${VAR:-default}`` in every string of a loaded config."""
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def load_config(explicit: Optional[str] = None) -> Dict[str, Any]:
    """Load ``config.yaml`` (environment variables expanded).  ``{}`` when absent."""
    path = find_config(explicit)
    if path is None:
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            return _expand(yaml.safe_load(fh) or {})
    except Exception as exc:
        sys.stderr.write(f"WARNING: could not read {path}: {exc}\n")
        return {}


def detect_system_info() -> Any:
    """Detector ``SystemInfo`` or ``None`` when detection is unavailable."""
    try:
        from local_llm_core.hal.hardware_detector import HardwareDetector

        return HardwareDetector().detect_all()
    except Exception as exc:
        sys.stderr.write(f"WARNING: hardware detection failed ({exc}); continuing\n")
        return None


def server_is_listening(cfg: Dict[str, Any], timeout: float = 0.5) -> bool:
    """True when something accepts connections on the configured server port."""
    server = (cfg or {}).get("server") or {}
    host = str(server.get("host") or "127.0.0.1")
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    try:
        port = int(server.get("port") or 8080)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False
