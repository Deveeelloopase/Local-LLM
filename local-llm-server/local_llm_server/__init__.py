"""
local_llm_server/__init__.py — Local LLM Cloud Server v1.0.0
===============================================================
Top‑level package for the production REST API server.

    - All internal subpackages upgraded to v4.0.0 (auth, memory, monitoring, serving).
    - `load_config()` now supports environment variable expansion natively.
    - Lazy imports are cached for performance.
    - `get_version()` helper added.

Public exports:
    - __version__ : Package version
    - __author__  : Package author
    - load_config : Helper to load configuration from default YAML paths
    - get_version : Returns the package version string
    - auth        : Authentication and API key management
    - memory      : Conversation history storage
    - monitoring  : Metrics, telemetry, and TUI monitor
    - serving     : FastAPI server, middleware, WebSocket, models
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

__version__ = "1.0.0"
__author__ = "Devellopase"

_log = logging.getLogger(__name__)

# Cache for lazy subpackage imports
_IMPORT_CACHE: Dict[str, Any] = {}


# ------------------------------------------------------------------------------
# Environment variable expansion helper
# ------------------------------------------------------------------------------
def _expand_env_vars(value: Any, _depth: int = 0) -> Any:
    """Recursively expand ${VAR} and ${VAR:default} in strings."""
    if _depth > 10:
        return value
    if isinstance(value, str):
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        def repl(match):
            var = match.group(1)
            default = match.group(2)
            return os.environ.get(var, default if default is not None else "")
        return re.sub(pattern, repl, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v, _depth + 1) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item, _depth + 1) for item in value]
    return value


# ------------------------------------------------------------------------------
# Configuration loading (default: ~/config.yaml or ./config.yaml)
# ------------------------------------------------------------------------------
def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file, expanding environment variables.

    Args:
        config_path: Explicit path to config.yaml. If None, searches:
                     ~/config.yaml, ./config.yaml.

    Returns:
        Configuration dictionary. Returns empty dict if no file found.

    Raises:
        FileNotFoundError: If explicit config_path is provided but not found.
    """
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        paths = [
            Path.home() / "config.yaml",
            Path.cwd() / "config.yaml",
        ]
        cfg = {}
        for p in paths:
            if p.exists():
                with open(p, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                    _log.debug("Loaded config from %s", p)
                    break
        else:
            _log.warning("No config.yaml found in default locations; using empty configuration.")

    return _expand_env_vars(cfg)


# ------------------------------------------------------------------------------
# Version helper
# ------------------------------------------------------------------------------
def get_version() -> str:
    """Return the package version as a string."""
    return __version__


# ------------------------------------------------------------------------------
# Lazy imports for subpackages (cached)
# ------------------------------------------------------------------------------
def __getattr__(name: str):
    """Lazy attribute access for subpackages. Cached after first import."""
    if name in _IMPORT_CACHE:
        return _IMPORT_CACHE[name]

    if name == "auth":
        try:
            from . import auth as _auth
            _IMPORT_CACHE[name] = _auth
            return _auth
        except ImportError as e:
            raise AttributeError(f"Subpackage 'auth' could not be imported: {e}") from e

    if name == "memory":
        try:
            from . import memory as _memory
            _IMPORT_CACHE[name] = _memory
            return _memory
        except ImportError as e:
            raise AttributeError(f"Subpackage 'memory' could not be imported: {e}") from e

    if name == "monitoring":
        try:
            from . import monitoring as _monitoring
            _IMPORT_CACHE[name] = _monitoring
            return _monitoring
        except ImportError as e:
            raise AttributeError(f"Subpackage 'monitoring' could not be imported: {e}") from e

    if name == "serving":
        try:
            from . import serving as _serving
            _IMPORT_CACHE[name] = _serving
            return _serving
        except ImportError as e:
            raise AttributeError(f"Subpackage 'serving' could not be imported: {e}") from e

    raise AttributeError(f"module 'local_llm_server' has no attribute {name!r}")


# ------------------------------------------------------------------------------
# Module exports
# ------------------------------------------------------------------------------
__all__ = [
    "__version__",
    "__author__",
    "load_config",
    "get_version",
    "auth",
    "memory",
    "monitoring",
    "serving",
]
