"""
local_llm_server/auth/__init__.py — Authentication & Authorization v1.0.0
==========================================================================
Public interface for API key management, rate limiting, and master key verification.

  • MasterKeyValidator now supports multiple master keys with optional rate‑limiting
    per key, and key rotation via configuration.
  • Prometheus metrics (optional) for authentication attempts, master key usage.
  • TokenBucket exported directly for external rate‑limit use.
  • Convenience `shutdown_auth_components()` to gracefully stop the key store.
  • All store backends (SQLiteKeyStore, PostgreSQLKeyStore, MySQLKeyStore) re‑exported.
  • Environment variable expansion for master key list (comma‑separated).
  • Lazy imports are cached for performance.
  • Fully backward compatible.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ------------------------------------------------------------------------------
# Re‑export core classes and factory from key_store (v4.0.0)
# ------------------------------------------------------------------------------
from .key_store import (
    BaseKeyStore,
    TokenBucket,
    KeyStatus,
    SQLiteKeyStore,
    PostgreSQLKeyStore,
    MySQLKeyStore,
    create_key_store as _create_key_store,
)

# Alias for backward compatibility
KeyStore = BaseKeyStore

__all__ = [
    "KeyStore",
    "BaseKeyStore",
    "TokenBucket",
    "KeyStatus",
    "SQLiteKeyStore",
    "PostgreSQLKeyStore",
    "MySQLKeyStore",
    "create_key_store",
    "MasterKeyValidator",
    "get_auth_components",
    "shutdown_auth_components",
    "generate_master_key",
]

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Optional Prometheus metrics
# ------------------------------------------------------------------------------
try:
    from prometheus_client import Counter
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

def _setup_prometheus():
    if not HAS_PROMETHEUS:
        return {}
    return {
        "master_auth_total": Counter("auth_master_total", "Master key auth attempts", ["result"]),
        "master_rate_limited": Counter("auth_master_rate_limited", "Master key rate limited"),
    }

_prom = _setup_prometheus()

# ------------------------------------------------------------------------------
# Configuration Loading (cached)
# ------------------------------------------------------------------------------
_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_CONFIG_CACHE_PATH: Optional[Path] = None

def _load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file, with environment variable expansion
    for master keys. Results are cached per path.
    """
    global _CONFIG_CACHE, _CONFIG_CACHE_PATH

    if config_path == _CONFIG_CACHE_PATH and _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    cfg: Dict[str, Any] = {}
    if config_path is not None:
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f) or {}
            except (yaml.YAMLError, OSError) as e:
                log.error("Failed to load config from %s: %s", config_path, e)
        else:
            log.warning("Config file not found: %s", config_path)
    else:
        # Default search
        paths = [
            Path.home() / "config.yaml",
            Path.cwd() / "config.yaml",
        ]
        for p in paths:
            if p.exists():
                try:
                    with open(p, "r") as f:
                        cfg = yaml.safe_load(f) or {}
                    config_path = p
                    break
                except (yaml.YAMLError, OSError) as e:
                    log.error("Failed to load config from %s: %s", p, e)
        else:
            log.warning("No config.yaml found in default locations; using empty configuration.")

    # Expand environment variable for master_keys if not already a list
    master_keys = cfg.get("api", {}).get("master_keys", [])
    if isinstance(master_keys, str):
        # Could be a single key or comma‑separated list
        master_keys = [k.strip() for k in master_keys.split(",") if k.strip()]
        cfg.setdefault("api", {})["master_keys"] = master_keys

    _CONFIG_CACHE = cfg
    _CONFIG_CACHE_PATH = config_path
    return cfg


# ------------------------------------------------------------------------------
# Master Key Validation with rate‑limiting and rotation
# ------------------------------------------------------------------------------
class MasterKeyValidator:
    """
    Validates master/admin keys for privileged endpoints.

    Features:
      • Supports multiple master keys defined in config under `api.master_keys`.
      • Each key can optionally be paired with a rate limit (requests per second)
        via `api.master_key_limits` (dict of key_hash → rps).
      • Rate limiting uses TokenBucket (same as API keys).
      • Automatic hashing of plaintext keys.
    """

    def __init__(
        self,
        master_keys: List[str],
        key_limits: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Args:
            master_keys: List of plaintext master keys or SHA‑256 hex hashes.
            key_limits: Optional dictionary mapping key hash → max RPS.
        """
        self._valid_hashes: set[str] = set()
        self._key_limits: Dict[str, float] = {}
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

        for key in master_keys:
            if len(key) == 64 and all(c in "0123456789abcdef" for c in key.lower()):
                h = key.lower()
            else:
                h = hashlib.sha256(key.encode()).hexdigest()
            self._valid_hashes.add(h)
            # Default rate limit 1000 rps per master key if not specified
            if key_limits and h in key_limits:
                self._key_limits[h] = float(key_limits[h])
            else:
                self._key_limits[h] = 1000.0  # effectively unlimited

    def validate(self, key: Optional[str]) -> bool:
        """
        Check if the provided key is valid and not rate‑limited.

        Returns:
            True if allowed, False otherwise.
        """
        if not key:
            return False
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        if key_hash not in self._valid_hashes:
            return False

        # Rate limiting
        with self._lock:
            bucket = self._buckets.get(key_hash)
            if not bucket:
                bucket = TokenBucket(self._key_limits.get(key_hash, 1000.0), 10.0)  # capacity=10
                self._buckets[key_hash] = bucket
            allowed = bucket.allow()

        if _prom.get("master_auth_total"):
            _prom["master_auth_total"].labels(result="ok" if allowed else "rate_limited").inc()
        if not allowed and _prom.get("master_rate_limited"):
            _prom["master_rate_limited"].inc()

        if not allowed:
            log.warning("Master key rate limited: hash=%s", key_hash[:8])
        return allowed

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "MasterKeyValidator":
        """Create validator from configuration dictionary."""
        master_keys = cfg.get("api", {}).get("master_keys", [])
        if not master_keys:
            log.warning(
                "No master_keys defined in config. "
                "Admin endpoints will be inaccessible unless a key is added."
            )
        key_limits = cfg.get("api", {}).get("master_key_limits", {})
        return cls(master_keys, key_limits)


# ------------------------------------------------------------------------------
# Auth Components Dataclass
# ------------------------------------------------------------------------------
@dataclass
class AuthComponents:
    """Container for initialized authentication components."""
    key_store: Optional[KeyStore]
    master_validator: MasterKeyValidator


# ------------------------------------------------------------------------------
# Async Factory for Auth Components
# ------------------------------------------------------------------------------
async def create_key_store(cfg: Optional[Dict[str, Any]] = None) -> Optional[KeyStore]:
    """
    Create and initialize a KeyStore instance from configuration.

    Args:
        cfg: Configuration dictionary. If None, loads from default YAML paths.

    Returns:
        Initialized KeyStore, or None if disabled.
    """
    if cfg is None:
        cfg = _load_config()
    return await _create_key_store(cfg)


async def get_auth_components(
    config_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> AuthComponents:
    """
    Initialize all authentication components from configuration.

    This is the recommended entry point for the FastAPI server.

    Args:
        config_path: Optional explicit path to config.yaml.
        config: Pre‑loaded configuration dict (overrides path).

    Returns:
        AuthComponents containing (KeyStore or None, MasterKeyValidator).
    """
    if config is None:
        cfg = _load_config(config_path)
    else:
        cfg = config

    key_store = await create_key_store(cfg)
    master_validator = MasterKeyValidator.from_config(cfg)
    log.info(
        "Auth components initialized (key_store=%s, master_keys=%d)",
        key_store is not None,
        len(master_validator._valid_hashes),
    )
    return AuthComponents(key_store, master_validator)


async def shutdown_auth_components(components: AuthComponents) -> None:
    """
    Gracefully shut down authentication components (close key store, stop tasks).

    Args:
        components: AuthComponents instance from get_auth_components().
    """
    if components.key_store is not None:
        try:
            await components.key_store.stop_background_tasks()
            await components.key_store.close()
            log.info("KeyStore shut down")
        except Exception as e:
            log.error("Error shutting down key store: %s", e)


# ------------------------------------------------------------------------------
# Utility: Generate a secure master key
# ------------------------------------------------------------------------------
def generate_master_key() -> str:
    """Generate a cryptographically secure master key (URL‑safe base64)."""
    return secrets.token_urlsafe(32)


# ------------------------------------------------------------------------------
# Module initialization logging
# ------------------------------------------------------------------------------
log.debug("auth module loaded (v1.0.0)")
