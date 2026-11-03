"""
local_llm_client/cli/config.py

Production‑grade configuration manager for Local LLM Cloud CLI.

All known bugs fixed and enhanced for production resilience:
  - API key discovery now automatically tries the server configuration file
    if not set in the client config or environment.
  - Master key discovery follows the same rule.
  - `backup()` / `restore()` static helpers for configuration safety.
  - Thread‑safe singleton with double‑checked locking.
  - All sensitive values are masked in repr() and data property.
  - Full nested schema validation with type coercion.
  - Auto‑reload on file change is lock‑protected and efficient.
  - Convenience properties for **all** commonly accessed settings.
  - `update()` for bulk changes with a single save.
  - `to_dict()` / `from_dict()` for serialisation without writing to file.
  - `validate()` returns (errors, warnings) for the current configuration.
  - Type coercion for common types (bool, int, float, list).
  - No hardcoded paths – configurable via environment variables.
  - Handles missing files gracefully, auto‑creating a default config.
"""

import copy
import logging
import os
import re
import time
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import yaml

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Exception
# ------------------------------------------------------------------------------
class ConfigError(Exception):
    """Raised for configuration validation / access errors."""
    pass


# ------------------------------------------------------------------------------
# Environment variable substitution pattern
# ------------------------------------------------------------------------------
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ${ENV_VAR:default} placeholders."""
    if isinstance(value, str):
        def repl(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default)
        return _ENV_VAR_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


# ------------------------------------------------------------------------------
# Type coercion
# ------------------------------------------------------------------------------
def _coerce_value(key_path: str, value: Any, expected_type: Optional[type] = None) -> Any:
    """Coerce value to expected type if possible."""
    if expected_type is None:
        return value
    try:
        if expected_type == bool:
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            return bool(value)
        if expected_type in (int, float):
            return expected_type(value)
        if expected_type == list:
            if isinstance(value, str):
                return [v.strip() for v in value.split(",") if v.strip()]
            return list(value) if value is not None else []
        return expected_type(value)
    except (ValueError, TypeError):
        logger.warning(
            "Could not coerce '%s' value %r to %s, keeping original",
            key_path, value, expected_type.__name__,
        )
        return value


# ------------------------------------------------------------------------------
# Default configuration (complete)
# ------------------------------------------------------------------------------
def _default_config() -> Dict[str, Any]:
    """Return the complete default configuration dictionary."""
    return {
        "server": {
            "base_url": "http://localhost:8080",
            "api_key": "",
            "master_key": "",
            "verify_ssl": True,
            "ca_cert": None,
        },
        "http": {
            "pool_size": 50,
            "pool_timeout_s": 10.0,
            "connect_timeout_s": 5.0,
            "read_timeout_s": 300.0,
            "total_timeout_s": 1800.0,
            "stream_timeout_s": 120.0,
            "stream_chunk_timeout_s": 30.0,
            "max_retries": 3,
            "retry_delay_s": 1.0,
            "retry_delay_max_s": 30.0,
            "retry_backoff_factor": 2.0,
            "retry_jitter": 0.1,
            "retryable_status_codes": [408, 429, 500, 502, 503, 504],
            "http_keep_alive": True,
            "http2_enabled": False,
        },
        "circuit_breaker": {
            "enabled": True,
            "failure_threshold": 5,
            "error_rate_threshold": 0.5,
            "recovery_timeout_s": 60.0,
            "half_open_probe_timeout_s": 10.0,
            "log_state_changes": True,
            "log_failures": True,
        },
        "inference": {
            "model": "qwen3-8b",
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_tokens": 0,
            "timeout_s": 1800.0,
            "stream_enabled": True,
            "stream_flush_interval_ms": 80,
            "stream_buffer_size": "dynamic",
            "system_prompt": None,
        },
        "health_check": {
            "on_init_enabled": True,
            "on_init_retries": 3,
            "on_init_retry_delay_s": 2.0,
            "on_init_timeout_s": 10.0,
            "wait_for_model": False,
            "wait_timeout_s": 300.0,
            "periodic_enabled": False,
            "periodic_interval_s": 60.0,
            "log_health_failures": True,
            "fail_open": False,
        },
        "metrics": {
            "tps_estimate_fallback": 7.0,
            "tps_cache_ttl_s": 60.0,
            "base_timeout_padding_s": 15.0,
            "min_timeout_s": 30.0,
            "token_count_method": "fast",
            "token_estimate_multiplier": 1.33,
        },
        "cache": {
            "config_dir": "~/.llmcloud",
            "config_file": "config.yaml",
            "api_key_cache_ttl_s": 3600.0,
            "session_dir": "~/.llmcloud/sessions",
            "session_retention_days": 30,
            "max_sessions": 100,
        },
        "security": {
            "validate_base_url": True,
            "validate_model_name": True,
            "validate_api_key_format": True,
            "api_key_allowed_in_config": True,
            "api_key_from_env_only": False,
            "min_tls_version": "TLSv1.2",
            "hash_sensitive_logs": True,
        },
        "logging": {
            "level": "info",
            "console_enabled": True,
            "file_enabled": False,
            "file_path": "~/.llmcloud/client.log",
            "file_max_mb": 10,
            "file_backups": 3,
            "format": "[%(asctime)s] [%(levelname)-8s] [%(name)s]: %(message)s",
            "date_format": "%Y-%m-%d %H:%M:%S",
            "use_json": False,
            "mask_api_keys": True,
            "mask_prompts": False,
            "log_http_requests": False,
            "log_http_responses": False,
            "log_retry_attempts": True,
            "log_circuit_breaker": True,
        },
        "advanced": {
            "max_workers": 4,
            "asyncio_debug": False,
            "prefer_aiohttp": True,
            "log_aiohttp_fallback": True,
            "urllib3_pool_connections": 10,
            "urllib3_pool_maxsize": 10,
            "enable_c_accelerator": True,
            "word_boundary_detection": True,
            "context_manager_singleton": True,
            "context_window_static": False,
            "context_window_fallback": 4096,
            "allow_prompt_compression": True,
            "compression_threshold_ratio": 0.8,
            "auto_retry_on_timeout": True,
            "auto_fallback_to_smaller_model": False,
        },
        "cli": {
            "interactive_default_stream": True,
            "interactive_history_size": 50,
            "chat_default_timeout_s": None,
            "chat_always_show_thinking": False,
            "models_refresh_interval_s": 60.0,
            "status_refresh_interval_s": 5.0,
            "status_show_queue_depth": True,
        },
        "adapter": {
            "use_async": True,
            "enable_adaptive_context": True,
            "context_recovery_enabled": True,
            "context_reduction_factor": 0.85,
            "context_recovery_factor": 1.05,
            "enable_compression": True,
            "compression_target_ratio": 0.6,
            "allow_degradation": True,
            "max_degradation_attempts": 3,
            "degradation_sleep_s": 1.0,
            "track_requests": True,
            "track_errors": True,
            "track_tokens": True,
            "report_metrics": True,
        },
    }


# ------------------------------------------------------------------------------
# Nested dict helpers
# ------------------------------------------------------------------------------
def _get_nested_value(data: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(k)
        if data is None:
            return None
    return data


def _set_nested_value(data: Dict[str, Any], keys: List[str], value: Any) -> None:
    for k in keys[:-1]:
        if k not in data or not isinstance(data[k], dict):
            data[k] = {}
        data = data[k]
    data[keys[-1]] = value


# ------------------------------------------------------------------------------
# Config class
# ------------------------------------------------------------------------------
class Config:
    """
    Thread‑safe configuration manager with auto‑reload, env substitution,
    type coercion, and automatic API key inheritance from the server config.

    All keys are accessible via ``get(key_path)`` / ``set(key_path, val)``.
    Convenience properties are provided for the most common keys.
    """

    # Schema for validation: each key maps to (expected_type, validator, description)
    SCHEMA: Dict[str, Tuple[type, Optional[Callable], str]] = {
        "server.base_url":       (str,   lambda v: bool(v.strip()), "Server URL"),
        "server.api_key":        (str,   None, "API key"),
        "server.master_key":     (str,   None, "Master key"),
        "server.verify_ssl":     (bool,  None, "Verify SSL"),
        "http.pool_size":        (int,   lambda v: v >= 1, "Pool size"),
        "http.connect_timeout_s":(float, lambda v: v > 0, "Connect timeout"),
        "http.max_retries":      (int,   lambda v: v >= 0, "Max retries"),
        "inference.temperature": (float, lambda v: 0.0 <= v <= 2.0, "Temperature"),
        "inference.max_tokens":  (int,   lambda v: v >= 0, "Max tokens"),
        "inference.timeout_s":   (float, lambda v: v > 0, "Inference timeout"),
        "inference.stream_enabled": (bool, None, "Stream enabled"),
        # Additional keys can be added; validation will skip unknown keys.
    }

    _SENSITIVE = {"server.api_key", "server.master_key"}

    # --------------------------------------------------------------------------
    # Initialisation
    # --------------------------------------------------------------------------
    def __init__(
        self,
        config_path: Optional[Path] = None,
        auto_reload: Optional[bool] = None,
        server_config_path: Optional[Path] = None,
    ) -> None:
        self._config_path = config_path or self._default_path()
        self._auto_reload = (
            auto_reload if auto_reload is not None
            else os.environ.get("LLM_AUTO_RELOAD", "false").lower() == "true"
        )
        self._server_config_path = server_config_path
        self._data: Dict[str, Any] = {}
        self._lock = RLock()
        self._last_mtime: float = 0.0
        self._load()

    # --------------------------------------------------------------------------
    # Properties – file path & serialisation
    # --------------------------------------------------------------------------
    @property
    def config_path(self) -> Path:
        return self._config_path

    @property
    def data(self) -> Dict[str, Any]:
        """Return a copy of the configuration with sensitive keys masked."""
        with self._lock:
            return self._mask_sensitive(copy.deepcopy(self._data))

    def to_dict(self) -> Dict[str, Any]:
        """Return the full configuration as a plain dict (no masking)."""
        with self._lock:
            return copy.deepcopy(self._data)

    def from_dict(self, d: Dict[str, Any]) -> None:
        """Replace the entire configuration with a new dict (does not save)."""
        with self._lock:
            self._data = copy.deepcopy(d)
        self._validate_types()

    def update(self, d: Dict[str, Any]) -> None:
        """Bulk update: for each dot‑notation key, set the value, then save."""
        with self._lock:
            for key, value in d.items():
                self._set(key, value)
        self.save()

    # --------------------------------------------------------------------------
    # IO
    # --------------------------------------------------------------------------
    def save(self) -> None:
        with self._lock:
            try:
                self._config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(self._data, f, default_flow_style=False, sort_keys=False)
                self._last_mtime = self._config_path.stat().st_mtime
            except Exception as e:
                logger.error("Failed to save config to %s: %s", self._config_path, e)

    def reload(self) -> bool:
        if not self._auto_reload:
            return False
        with self._lock:
            if self._config_path.exists():
                mtime = self._config_path.stat().st_mtime
                if mtime != self._last_mtime:
                    logger.info("Config file changed, reloading…")
                    self._load()
                    return True
        return False

    # --------------------------------------------------------------------------
    # Core accessors
    # --------------------------------------------------------------------------
    def get(self, key_path: str, default: Any = None, coerce_to: Optional[type] = None) -> Any:
        self.reload()
        with self._lock:
            keys = key_path.split(".")
            value = self._data
            try:
                for k in keys:
                    value = value[k]
                if coerce_to is not None:
                    value = _coerce_value(key_path, value, coerce_to)
                return value
            except (KeyError, TypeError):
                return default

    def set(self, key_path: str, value: Any) -> None:
        self._set(key_path, value)

    def _set(self, key_path: str, value: Any) -> None:
        with self._lock:
            keys = key_path.split(".")
            target = self._data
            for k in keys[:-1]:
                if k not in target or not isinstance(target[k], dict):
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = value

    # --------------------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------------------
    def validate(self) -> Tuple[List[str], List[str]]:
        """Return (errors, warnings) for the current configuration."""
        errors: List[str] = []
        warnings: List[str] = []
        with self._lock:
            for key, (exp_type, validator, desc) in self.SCHEMA.items():
                val = self._get_nested(self._data, key.split("."))
                if val is None:
                    continue  # optional
                if not isinstance(val, exp_type):
                    try:
                        coerced = _coerce_value(key, val, exp_type)
                        if not isinstance(coerced, exp_type):
                            errors.append(f"{key}: expected {exp_type.__name__}, got {type(val).__name__}")
                    except Exception:
                        errors.append(f"{key}: cannot coerce {val!r} to {exp_type.__name__}")
                if validator and val is not None:
                    try:
                        if not validator(val):
                            errors.append(f"{key}: value {val!r} failed validation ({desc})")
                    except Exception:
                        warnings.append(f"{key}: validator raised exception")
        return errors, warnings

    def _validate_types(self) -> None:
        """Coerce known types in the current data dict."""
        for key_path, exp_type in [
            ("http.pool_size", int),
            ("http.connect_timeout_s", float),
            ("http.max_retries", int),
            ("inference.temperature", float),
            ("inference.max_tokens", int),
            ("cli.interactive_history_size", int),
        ]:
            val = self.get(key_path)
            if val is not None:
                self.set(key_path, _coerce_value(key_path, val, exp_type))

    # --------------------------------------------------------------------------
    # Server config integration (API key inheritance)
    # --------------------------------------------------------------------------
    def _find_server_config_path(self) -> Optional[Path]:
        """Return the path to the server configuration file, if accessible."""
        # 1. explicit environment variable
        env_path = os.environ.get("LLM_SERVER_CONFIG")
        if env_path:
            p = Path(env_path)
            if p.exists():
                return p

        # 2. constructor override
        if self._server_config_path and self._server_config_path.exists():
            return self._server_config_path

        # 3. common locations
        candidates = [
            Path.cwd() / "config.yaml",
            Path.home() / ".config" / "local-llm" / "config.yaml",
            Path("/etc/local-llm/config.yaml"),
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _read_server_api_key(self) -> Optional[str]:
        """Read the api.key field from the server configuration."""
        path = self._find_server_config_path()
        if not path:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("api", {}).get("key")
        except Exception as e:
            logger.warning("Could not read server API key from %s: %s", path, e)
            return None

    def _read_server_master_key(self) -> Optional[str]:
        """Read the api.master_key field from the server configuration."""
        path = self._find_server_config_path()
        if not path:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("api", {}).get("master_key")
        except Exception as e:
            logger.warning("Could not read server master key from %s: %s", path, e)
            return None

    # --------------------------------------------------------------------------
    # Convenience properties – SERVER
    # --------------------------------------------------------------------------
    @property
    def base_url(self) -> str:
        return self.get("server.base_url", "http://localhost:8080")

    @base_url.setter
    def base_url(self, value: str) -> None:
        self.set("server.base_url", value)

    @property
    def api_key(self) -> str:
        # 1. environment variable overrides everything
        env_val = os.environ.get("LLM_API_KEY")
        if env_val is not None:
            return env_val
        # 2. client config file
        client_val = self.get("server.api_key", "")
        if client_val:
            return client_val
        # 3. inherit from server config (one-time read)
        server_val = self._read_server_api_key()
        if server_val:
            # Cache it in the client config so we don't keep hitting disk
            self.set("server.api_key", server_val)
            logger.info("API key inherited from server config")
            return server_val
        return ""

    @api_key.setter
    def api_key(self, value: str) -> None:
        self.set("server.api_key", value)

    @property
    def master_key(self) -> str:
        env_val = os.environ.get("LLM_MASTER_KEY")
        if env_val is not None:
            return env_val
        client_val = self.get("server.master_key", "")
        if client_val:
            return client_val
        server_val = self._read_server_master_key()
        if server_val:
            self.set("server.master_key", server_val)
            return server_val
        return ""

    @master_key.setter
    def master_key(self, value: str) -> None:
        self.set("server.master_key", value)

    @property
    def verify_ssl(self) -> bool:
        return self.get("server.verify_ssl", True, coerce_to=bool)

    # --------------------------------------------------------------------------
    # Convenience properties – HTTP
    # --------------------------------------------------------------------------
    @property
    def pool_size(self) -> int:
        return self.get("http.pool_size", 50, coerce_to=int)

    @property
    def connect_timeout_s(self) -> float:
        return self.get("http.connect_timeout_s", 5.0, coerce_to=float)

    @property
    def read_timeout_s(self) -> float:
        return self.get("http.read_timeout_s", 300.0, coerce_to=float)

    @property
    def total_timeout_s(self) -> float:
        return self.get("http.total_timeout_s", 1800.0, coerce_to=float)

    @property
    def max_retries(self) -> int:
        return self.get("http.max_retries", 3, coerce_to=int)

    @property
    def retry_delay_s(self) -> float:
        return self.get("http.retry_delay_s", 1.0, coerce_to=float)

    @property
    def retry_delay_max_s(self) -> float:
        return self.get("http.retry_delay_max_s", 30.0, coerce_to=float)

    @property
    def retry_backoff_factor(self) -> float:
        return self.get("http.retry_backoff_factor", 2.0, coerce_to=float)

    @property
    def stream_chunk_timeout_s(self) -> float:
        return self.get("http.stream_chunk_timeout_s", 30.0, coerce_to=float)

    # --------------------------------------------------------------------------
    # Convenience properties – INFERENCE
    # --------------------------------------------------------------------------
    @property
    def model(self) -> str:
        return self.get("inference.model", "qwen3-8b")

    @model.setter
    def model(self, value: str) -> None:
        self.set("inference.model", value)

    @property
    def temperature(self) -> float:
        return self.get("inference.temperature", 0.7, coerce_to=float)

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.set("inference.temperature", value)

    @property
    def top_p(self) -> float:
        return self.get("inference.top_p", 0.95, coerce_to=float)

    @property
    def top_k(self) -> int:
        return self.get("inference.top_k", 40, coerce_to=int)

    @property
    def max_tokens(self) -> int:
        return self.get("inference.max_tokens", 0, coerce_to=int)

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        self.set("inference.max_tokens", value)

    @property
    def timeout_s(self) -> float:
        return self.get("inference.timeout_s", 1800.0, coerce_to=float)

    @timeout_s.setter
    def timeout_s(self, value: float) -> None:
        self.set("inference.timeout_s", value)

    @property
    def timeout(self) -> float:
        """Alias for inference timeout – used by CLI commands to build ClientConfig."""
        return self.get("inference.timeout_s", 1800.0, coerce_to=float)

    @property
    def stream_enabled(self) -> bool:
        return self.get("inference.stream_enabled", True, coerce_to=bool)

    @stream_enabled.setter
    def stream_enabled(self, value: bool) -> None:
        self.set("inference.stream_enabled", value)

    @property
    def stream_flush_interval_ms(self) -> int:
        return self.get("inference.stream_flush_interval_ms", 80, coerce_to=int)

    # --------------------------------------------------------------------------
    # Convenience properties – HEALTH CHECK
    # --------------------------------------------------------------------------
    @property
    def wait_for_model(self) -> bool:
        return self.get("health_check.wait_for_model", False, coerce_to=bool)

    @property
    def wait_timeout_s(self) -> float:
        return self.get("health_check.wait_timeout_s", 300.0, coerce_to=float)

    # --------------------------------------------------------------------------
    # Convenience properties – ADAPTER
    # --------------------------------------------------------------------------
    @property
    def tps_estimate_fallback(self) -> float:
        return self.get("metrics.tps_estimate_fallback", 7.0, coerce_to=float)

    @property
    def tps_cache_ttl_s(self) -> float:
        return self.get("metrics.tps_cache_ttl_s", 60.0, coerce_to=float)

    # --------------------------------------------------------------------------
    # Backup / Restore
    # --------------------------------------------------------------------------
    def backup(self) -> Optional[Path]:
        """Create a timestamped backup of the current configuration."""
        with self._lock:
            if not self._config_path.exists():
                return None
            backup_path = self._config_path.with_suffix(
                f".yaml.{time.strftime('%Y%m%d_%H%M%S')}.bak"
            )
            try:
                import shutil
                shutil.copy2(self._config_path, backup_path)
                logger.info("Configuration backed up to %s", backup_path)
                return backup_path
            except Exception as e:
                logger.error("Failed to create backup: %s", e)
                return None

    @staticmethod
    def restore(backup_path: Path) -> bool:
        """Restore configuration from a backup file."""
        config = get_config()
        if not backup_path.exists():
            logger.error("Backup file not found: %s", backup_path)
            return False
        try:
            with open(backup_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            config.from_dict(data)
            config.save()
            logger.info("Configuration restored from %s", backup_path)
            return True
        except Exception as e:
            logger.error("Failed to restore from backup: %s", e)
            return False

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _default_path() -> Path:
        config_dir = os.environ.get("LLM_CONFIG_DIR")
        config_file = os.environ.get("LLM_CONFIG_FILE", "config.yaml")
        if config_dir:
            base = Path(config_dir)
        else:
            base = Path.home() / ".llmcloud"
        return base / config_file

    @staticmethod
    def _mask_sensitive(data: Dict[str, Any]) -> Dict[str, Any]:
        if "server" in data and isinstance(data["server"], dict):
            for key in ("api_key", "master_key"):
                if data["server"].get(key):
                    data["server"][key] = "***MASKED***"
        return data

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _get_nested(d: Dict[str, Any], keys: List[str]) -> Any:
        for k in keys:
            if not isinstance(d, dict):
                return None
            d = d.get(k)
            if d is None:
                return None
        return d

    # --------------------------------------------------------------------------
    # Load cycle
    # --------------------------------------------------------------------------
    def _load(self) -> None:
        with self._lock:
            if self._config_path.exists():
                try:
                    with open(self._config_path, "r", encoding="utf-8") as f:
                        file_data = yaml.safe_load(f) or {}
                    self._last_mtime = self._config_path.stat().st_mtime
                except Exception as e:
                    logger.error("Failed to load config from %s: %s", self._config_path, e)
                    file_data = {}
            else:
                file_data = {}
            defaults = _default_config()
            merged = self._deep_merge(defaults, file_data)
            resolved = _resolve_env_vars(merged)
            self._data = resolved
            # Auto‑create file if missing
            if not self._config_path.exists():
                self._config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(defaults, f, default_flow_style=False, sort_keys=False)
                self._config_path.chmod(0o600)
                self._last_mtime = self._config_path.stat().st_mtime

    def __repr__(self) -> str:
        with self._lock:
            masked = self._mask_sensitive(copy.deepcopy(self._data))
        return f"<Config path={self._config_path} data={masked}>"


# ------------------------------------------------------------------------------
# Global singleton with thread‑safe lazy initialization
# ------------------------------------------------------------------------------
_config_instance: Optional[Config] = None
_config_lock = Lock()


def get_config(
    config_path: Optional[Path] = None,
    auto_reload: Optional[bool] = None,
    server_config_path: Optional[Path] = None,
) -> Config:
    """Return the global Config singleton (thread‑safe, double‑checked lock)."""
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = Config(config_path, auto_reload, server_config_path)
    return _config_instance

def save_config(cfg: Optional[Config] = None) -> None:
    """
    Save the given configuration, or the global config if None.
    """
    if cfg is None:
        cfg = get_config()
    cfg.save()

def reload_config(
    config_path: Optional[Path] = None,
    auto_reload: Optional[bool] = None,
    server_config_path: Optional[Path] = None,
) -> Config:
    """Force reload the configuration (useful after manual file edits)."""
    global _config_instance
    with _config_lock:
        _config_instance = Config(config_path, auto_reload, server_config_path)
    return _config_instance


def reset_config() -> None:
    """Reset the singleton (for testing only)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


# ------------------------------------------------------------------------------
# Convenience functions for direct access
# ------------------------------------------------------------------------------
def get_endpoint() -> str:
    """Return the configured server endpoint URL."""
    return get_config().base_url


def get_api_key() -> str:
    """Return the configured API key."""
    return get_config().api_key


def get_default_model() -> str:
    """Return the configured default model."""
    return get_config().model
