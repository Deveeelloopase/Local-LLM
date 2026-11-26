"""
local_llm_server/serving/__init__.py — Serving Layer v1.0.0
=============================================================
Public interface for the unified FastAPI server and all supporting components.

  • Temporary YAML file created when a config dict is supplied is now
    automatically deleted after use, preventing disk pollution.
  • The shutdown helper now explicitly handles None components gracefully
    (already did, but documentation clarified).
  • All existing v4.0.0 features retained; fully backward compatible.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

# Lazy import cache – shared across all workers in the same process.
_MODULE_CACHE: Dict[str, Any] = {}
_log = logging.getLogger(__name__)

__version__ = "1.0.0"

# ------------------------------------------------------------------------------
# Type stubs (for IDEs) – never executed at runtime
# ------------------------------------------------------------------------------
if TYPE_CHECKING:
    from .api_server import create_app, app as fastapi_app
    from .middleware import (
        ProductionMiddleware,
        IPRateLimiter,
        create_ip_rate_limiter,
        install_request_id_filter,
    )
    from .log_handler import (
        TUILogHandler,
        create_log_handler,
        get_log_handler,
    )

    from .diagnose_web import main as diagnose_web_main
    from .models import (
        ChatRequest,
        ChatResponse,
        ChatMessage,
        CompletionRequest,
        CompletionResponse,
        LoadModelRequest,
        UnloadModelRequest,
        ModelInfo,
        ModelList,
        HealthResponse,
        MetricsResponse,
        ErrorResponse,
        ErrorDetail,
        FinishReason,
        RoleEnum,
        SeverityEnum,
        LogLevelEnum,
        ModelType,
    )


# ------------------------------------------------------------------------------
# Lazy attribute access (cached, thread‑safe enough for typical use)
# ------------------------------------------------------------------------------
def __getattr__(name: str):
    """Lazy load module attributes, caching after first access."""
    # Application
    if name == "create_app":
        if "api_server" not in _MODULE_CACHE:
            from .api_server import create_app
            _MODULE_CACHE["api_server"] = create_app
        return _MODULE_CACHE["api_server"]
    if name == "app":
        if "app" not in _MODULE_CACHE:
            from .api_server import app
            _MODULE_CACHE["app"] = app
        return _MODULE_CACHE["app"]

    # Middleware
    if name == "ProductionMiddleware":
        if "ProductionMiddleware" not in _MODULE_CACHE:
            from .middleware import ProductionMiddleware
            _MODULE_CACHE["ProductionMiddleware"] = ProductionMiddleware
        return _MODULE_CACHE["ProductionMiddleware"]
    if name == "IPRateLimiter":
        if "IPRateLimiter" not in _MODULE_CACHE:
            from .middleware import IPRateLimiter
            _MODULE_CACHE["IPRateLimiter"] = IPRateLimiter
        return _MODULE_CACHE["IPRateLimiter"]
    if name == "create_ip_rate_limiter":
        if "create_ip_rate_limiter" not in _MODULE_CACHE:
            from .middleware import create_ip_rate_limiter
            _MODULE_CACHE["create_ip_rate_limiter"] = create_ip_rate_limiter
        return _MODULE_CACHE["create_ip_rate_limiter"]
    if name == "install_request_id_filter":
        if "install_request_id_filter" not in _MODULE_CACHE:
            from .middleware import install_request_id_filter
            _MODULE_CACHE["install_request_id_filter"] = install_request_id_filter
        return _MODULE_CACHE["install_request_id_filter"]

    # Log handler
    if name == "TUILogHandler":
        if "TUILogHandler" not in _MODULE_CACHE:
            from .log_handler import TUILogHandler
            _MODULE_CACHE["TUILogHandler"] = TUILogHandler
        return _MODULE_CACHE["TUILogHandler"]
    if name == "create_log_handler":
        if "create_log_handler" not in _MODULE_CACHE:
            from .log_handler import create_log_handler
            _MODULE_CACHE["create_log_handler"] = create_log_handler
        return _MODULE_CACHE["create_log_handler"]
    if name == "get_log_handler":
        if "get_log_handler" not in _MODULE_CACHE:
            from .log_handler import get_log_handler
            _MODULE_CACHE["get_log_handler"] = get_log_handler
        return _MODULE_CACHE["get_log_handler"]

    # Diagnostics
    if name == "diagnose_web_main":
        if "diagnose_web_main" not in _MODULE_CACHE:
            from .diagnose_web import main as diagnose_web_main
            _MODULE_CACHE["diagnose_web_main"] = diagnose_web_main
        return _MODULE_CACHE["diagnose_web_main"]

    # Models – cache the whole module
    if name in {
        "ChatRequest", "ChatResponse", "ChatMessage",
        "CompletionRequest", "CompletionResponse",
        "LoadModelRequest", "UnloadModelRequest",
        "ModelInfo", "ModelList",
        "HealthResponse", "MetricsResponse",
        "ErrorResponse", "ErrorDetail",
        "FinishReason", "RoleEnum", "SeverityEnum", "LogLevelEnum",
        "ModelType",
    }:
        if "models_module" not in _MODULE_CACHE:
            import local_llm_server.serving.models as _models
            _MODULE_CACHE["models_module"] = _models
        else:
            _models = _MODULE_CACHE["models_module"]
        return getattr(_models, name)

    raise AttributeError(f"module 'local_llm_server.serving' has no attribute {name!r}")


# ------------------------------------------------------------------------------
# Exports (for documentation and IDE discovery)
# ------------------------------------------------------------------------------
__all__ = [
    "create_app",
    "ProductionMiddleware",
    "IPRateLimiter",
    "create_ip_rate_limiter",
    "install_request_id_filter",
    "TUILogHandler",
    "create_log_handler",
    "get_log_handler",
    "diagnose_web_main",
    "ChatRequest",
    "ChatResponse",
    "ChatMessage",
    "CompletionRequest",
    "CompletionResponse",
    "LoadModelRequest",
    "UnloadModelRequest",
    "ModelInfo",
    "ModelList",
    "HealthResponse",
    "MetricsResponse",
    "ErrorResponse",
    "ErrorDetail",
    "FinishReason",
    "RoleEnum",
    "SeverityEnum",
    "LogLevelEnum",
    "ModelType",
    "get_serving_components",
    "shutdown_serving_components",
]


# ------------------------------------------------------------------------------
# Convenience initializer
# ------------------------------------------------------------------------------
def get_serving_components(
    config_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any, Any, Any]:
    """
    Initialize and return the core serving components.

    Args:
        config_path: Path to config.yaml. If None, default locations are tried.
        config: If a configuration dictionary is provided *instead* of a file path,
                it will be written to a temporary YAML file and that path will be
                passed to the internal factories.  The temporary file is deleted
                after the app is created (no leftovers).

    Returns:
        (FastAPI app, IPRateLimiter, WebSocketManager, LoadBalancer)
    """
    from .api_server import create_app
    from .middleware import create_ip_rate_limiter

    # Resolve config to a path that create_app can consume
    effective_path: Optional[str] = config_path
    temp_config_file: Optional[str] = None
    try:
        if config is not None:
            # Write dictionary to a temporary file – create_app expects a file path.
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, prefix="llm_cloud_"
            )
            try:
                import yaml
                yaml.safe_dump(config, tmp)
                tmp.flush()   # ensure data written before we pass the path
                tmp_name = tmp.name
                temp_config_file = tmp_name  # remember for cleanup
                effective_path = tmp_name
            finally:
                tmp.close()  # close the file so create_app can open it on disk
            _log.debug("Temporary config created at %s", temp_config_file)

        if effective_path:
            path_to_use = effective_path
        else:
            # Try common default locations
            candidates = [str(Path.home() / "config.yaml"), str(Path.cwd() / "config.yaml")]
            path_to_use = None
            for p in candidates:
                if Path(p).exists():
                    path_to_use = p
                    break

        if path_to_use:
            app = create_app(path_to_use)
        else:
            app = create_app()  # uses "config.yaml" default

        # For rate limiter and others, pass the loaded config dict if available;
        # otherwise the factory will load from file.
        if config is not None:
            rate_limiter = create_ip_rate_limiter(config)
        else:
            rate_limiter = create_ip_rate_limiter()

        return rate_limiter

    finally:
        # Clean up temporary file if it was created
        if temp_config_file is not None:
            try:
                os.unlink(temp_config_file)
                _log.debug("Temporary config file %s removed.", temp_config_file)
            except OSError as e:
                _log.warning("Failed to remove temporary config file %s: %s", temp_config_file, e)


async def shutdown_serving_components(
    app: Any = None,
    rate_limiter: Any = None,
    websocket_manager: Any = None,
    load_balancer: Any = None,
) -> None:
    """
    Gracefully shut down all serving components.

    This is a helper for FastAPI lifespan shutdown events. All parameters are
    optional and safely ignored when None.
    """
    tasks = []
    if rate_limiter is not None and hasattr(rate_limiter, "shutdown"):
        tasks.append(rate_limiter.shutdown())
    if websocket_manager is not None and hasattr(websocket_manager, "shutdown"):
        tasks.append(websocket_manager.shutdown())
    if load_balancer is not None and hasattr(load_balancer, "shutdown"):
        tasks.append(load_balancer.shutdown())
    # The app itself doesn't need explicit shutdown, but middleware's rate limiter is
    # handled via the rate_limiter object.
    if tasks:
        import asyncio
        await asyncio.gather(*tasks, return_exceptions=True)
    _log.info("Serving components shut down.")


# ------------------------------------------------------------------------------
# Module initialization logging
# ------------------------------------------------------------------------------
_log.debug("serving package loaded (v%s)", __version__)
