"""
local_llm_client/web/__init__.py

Web UI and web‑related utilities for Local LLM Cloud client.
"""

from .web_ui import HTML

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "HTML",
    "get_web_ui",
]


def get_web_ui() -> str:
    """Return the HTML string of the web UI."""
    return HTML