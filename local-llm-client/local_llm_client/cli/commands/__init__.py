"""
local_llm_client/cli/commands/__init__.py

CLI command implementations for Local LLM Cloud.

Exports:
    - __version__: Package version.
    - chat, models, keys, status, config_cmd: Click command groups.
    - get_all_commands(): Returns list of all command groups.
"""

from .chat import chat
from .models import models
from .keys import keys
from .status import status
from .config_cmd import config_cmd

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "chat",
    "models",
    "keys",
    "status",
    "config_cmd",
    "get_all_commands",
]


def get_all_commands():
    """
    Return a list of all available command groups.

    Useful for programmatic registration or testing.
    """
    return [chat, models, keys, status, config_cmd]
