#!/usr/bin/env python3
"""
setup.py — Local LLM Client Production‑Grade Installer
======================================================
Fully self‑contained installer for the Local LLM Cloud client.
Communicates with the server exclusively over HTTP/WebSocket – no
local‑llm‑core or local‑llm‑server dependencies are required.

Installation:
    pip install -e .                  # Editable (development)
    pip install .                     # Standard install (production)
    pip install .[dev]                # With development tools
    pip install .[perf]               # With performance extras (aiohttp, cffi)
    pip install .[all]                # Everything

This script is compatible with setuptools ≥ 64 and Python ≥ 3.10.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from setuptools import setup, find_packages

# ------------------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------------------
def read_file(filename: str) -> str:
    """Read a file relative to this setup.py safely."""
    filepath = Path(__file__).parent / filename
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")
    return ""


def get_version() -> str:
    """
    Extract the version string from local_llm_client/__init__.py.
    Returns '0.0.0' if the version cannot be determined.
    """
    init_path = Path(__file__).parent / "local_llm_client" / "__init__.py"
    if not init_path.exists():
        return "0.0.0"
    content = init_path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    return match.group(1) if match else "0.0.0"


# ------------------------------------------------------------------------------
# Core dependencies – minimal set that works on any platform
# ------------------------------------------------------------------------------
INSTALL_REQUIRES: List[str] = [
    "httpx",            # Async HTTP client (preferred)
    "anyio",            # Async concurrency layer
    "click",            # CLI framework
    "rich",             # Terminal formatting, tables, progress
    "pyyaml",           # YAML configuration parsing
    "pydantic",         # Data validation & settings
    "psutil",           # System metrics
    "urllib3",          # HTTP backend for synchronous client
]

# ------------------------------------------------------------------------------
# Optional extras – enable specific features
# ------------------------------------------------------------------------------
EXTRAS_REQUIRE: Dict[str, List[str]] = {
    # Development tools: testing, linting, type checking
    "dev": [
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "mypy",
        "ruff",
        "black",
    ],
    # Performance: faster async networking + C token counter
    "perf": [
        "aiohttp",   # Alternative async HTTP (faster streaming)
        "cffi",      # C accelerator for token counting / word boundaries
    ],
    # Separate extra for just aiohttp
    "aiohttp": [
        "aiohttp",
    ],
    # Separate extra for just the C accelerator
    "cffi": [
        "cffi",
    ],
    # HTTP/2 support (experimental)
    "http2": [
        "httpx[http2]",
    ],
    # Metrics export (Prometheus)
    "prometheus": [
        "prometheus-client",
    ],
    # OpenTelemetry tracing
    "otel": [
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-exporter-otlp",
    ],
    # Syntax‑highlighted JSON in the CLI
    "pygments": [
        "pygments",
    ],
}

# "all" extra – everything except heavy dev‑only tools
EXTRAS_REQUIRE["all"] = sorted({
    pkg for extra in ["perf", "aiohttp", "cffi", "http2", "prometheus", "otel", "pygments"]
    for pkg in EXTRAS_REQUIRE.get(extra, [])
})


# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------
setup(
    name="local-llm-client",
    version=get_version(),
    author="Deveeelloopase",
    author_email="dev@example.com",
    description="Standalone SDK and CLI client for Local LLM Cloud inference",
    long_description=read_file("README.md") or "Local LLM Client – Standalone inference client",
    long_description_content_type="text/markdown",
    url="https://github.com/Deveeelloopase/LLM-Local",
    license="MIT",

    # Package discovery – include the whole local_llm_client package
    packages=find_packages(
        where=".",
        include=["local_llm_client", "local_llm_client.*"],
    ),
    include_package_data=True,
    package_data={
        "local_llm_client": ["config.yaml", "py.typed"],
        "local_llm_client.web": ["*.html", "*.css", "*.js"],
    },

    # Python version requirement
    python_requires=">=3.10,<4.0",

    # Dependencies
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,

    # Entry points (CLI commands)
    entry_points={
        "console_scripts": [
            "llm = local_llm_client.cli.main:cli",
            "local-llm = local_llm_client.cli.main:cli",
            "llm-install = local_llm_client.cli.install_cmd:main",
        ],
    },

    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities",
    ],

    keywords="llm, client, sdk, cli, ai, inference, local, private, chatbot",

    project_urls={
        "Source": "https://github.com/Deveeelloopase/LLM-Local",
        "Issues": "https://github.com/Deveeelloopase/LLM-Local/issues",
        "Documentation": "https://github.com/Deveeelloopase/LLM-Local/blob/master/local-llm-client/README.md",
        "Changelog": "https://github.com/Deveeelloopase/LLM-Local/releases",
    },

    # Package settings
    zip_safe=False,
)
