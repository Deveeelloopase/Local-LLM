#!/usr/bin/env python3
"""
setup.py — Local LLM Core Production Installer v1.0.0

PRODUCTION SETUP — Dedicated inference engine (GGUF + BitNet)
  - Installs the local_llm_core Python package and all dependencies
  - Registers the `local-llm-setup` command (runs install.sh + bitnet_setup.sh)
  - Compiles the optional _fast CPU accelerator (AVX2 / NEON)

Installation:
    pip install -e .        # Editable install (development)
    pip install .           # Standard install

After install, run the one-time system setup:
    local-llm-setup         # runs install.sh then bitnet_setup.sh
    local-llm-setup --help  # see all options

Environment variables (passed through to the setup scripts):
    LLM_INSTALL_DIR      – override project root
    LLM_VENV_DIR         – override virtual environment path
    LLM_SKIP_KERNEL      – "true" to skip kernel modules (Linux only)
    LLM_MODEL_DIR        – directory for model files
    BITNET_DIR           – override BitNet.cpp install directory
    BITNET_MODEL_DIR     – override BitNet model download directory
    BITNET_QUANT_EMBD    – "1" to enable Q6_K embedding quantization
    BITNET_SKIP_CLONE    – "1" to skip cloning BitNet.cpp (use existing)
    BITNET_SKIP_MODEL    – "1" to skip model download/selection
"""

import os
import re
import sys
from pathlib import Path
from setuptools import setup, find_packages, Extension


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_file(filename: str) -> str:
    filepath = Path(__file__).parent / filename
    return filepath.read_text(encoding="utf-8") if filepath.exists() else ""


def _get_version() -> str:
    init_path = Path(__file__).parent / "local_llm_core" / "__init__.py"
    if init_path.exists():
        content = init_path.read_text(encoding="utf-8")
        match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
        if match:
            return match.group(1)
    return "1.0.0"


# ── _fast C extension (optional AVX2 / NEON accelerator) ──────────────────────

extra_compile_args = ["-O3", "-std=c99"]
extra_link_args: list[str] = []

if sys.platform.startswith("linux"):
    extra_compile_args.extend(["-mavx2", "-mfma", "-funroll-loops"])
elif sys.platform == "darwin":
    extra_compile_args.append("-D_DARWIN_C_SOURCE")
elif sys.platform == "win32":
    extra_compile_args = ["/O2"]

_fast_source = Path(__file__).parent / "local_llm_core" / "_fast.c"
ext_modules = []
if _fast_source.exists():
    ext_modules = [
        Extension(
            "local_llm_core._fast",
            sources=["local_llm_core/_fast.c"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]


# ── Dependencies — NO VERSION PINS ────────────────────────────────────────────

INSTALL_REQUIRES = [
    "llama-cpp-python",
    "transformers",       # No pin — bitnet_setup.sh installs the required fork
    "torch",
    "tokenizers",
    "numpy",
    "pydantic",
    "pyyaml",
    "psutil",
    "prometheus-client",
    "aiofiles",
    "aiosqlite",
    "lz4",
    "structlog",
    "huggingface-hub",
    "safetensors",
    "httpx",
    "sentencepiece",
    "tqdm",
    "packaging",
    "filelock",
    "regex",
]

EXTRAS_REQUIRE: dict[str, list[str]] = {
    "dev": [
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "mypy",
        "ruff",
        "black",
        "pre-commit",
    ],
    "gpu":    ["cuda-python"],
    "vulkan": [],
    "onnx":   ["onnxruntime", "onnx"],
}

EXTRAS_REQUIRE["all"] = sorted(set(
    EXTRAS_REQUIRE["gpu"] +
    EXTRAS_REQUIRE["vulkan"] +
    EXTRAS_REQUIRE["onnx"]
))
EXTRAS_REQUIRE["full"] = sorted(set(
    EXTRAS_REQUIRE["all"] + EXTRAS_REQUIRE["dev"]
))


# ── Setup ──────────────────────────────────────────────────────────────────────

setup(
    name="local-llm-core",
    version=_get_version(),
    author="Deveeelloopase",
    author_email="dev@example.com",
    description="Production-ready LLM inference engine (GGUF & BitNet, unified env)",
    long_description=_read_file("README.md") or "",
    long_description_content_type="text/markdown",
    url="https://github.com/Deveeelloopase/Local-LLM",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "scripts", "docs"]),
    include_package_data=True,
    package_data={
        "local_llm_core": [
            "*.so", "*.dylib", "*.dll", "*.pyd",
            "config.yaml.example",
            "py.typed",
            # Shell scripts bundled so non-editable wheel installs
            # can locate them via local_llm_core/_setup_cli.py
            "scripts/*.sh",
            "scripts/*.py",
        ],
    },
    entry_points={
        "console_scripts": [
            # Registered as a shell command after `pip install local-llm-core`
            # Usage:
            #   local-llm-setup               → full setup (install.sh + bitnet_setup.sh)
            #   local-llm-setup --status      → check if already configured
            #   local-llm-setup --reset       → wipe sentinel and re-run
            #   local-llm-setup --skip-bitnet → install.sh only
            "local-llm-setup=local_llm_core._setup_cli:main",
        ],
    },
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    ext_modules=ext_modules,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm inference gguf bitnet gpu vulkan quantization",
    project_urls={
        "Source": "https://github.com/Deveeelloopase/Local-LLM",
        "Issues": "https://github.com/Deveeelloopase/Local-LLM/issues",
    },
    zip_safe=False,
)
