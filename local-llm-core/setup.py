"""
setup.py — Local LLM Core

All metadata lives in pyproject.toml (single source of truth). This file
exists for exactly ONE job: compiling the optional `_fast` C accelerator
(AVX2 on x86-64 / NEON on ARM), because setuptools>=77 does not yet have
a stable pyproject.toml equivalent for `ext_modules`.

If `local_llm_core/_fast.c` is absent, this file is a no-op and the package
installs as pure Python.
"""
from pathlib import Path
import platform
import sys

from setuptools import Extension, setup


def _build_extensions() -> list[Extension]:
    src = Path(__file__).parent / "local_llm_core" / "_fast.c"
    if not src.exists():
        return []

    if sys.platform == "win32":
        compile_args = ["/O2"]
    elif sys.platform.startswith("linux"):
        if platform.machine().lower() in ("x86_64", "amd64"):
            compile_args = ["-O3", "-std=c99", "-mavx2", "-mfma", "-funroll-loops"]
        else:                       # aarch64 etc.: NEON is baseline, no x86 flags
            compile_args = ["-O3", "-std=c99", "-funroll-loops"]
    elif sys.platform == "darwin":
        compile_args = ["-O3", "-std=c99", "-D_DARWIN_C_SOURCE"]
    else:
        compile_args = ["-O3", "-std=c99"]

    return [
        Extension(
            "local_llm_core._fast",
            sources=["local_llm_core/_fast.c"],
            extra_compile_args=compile_args,
        )
    ]


setup(ext_modules=_build_extensions())
