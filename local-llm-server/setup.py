#!/usr/bin/env python3
"""
setup.py — legacy shim for local-llm-server
============================================
All project metadata, dependencies, and build configuration now live in
pyproject.toml (build backend: hatchling).

The install.sh hook that used to be wired here via cmdclass
(PostInstallCommand / PostDevelopCommand) is now in hatch_build.py.
Hatchling fires it correctly on `pip install -e .` via PEP 517/660.

This file is kept ONLY as a fallback for tools that still look for setup.py.
Do NOT add logic here — it will be ignored by pip.
"""

from setuptools import setup

setup()
