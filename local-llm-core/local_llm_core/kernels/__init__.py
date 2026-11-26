"""
kernels/__init__.py — Low‑Level Kernel Modules
=============================================
This directory contains source code for optional kernels modules (e.g., AMD GPU
compatibility, memory optimizers). These are NOT Python modules and cannot be
imported directly. They are compiled and loaded separately via system tools.

If you are looking for Python APIs, see the `hal` and `memory` subpackages.
"""

# This file intentionally left empty to prevent Python from treating the
# directory as a namespace package.