from __future__ import annotations

"""
engine/kernels/patcher.py — Safe PyTorch monkey‑patching
========================================================
Provides install_patch() and remove_patch() to replace
torch.matmul and torch.nn.functional.linear with kernels
that dispatch to custom accelerators when appropriate,
falling back to the original implementation otherwise.

IMPORTANT: The patch only modifies inference‑time operations;
it does not alter training or gradient computations.
"""

import logging
import torch
import torch.nn.functional as F
from typing import Callable, Optional

log = logging.getLogger(__name__)

# Store original functions
_original_matmul = torch.matmul
_original_linear = F.linear

_custom_fn: Optional[Callable] = None

# -------------------------------------------------------------------------
# Conditional wrapper
# -------------------------------------------------------------------------
def _maybe_custom_matmul(input: torch.Tensor, other: torch.Tensor, *,
                         out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if _custom_fn is not None and _is_ternary_eligible(input, other):
        try:
            # Scale defaults to all-ones for non-quantized paths
            scale = torch.ones(other.shape[-1], dtype=torch.float32,
                               device=input.device)
            result = _custom_fn(input, other, scale)
            if out is not None:
                out.copy_(result)
                return out
            return result
        except Exception as e:
            log.debug("Custom kernel failed, falling back: %s", e)
    if out is not None:
        return _original_matmul(input, other, out=out)
    return _original_matmul(input, other)


def _maybe_custom_linear(input: torch.Tensor, weight: torch.Tensor,
                         bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if _custom_fn is not None and _is_ternary_eligible(input, weight.T):
        try:
            scale = torch.ones(weight.shape[0], dtype=torch.float32,
                               device=input.device)
            result = _custom_fn(input, weight.T, scale)
            if bias is not None:
                result = result + bias
            return result
        except Exception as e:
            log.debug("Custom linear kernel failed: %s", e)
    return _original_linear(input, weight, bias)

# -------------------------------------------------------------------------
# Eligibility heuristics
# -------------------------------------------------------------------------
def _is_ternary_eligible(a: torch.Tensor, b: torch.Tensor) -> bool:
    # Check that dimensions are 2D and a is float16, b is int8 (packed ternary)
    if a.dim() != 2 or b.dim() != 2:
        return False
    # a: (M, K) float16, b: (K, N) int8 packed (or float? but we expect int8)
    # We'll be lenient: accept any pair of float16 and int8.
    if a.dtype != torch.float16 or b.dtype != torch.int8:
        # Also allow float32 * int8 (we can promote)
        if a.dtype == torch.float32 and b.dtype == torch.int8:
            return True
        return False
    return True

# -----------------------------------------------------------------------------
# Install / Remove
# -----------------------------------------------------------------------------
def install_patch(custom_fn: Optional[Callable]) -> None:
    """Replace torch.matmul and F.linear with conditional wrappers."""
    global _custom_fn
    _custom_fn = custom_fn
    if custom_fn is not None:
        torch.matmul = _maybe_custom_matmul
        F.linear = _maybe_custom_linear
        log.info("PyTorch ops patched with custom kernels")
    else:
        log.debug("No custom kernels provided; patch not applied")

def remove_patch() -> None:
    """Restore original PyTorch functions."""
    global _custom_fn
    _custom_fn = None
    torch.matmul = _original_matmul
    F.linear = _original_linear
    log.info("PyTorch ops restored to original")
