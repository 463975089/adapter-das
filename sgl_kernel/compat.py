# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Environment detection and small shared helpers for the HCU sgl_kernel port.

The package targets HCU (Hygon HCU, DTK/ROCm stack), where torch reports
HIP (`torch.version.hip is not None`) but CUDA-specific extensions from the
official sgl-kernel wheel are unavailable. All kernels here are pure triton,
so they compile and run on any triton-supported backend (CUDA, HIP/HCU).
"""

from __future__ import annotations

HAS_TRITON = False
try:
    import triton as _triton

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only on CPU-only hosts
    _triton = None

# libdevice moved around across triton versions; erf is all we need.
if HAS_TRITON:
    try:  # triton >= 2.1 standard layout
        from triton.language.extra.libdevice import erf as _tl_erf
    except ImportError:
        try:
            from triton.language.math import erf as _tl_erf  # older layout
        except ImportError:
            try:  # HIP-flavoured triton (incl. dtk port)
                from triton.language.extra.cuda.libdevice import erf as _tl_erf
            except ImportError:
                _tl_erf = None
else:
    _tl_erf = None

HAS_ERF = _tl_erf is not None


def next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n - 1).bit_length())


def check_cuda_tensor(*tensors) -> None:
    """Raise a clear error when a kernel gets CPU tensors (native fallback
    paths handle CPU, triton paths cannot)."""
    for t in tensors:
        if t is not None and not t.is_cuda:
            raise ValueError(
                f"sgl_kernel triton kernels require CUDA/HCU tensors, got {t.device}. "
                "This build replaces the official sgl-kernel AOT backend on HCU."
            )
