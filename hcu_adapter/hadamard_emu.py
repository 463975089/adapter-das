# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""torch Hadamard transform for the DSA indexer rotate_activation (is_cuda arm).

Upstream CUDA JIT kernel (sglang.kernels.ops.quantization.hadamard) compiles
CUDA-only warp shuffle intrinsics that the DTK toolchain rejects, and the
HIP arm of the caller imports ``fast_hadamard_transform`` (not shipped on
this wheel set). Provide the reference math: normalized Hadamard matrix
dot-product; exact in fp32 for the 128/256-wide indexer heads. Sizes here are
power-of-2 (contract asserted upstream), so torch.functional-free construction
via the Sylvester recursion is cheap and capture-safe (fixed shapes).
"""

import torch


def _hadamard_matrix(n: int, device, dtype=torch.float32) -> torch.Tensor:
    assert n & (n - 1) == 0 and n > 0
    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    return H


_H_CACHE: dict = {}


def hadamard_transform(x: torch.Tensor, scale: float = 1.0, **_ignored) -> torch.Tensor:
    shapes_og = x.size()
    dim = x.size(-1)
    xf = x.reshape(-1, dim).to(torch.float32)
    H = _H_CACHE.get((dim, xf.device.type, xf.device.index))
    if H is None:
        H = _hadamard_matrix(dim, xf.device)
        _H_CACHE[(dim, xf.device.type, xf.device.index)] = H
    out = xf @ H * scale
    return out.to(x.dtype).reshape(shapes_og)
