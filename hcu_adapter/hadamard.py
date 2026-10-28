"""`fast_hadamard_transform` (NVIDIA CUDA extension) adapter for HCU.

sglang's DSA indexer (`dsa_indexer.py: rotate_activation`, the NSA core)
takes the `_is_hip` branch and imports this CUDA-only extension. We serve
`hadamard_transform(x, scale=)` with a device-side torch matmul against a
Sylvester-constructed Walsh-Hadamard matrix (cached per device/dtype/size):
same semantics as the extension (y = scale @ H x, H symmetric), fp32
accumulated, no host syncs (safe under cuda-graph capture).

Registered via `hcu_adapter.register_all()`.
"""

from hcu_adapter._base import LibStubFinder, is_registered

import torch

_HADAMARD_CACHE = {}


def _sylvester_hadamard(n: int, device, dtype):
    """Unnormalized Walsh-Hadamard matrix (entries ±1) for power-of-2 n,
    matching the CUDA extension's semantics: out = scale @ H x."""
    key = (str(device), dtype, n)
    h = _HADAMARD_CACHE.get(key)
    if h is None:
        if n & (n - 1):
            raise NotImplementedError(
                f"hcu-adapter 'fast_hadamard_transform': size {n} is not a"
                " power of 2 (rotationActivation pads upstream; add padding"
                " here if a caller ever needs it)"
            )
        h = torch.ones(1, 1, device=device, dtype=torch.float32)
        while h.shape[0] < n:
            h = torch.cat(
                [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
            )
        _HADAMARD_CACHE[key] = h
        h = h.to(dtype)
    return h


def _hcu_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    shapes_og = x.shape
    n = x.shape[-1]
    flat = x.reshape(-1, n)
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    h = _sylvester_hadamard(n, x.device, x.dtype)
    out = (flat.to(torch.float32) @ h.to(torch.float32)) * float(scale)
    return out.to(x.dtype).reshape(shapes_og)


class _HadamardFinder(LibStubFinder):
    def __init__(self):
        super().__init__(
            "fast_hadamard_transform",
            allowlist={
                "fast_hadamard_transform": {
                    "hadamard_transform": _hcu_hadamard_transform,
                },
            },
        )


def register() -> None:
    if not is_registered(_HadamardFinder, "fast_hadamard_transform"):
        import sys

        sys.meta_path.insert(0, _HadamardFinder())
