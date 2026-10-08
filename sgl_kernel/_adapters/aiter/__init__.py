"""`aiter` (AMD ROCm kernel library) adapter for HCU.

Upstream sglang does `if _is_hip: from aiter...` in many module-level shims
(quark mxfp4 schemes, activation AITER backend, assembly attention utils).
On HCU the AMD library is absent, so imports would hard-fail. This adapter
provides:

- a real triton-backed `aiter.silu_and_mul(out, input, limit=0)` matching
  aiter's ROCm signature (limit=0 means no clamp) — exercised by upstream
  parity tests (12/12, see subtask #229);
- working dtype/representation helpers from `aiter.ops.triton.utils` that
  sglang imports for kernels whose bodies never run on HCU;
- loud lazy stubs for everything else (AMD-only assembly kernels, fused MoE,
  paged attention), so unsupported paths fail with a clear message instead of
  a mysterious ImportError.

Registered via `sgl_kernel._adapters.register_all()`.
"""

from sgl_kernel._adapters._base import LibStubFinder, is_registered


def _hcu_silu_and_mul(out, input, limit=0.0):
    """aiter's ROCm silu_and_mul: (out, input, limit); limit=0 = no clamp."""
    from sgl_kernel.elementwise import silu_and_mul

    silu_and_mul(input, out)
    if limit:
        out.clamp_(max=limit)
    return out


def _extra(fullname):
    if fullname == "aiter.ops.triton.utils.types":
        import torch

        return {"e4m3_dtype": getattr(torch, "float8_e4m3fn", torch.float16)}
    if fullname == "aiter.ops.triton.utils._triton.kernel_repr":
        return {"make_kernel_repr": lambda *a, **k: (lambda *ca, **ck: "")}
    if fullname == "aiter":
        return {"silu_and_mul": _hcu_silu_and_mul}
    return {}


class _AiterFinder(LibStubFinder):
    def __init__(self):
        super().__init__("aiter")

    def _extra_attrs(self, fullname):
        return _extra(fullname)


def register() -> None:
    if not is_registered(_AiterFinder, "aiter"):
        import sys

        sys.meta_path.insert(0, _AiterFinder())
