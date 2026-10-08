"""External-library adapters for HCU.

sglang's HIP code paths import several AMD/NVIDIA-only kernel libraries at
module level. Each subpackage here installs a compatibility adapter for one
of them, so upstream code runs unmodified on HCU:

    aiter        AMD aiter (real triton-backed silu_and_mul + lazy stubs)
    flashinfer   NVIDIA flashinfer (lazy stubs; attention uses triton backend)
    flash_attn   flash-attn / vLLM FA3 (lazy stubs; same)

All adapters are meta-path based: `from <lib> import <name>` succeeds at
import time (upstream shims bind eagerly), while *calling* an unimplemented
entry raises a loud NotImplementedError naming the missing kernel. Entries
that have device-agnostic semantics are served by our triton kernels
(currently `aiter.silu_and_mul`).

Registered by `sgl_kernel.__init__` via `_adapters.register_all()`.
"""

from sgl_kernel._adapters import flash_attn as _fa
from sgl_kernel._adapters import flashinfer as _fi
from sgl_kernel._adapters import aiter as _aiter


def register_all() -> None:
    """Idempotently install every adapter's meta-path finder."""
    _aiter.register()
    _fi.register()
    _fa.register()
