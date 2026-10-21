"""tvm-ffi fp8 dtype aliasing for the DSV4 JIT MoE activation kernels.

``silu_and_mul_masked_post_quant`` / ``silu_and_mul_contig_post_quant``
(the between-GEMMs SiLU+per-token-group-quant for the DeepEP fp8 route)
declare their fp8 output tensor as ``uint8`` in the tvm-ffi signature.
On CUDA torch fp8 tensors present as uint8 through DLPack; on this DTK
stack they present as ``float8_e4m3fn`` and tvm-ffi's type match rejects
them ("Dtype value [float8_e4m3fn] not in the allowed options: [uint8]",
DSV4F fp8 dp8ep8 graph capture, nmz16 2026-10-21). Viewing fp8 args as
uint8 before ``module.run`` is byte-identical: the kernel writes raw fp8
bit patterns and the caller keeps consuming its fp8 view of the same
storage.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("hcu.adapter")

_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
)


def _as_uint8(t):
    if torch.is_tensor(t) and t.dtype in _FP8_DTYPES:
        return t.view(torch.uint8)
    return t


def _wrap(orig):
    def inner(*args, **kwargs):
        args = tuple(_as_uint8(a) for a in args)
        kwargs = {k: _as_uint8(v) for k, v in kwargs.items()}
        return orig(*args, **kwargs)

    inner._hcu_fp8_ffi = True
    return inner


def install() -> None:
    from . import hcu_env as _env

    if not _env.enabled("HCU_JIT_FP8_FFI"):
        return

    import sglang.kernels.ops.moe.dsv4 as m

    names = (
        "silu_and_mul_masked_post_quant",
        "silu_and_mul_contig_post_quant",
    )
    patched = []
    for name in names:
        orig = getattr(m, name, None)
        if orig is None or getattr(orig, "_hcu_fp8_ffi", False):
            continue
        wrapped = _wrap(orig)
        setattr(m, name, wrapped)
        patched.append(name)
        # modules that already did `from ...dsv4 import <name>` keep the
        # ORIGINAL binding (e.g. the moe_runner deep_gemm module imported
        # earlier in the adapter chain); rebind every existing holder so
        # the alias does not depend on import order
        import sys

        for mod in list(sys.modules.values()):
            if getattr(mod, name, None) is orig:
                setattr(mod, name, wrapped)

    logger.warning(
        "HCU jit fp8 ffi alias installed (%s: fp8 args viewed uint8 for "
        "tvm-ffi); HCU_JIT_FP8_FFI=0 disables",
        ", ".join(patched) or "nothing to patch",
    )
