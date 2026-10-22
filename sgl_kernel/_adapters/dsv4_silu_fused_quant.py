"""DSV4 MoE silu-clamp x per-token-group-128 fp8 quant, fused in ONE triton kernel.

Production pair on HIP (dsv41f decode, triton fused_moe route): adapter triton
``silu_and_mul_clamp`` writes a bf16 intermediate, then
``invoke_fused_moe_kernel`` internally runs ``per_token_group_quant_fp8``
(``_per_token_group_quant_8bit_raw``) on it — two kernels plus a full bf16
write+read round-trip per MoE layer per forward (~263+43 launches/step family).

The upstream fused JIT family (``silu_and_mul_{masked,contig}_post_quant.cuh``)
is NOT usable on gfx938 (unit probes 2026-10-23, nmz26):
1. deterministic miscompile: ~3.5% of output elements get wrong values (e.g.
   dequant 99.99 where the true activation is 60.5); deterministic across
   repeated runs, every row, both clamp and no-clamp template variants;
2. ABI mismatch: the family writes e4m3fn with /448 scales while this stack's
   production quant is e4m3fnuz /224 (``fp8_kernel.is_fp8_fnuz`` on gfx94x).

This module's kernel mirrors the production pair exactly (same clamp formula
as ``sgl_kernel.elementwise.silu_and_mul_clamp``; same eps-then-divide-and-
clamp quant as ``_per_token_group_quant_8bit_raw``), reading gate/up straight
from the gateup GEMM output and emitting (fp8, scale) directly.

Wiring: ``install()`` recompiles ``_fused_moe_kernel_sequence`` in place (the
same source-recompile mechanism as the HCU_SWIGLU_CLAMP assert relax) so the
dsv4 swiglu branch calls the fused kernel and feeds (q, s) to the down GEMM
via ``a2_scale`` — ``invoke_fused_moe_kernel`` skips its internal quant when
``A.dtype`` is already fp8. Falls back to the pair whenever the shape is not
the supported one (non-[128,128] blocks, LoRA hooks, non-fp8 route).

Gate: ``HCU_SILU_FUSED_QUANT`` (default on; ``=0`` restores the pair).
"""

from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl

LOG = logging.getLogger("hcu.adapter")


@triton.jit
def _silu_clamp_ptgq_kernel(
    X,  # *input, row-major (M, 2D): gate half then up half
    Q,  # *out fp8 (M, D)
    S,  # *out fp32 (M, D // GROUP)
    D,
    x_stride,
    q_stride,
    s_stride,
    limit,
    eps,
    bit8_min,
    bit8_max,
    GROUP: tl.constexpr,
):
    # DSV4/GLM swiglu-clamp (glm5_next.swiglu_clamped semantics: gate caps max
    # only, up clamps both sides) composed with the sglang per-token-group
    # quant formula (eps-max absmax, divide by bit8_max, clamp, cast) in one
    # pass with no bf16 round-trip.
    row = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    cols = g * GROUP + tl.arange(0, GROUP)
    mask = cols < D
    xb = X + row * x_stride
    gate = tl.load(xb + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(xb + D + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.minimum(gate, limit)
    up = tl.minimum(tl.maximum(up, -limit), limit)
    y = gate * tl.sigmoid(gate) * up
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / bit8_max
    y_q = tl.clamp(y * (1.0 / y_s), bit8_min, bit8_max).to(Q.dtype.element_ty)
    tl.store(Q + row * q_stride + cols, y_q, mask=mask)
    tl.store(S + row * s_stride + g, y_s)


def silu_and_mul_clamp_quant(
    x: torch.Tensor,
    limit: float,
    group_size: int = 128,
    output_q: torch.Tensor | None = None,
    output_s: torch.Tensor | None = None,
):
    """silu(min(gate, limit)) * clamp(up, -limit, limit) -> per-group fp8 + scale."""
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, fp8_max

    M, N2 = x.shape
    D = N2 // 2
    if D % group_size != 0:
        raise ValueError(f"{D} not divisible by group {group_size}")
    q = output_q if output_q is not None else torch.empty(
        (M, D), device=x.device, dtype=fp8_dtype
    )
    s = output_s if output_s is not None else torch.empty(
        (M, D // group_size), device=x.device, dtype=torch.float32
    )
    grid = (M, D // group_size)
    _silu_clamp_ptgq_kernel[grid](
        x,
        q,
        s,
        D,
        x.stride(0),
        q.stride(0),
        s.stride(0),
        float(limit),
        1e-10,
        -float(fp8_max),
        float(fp8_max),
        GROUP=group_size,
        num_warps=1,
        num_stages=1,
    )
    return q, s


# source anchors for the in-place recompile (loud on upstream drift)
_ASSERT_OLD = "assert _is_cuda or _is_xpu, ("
_ASSERT_NEW = "assert _is_cuda or _is_xpu or _is_hip, ("

_ANCHOR = """                if swiglu_limit_for_silu_and_mul_clamp is not None:
                    from sglang.kernels.ops.moe.dsv4 import silu_and_mul_clamp

                    silu_and_mul_clamp(
                        intermediate_cache1.view(-1, N),
                        intermediate_cache2,
                        swiglu_limit_for_silu_and_mul_clamp,
                    )
"""

_REPLACEMENT = """                if swiglu_limit_for_silu_and_mul_clamp is not None:
                    _hcu_fq = _hcu_silu_fused_quant(
                        intermediate_cache1.view(-1, N),
                        swiglu_limit_for_silu_and_mul_clamp,
                        use_fp8_w8a8,
                        block_shape,
                        hooks,
                    )
                    if _hcu_fq is not None:
                        intermediate_cache2, a2_scale = _hcu_fq
                    else:
                        from sglang.kernels.ops.moe.dsv4 import silu_and_mul_clamp

                        silu_and_mul_clamp(
                            intermediate_cache1.view(-1, N),
                            intermediate_cache2,
                            swiglu_limit_for_silu_and_mul_clamp,
                        )
"""


def _hcu_silu_fused_quant(x, limit, use_fp8_w8a8, block_shape, hooks):
    """Route guard: only the dsv41f production shape (fp8 w8a8, [128,128]
    blocks, no LoRA hooks). Anything else keeps the stock two-kernel pair."""
    try:
        if (
            not use_fp8_w8a8
            or hooks is not None
            or tuple(block_shape or ()) != (128, 128)
        ):
            return None
        return silu_and_mul_clamp_quant(x, float(limit), 128)
    except Exception:
        LOG.warning("HCU silu fused-quant route failed; falling back to pair",
                    exc_info=True)
        return None


def install() -> None:
    if os.environ.get("HCU_SILU_FUSED_QUANT") == "0":
        return
    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as _fm

    if getattr(_fm, "_fused_moe_kernel_sequence", None) is None:
        raise RuntimeError("_fused_moe_kernel_sequence not found (source drift?)")
    # Read the ORIGINAL source from the module file: earlier needles (the
    # HCU_SWIGLU_CLAMP assert relax) already recompiled this function in
    # place, so its code object's line info no longer maps to the file and
    # inspect.getsource returns the wrong slice. Slicing the file text at the
    # top-level def also excludes decorators by construction.
    with open(_fm.__file__, encoding="utf-8") as f:
        text = f.read()
    i = text.find("def _fused_moe_kernel_sequence(")
    if i < 0:
        raise RuntimeError("def _fused_moe_kernel_sequence not in module file")
    k = text.find("\ndef ", i + 1)
    k2 = text.find("\nclass ", i + 1)
    k = k if k >= 0 and (k2 < 0 or k < k2) else (k2 if k2 >= 0 else len(text))
    src = text[i:k]
    if src.count(_ANCHOR) != 1:
        raise RuntimeError(
            "dsv4 swiglu clamp call anchor not found exactly once (source drift?)"
        )
    src = src.replace(_ASSERT_OLD, _ASSERT_NEW, 1)
    src = src.replace(_ANCHOR, _REPLACEMENT, 1)
    _fm._hcu_silu_fused_quant = _hcu_silu_fused_quant
    exec(compile(src, _fm.__file__, "exec"), _fm.__dict__)
    code = getattr(_fm._fused_moe_kernel_sequence, "__code__", None)
    if code is None or "_hcu_silu_fused_quant" not in code.co_names:
        raise RuntimeError("fused-quant swap did not land in compiled sequence")
    LOG.warning(
        "HCU adapter: DSV4 swiglu-clamp x ptgq fused into one triton kernel "
        "(silu_and_mul_clamp_quant; HCU_SILU_FUSED_QUANT=0 disables)"
    )
