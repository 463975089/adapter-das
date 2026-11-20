# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DSV4 MoE silu-clamp x per-token-group-128 fp8 quant, fused in ONE triton kernel.

Production pair on HIP (dsv41f decode, triton fused_moe route): adapter triton
``silu_and_mul_clamp`` writes a bf16 intermediate, then
``invoke_fused_moe_kernel`` internally runs ``per_token_group_quant_fp8``
(``_per_token_group_quant_8bit_raw``) on it — two kernels plus a full bf16
write+read round-trip per MoE layer per forward (~263+43 launches/step family).

The upstream fused JIT family (``silu_and_mul_{masked,contig}_post_quant.cuh``)
is NOT usable on gfx938 (unit probes 2026-10-23, validation-host-26):
1. deterministic miscompile: ~3.5% of output elements get wrong values (e.g.
   dequant 99.99 where the true activation is 60.5); deterministic across
   repeated runs, every row, both clamp and no-clamp template variants;
2. (ruled out later: an ABI/e4m3fnuz mismatch was suspected and disproven —
   gfx938 gcnArchName carries no "gfx94" substring, so is_fp8_fnuz() is False
   and the production quant is e4m3fn /448, exactly what the cuh emits.)

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
    GPB: tl.constexpr,  # groups per CTA; GROUP*GPB must be a power of 2
):
    # DSV4/GLM swiglu-clamp (glm5_next.swiglu_clamped semantics: gate caps max
    # only, up clamps both sides) composed with the sglang per-token-group
    # quant formula (eps-max absmax, divide by bit8_max, clamp, cast) in one
    # pass with no bf16 round-trip. One CTA handles GPB adjacent 128-groups
    # (tile sweep validation-host-26 2026-10-23: GPB=1 halves large-M throughput vs the
    # production pair — tiny 512B CTAs starve; GPB=2 keeps the 2.4-2.7x
    # decode win and restores pair-parity at prefill M).
    row = tl.program_id(0).to(tl.int64)
    gb = tl.program_id(1)
    W: tl.constexpr = GROUP * GPB
    cols = gb * W + tl.arange(0, W)
    mask = cols < D
    xb = X + row * x_stride
    gate = tl.load(xb + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(xb + D + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.minimum(gate, limit)
    up = tl.minimum(tl.maximum(up, -limit), limit)
    y = gate * tl.sigmoid(gate) * up
    y2 = tl.reshape(y, (GPB, GROUP))
    _absmax = tl.maximum(tl.max(tl.abs(y2), axis=1), eps)
    y_s = _absmax / bit8_max
    y_q = tl.clamp(y2 * (1.0 / y_s)[:, None], bit8_min, bit8_max)
    tl.store(
        Q + row * q_stride + cols,
        tl.reshape(y_q, (W,)).to(Q.dtype.element_ty),
        mask=mask,
    )
    tl.store(S + row * s_stride + gb * GPB + tl.arange(0, GPB), y_s)


def silu_and_mul_clamp_quant(
    x: torch.Tensor,
    limit: float,
    group_size: int = 128,
    output_q: torch.Tensor | None = None,
    output_s: torch.Tensor | None = None,
):
    """silu(min(gate, limit)) * clamp(up, -limit, limit) -> per-group fp8 + scale."""
    # 替代档: 无（fp8_dtype/fp8_max 常量；模块体导出）
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, fp8_max

    M, N2 = x.shape
    D = N2 // 2
    if D % group_size != 0:
        raise ValueError(f"{D} not divisible by group {group_size}")
    n_groups = D // group_size
    q = output_q if output_q is not None else torch.empty(
        (M, D), device=x.device, dtype=fp8_dtype
    )
    s = output_s if output_s is not None else torch.empty(
        (M, D // group_size), device=x.device, dtype=torch.float32
    )
    gpb = 2 if n_groups % 2 == 0 else 1
    grid = (M, n_groups // gpb)
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
        GPB=gpb,
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

# second branch: the plain-silu arm (swiglu_limit=None — the shared-expert
# fused_experts calls, trace: fused_moe_kernel => act_and_mul_kernel =>
# per_token_group_quant_flat, ~43/step on dsv41f decode)
_ANCHOR2 = """                else:
                    silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
"""

_REPLACEMENT2 = """                else:
                    _hcu_fq = _hcu_silu_fused_quant(
                        intermediate_cache1.view(-1, N),
                        None,
                        use_fp8_w8a8,
                        block_shape,
                        hooks,
                    )
                    if _hcu_fq is not None:
                        intermediate_cache2, a2_scale = _hcu_fq
                    else:
                        silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
"""


# third branch: the EP-filtered routed arm (filter_expert=True whenever
# num_experts != num_local_experts — always true under EP8): stock runs
# act_and_mul_triton (trace: act_and_mul_kernel, 43/step on dsv41f) then a
# separate per-token-group quant before the down GEMM.
_ANCHOR3 = """            else:
                act_and_mul_triton(
                    intermediate_cache1.view(-1, N),
                    intermediate_cache2,
                    config,
                    topk_ids,
                    expert_ids,
                    down_moe_use_tma,
                    activation,
                    swiglu_limit=swiglu_limit_for_triton,
                )
"""

_REPLACEMENT3 = """            else:
                _hcu_fq = _hcu_silu_fused_quant(
                    intermediate_cache1.view(-1, N),
                    swiglu_limit_for_triton,
                    use_fp8_w8a8,
                    block_shape,
                    hooks,
                )
                if _hcu_fq is not None:
                    intermediate_cache2, a2_scale = _hcu_fq
                else:
                    act_and_mul_triton(
                        intermediate_cache1.view(-1, N),
                        intermediate_cache2,
                        config,
                        topk_ids,
                        expert_ids,
                        down_moe_use_tma,
                        activation,
                        swiglu_limit=swiglu_limit_for_triton,
                    )
"""

_route_state = {"miss_logged": False, "hit_logged": False}


def _hcu_silu_fused_quant(x, limit, use_fp8_w8a8, block_shape, hooks):
    """Route guard: only the production shape (fp8 w8a8, [128,128] blocks, no
    LoRA hooks). limit=None means plain silu; anything else unsupported keeps
    the stock two-kernel pair. First miss logs the failing conditions once per
    process (validation-host-18 2026-10-23: the needle sat installed-but-inert because EP
    routing takes the filter_expert arm the original anchors never covered;
    the shared-expert arm's guard also silently rejected — diagnose, don't
    guess)."""
    try:
        if (
            not use_fp8_w8a8
            or tuple(block_shape or ()) != (128, 128)
            or hooks is not None
        ):
            if not _route_state["miss_logged"]:
                _route_state["miss_logged"] = True
                LOG.warning(
                    "HCU silu fused-quant route MISS (once): fp8_w8a8=%s"
                    " block_shape=%r hooks=%r limit=%r x=%s",
                    use_fp8_w8a8, block_shape,
                    type(hooks).__name__ if hooks is not None else None,
                    limit, tuple(x.shape),
                )
            return None
        lim = float("inf") if limit is None else float(limit)
        out = silu_and_mul_clamp_quant(x, lim, 128)
        if not _route_state["hit_logged"]:
            _route_state["hit_logged"] = True
            LOG.warning(
                "HCU silu fused-quant ENGAGED (limit=%r shape=%s)",
                limit, tuple(x.shape),
            )
        return out
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
    if (
        src.count(_ANCHOR) != 1
        or src.count(_ANCHOR2) != 1
        or src.count(_ANCHOR3) != 1
    ):
        raise RuntimeError(
            "dsv4 swiglu activation anchors not found exactly once (source drift?)"
        )
    src = src.replace(_ASSERT_OLD, _ASSERT_NEW, 1)
    src = src.replace(_ANCHOR, _REPLACEMENT, 1)
    src = src.replace(_ANCHOR2, _REPLACEMENT2, 1)
    src = src.replace(_ANCHOR3, _REPLACEMENT3, 1)
    _fm._hcu_silu_fused_quant = _hcu_silu_fused_quant
    exec(compile(src, _fm.__file__, "exec"), _fm.__dict__)
    code = getattr(_fm._fused_moe_kernel_sequence, "__code__", None)
    if code is None or "_hcu_silu_fused_quant" not in code.co_names:
        raise RuntimeError("fused-quant swap did not land in compiled sequence")
    LOG.warning(
        "HCU adapter: DSV4 swiglu-clamp x ptgq fused into one triton kernel "
        "(silu_and_mul_clamp_quant; HCU_SILU_FUSED_QUANT=0 disables)"
    )
