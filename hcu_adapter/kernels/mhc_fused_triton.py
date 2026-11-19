# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Fuse the delayed mHC pre-mix gate and the residual collapse into one launch.

The adapter's delayed mHC path otherwise launches two adjacent Triton kernels
that share no input:

    _mhc_pre_mix_kernel          gemm/sqrsum          -> next_pre[tokens, hc]
    _hc_head_reduce_store_kernel residual/pre_mix_in  -> layer[tokens, hidden]

Both are per-token kernels, so a single launch can serve both: block 0 of each
token computes the gate while every block performs its slice of the collapse.
This removes one launch per decoder layer; it does not change any arithmetic.

Numerics are held bit-identical to the two originals, which constrains the
launch configuration:

* ``enable_fp_fusion=False`` on both originals, so the FP32 multiply and add
  stay separate here too.
* ``_mhc_pre_mix_kernel`` runs with ``num_warps=1``; ``_hc_head_reduce_store_
  kernel`` with ``num_warps=4``. A fused launch has one value, so it uses 1.
  That is the safe direction: measured on the real kernels, the collapse is
  insensitive to num_warps (bit-identical at 1 and 4), while the gate is not
  (1 vs 4 differs by one FP32 ulp at this point in the split-k reduction).
* The split-k reduction order, the ``hc_mult`` loop order and the FP32->bf16
  store of the collapse all follow the originals exactly.
"""

import torch
import triton
import triton.language as tl

from torch import Tensor

# Mirrors vllm/model_executor/kernels/mhc/triton.py: block_h used by
# hc_collapse_triton's launcher.
BLOCK_H = 1024


@triton.jit
def _pre_mix_collapse_kernel(
    gemm_ptr,
    sqrsum_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    pre_ptr,
    x_ptr,
    next_pre_ptr,
    layer_ptr,
    splitk,
    hc_mult: tl.constexpr,
    hidden_size: tl.constexpr,
    gemm_stride_k,
    gemm_stride_t,
    gemm_stride_j,
    sqrsum_stride_k,
    sqrsum_stride_t,
    pre_stride_t,
    pre_stride_m,
    x_stride_t,
    x_stride_m,
    x_stride_h,
    next_pre_stride_t,
    next_pre_stride_j,
    layer_stride_t,
    layer_stride_h,
    inv_hc_hidden,
    rms_eps,
    hc_pre_eps,
    SPLITK_BLOCK: tl.constexpr,
    HC_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Gate for one token on block 0; collapse slice for every block."""
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)

    if block_idx == 0:
        # --- _mhc_pre_mix_kernel, unchanged arithmetic and reduction order ---
        ks = tl.arange(0, SPLITK_BLOCK)
        js = tl.arange(0, HC_BLOCK)
        kmask = ks < splitk
        jmask = js < hc_mult
        # Only the first hc_mult GEMM columns feed the pre gate.
        gemm = tl.load(
            gemm_ptr
            + ks[:, None] * gemm_stride_k
            + token_idx * gemm_stride_t
            + js[None, :] * gemm_stride_j,
            mask=kmask[:, None] & jmask[None, :],
            other=0.0,
        ).to(tl.float32)
        mixes = tl.sum(gemm, 0)

        sqrsum = tl.load(
            sqrsum_ptr + ks * sqrsum_stride_k + token_idx * sqrsum_stride_t,
            mask=kmask,
            other=0.0,
        ).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(sqrsum, 0) * inv_hc_hidden + rms_eps)

        scale = tl.load(hc_scale_ptr).to(tl.float32)
        base = tl.load(hc_base_ptr + js, mask=jmask, other=0.0).to(tl.float32)

        pre = tl.sigmoid(mixes * rstd * scale + base) + hc_pre_eps
        tl.store(
            next_pre_ptr + token_idx * next_pre_stride_t + js * next_pre_stride_j,
            pre,
            mask=jmask,
        )

    # --- _hc_head_reduce_store_kernel, unchanged arithmetic and loop order ---
    offsets = block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_size

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for mix_idx in tl.static_range(0, hc_mult):
        pre = tl.load(pre_ptr + token_idx * pre_stride_t + mix_idx * pre_stride_m).to(
            tl.float32
        )
        x = tl.load(
            x_ptr
            + token_idx * x_stride_t
            + mix_idx * x_stride_m
            + offsets * x_stride_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += pre * x

    tl.store(
        layer_ptr + token_idx * layer_stride_t + offsets * layer_stride_h,
        acc,
        mask=mask,
    )


def _launch(gemm_out, sqrsum, hc_scale, hc_base, residual, pre_mix, next_pre, layer,
            hc_hidden_size, rms_eps, hc_pre_eps):
    num_tokens, hc_mult, hidden_size = residual.shape
    if num_tokens == 0:
        return
    splitk = gemm_out.shape[0]
    _pre_mix_collapse_kernel[(num_tokens, triton.cdiv(hidden_size, BLOCK_H))](
        gemm_out,
        sqrsum,
        hc_scale,
        hc_base,
        pre_mix,
        residual,
        next_pre,
        layer,
        splitk,
        hc_mult,
        hidden_size,
        gemm_out.stride(0),
        gemm_out.stride(1),
        gemm_out.stride(2),
        sqrsum.stride(0),
        sqrsum.stride(1),
        pre_mix.stride(0),
        pre_mix.stride(1),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        next_pre.stride(0),
        next_pre.stride(1),
        layer.stride(0),
        layer.stride(1),
        1.0 / hc_hidden_size,
        rms_eps,
        hc_pre_eps,
        SPLITK_BLOCK=triton.next_power_of_2(splitk),
        HC_BLOCK=triton.next_power_of_2(hc_mult),
        BLOCK_H=BLOCK_H,
        # num_warps=1 keeps the pre-mix split-k reduction order identical to the
        # separate launch; the collapse is insensitive to it (measured).
        num_warps=1,
        # Both originals keep the FP32 multiply and add separate.
        enable_fp_fusion=False,
    )


def _shape_plan(gemm_out, sqrsum, hc_scale, hc_base, residual, pre_mix):
    """Validate the contract and return (next_pre, layer) shapes."""
    if residual.ndim != 3 or gemm_out.ndim != 3 or pre_mix.ndim != 2:
        raise ValueError('mHC fused gate/collapse expects 3-D residual and gemm')
    num_tokens, hc_mult, hidden_size = residual.shape
    if gemm_out.shape[1] != num_tokens or sqrsum.shape != (gemm_out.shape[0], num_tokens):
        raise ValueError('mHC fused gate/collapse split-k shapes disagree')
    if pre_mix.shape != (num_tokens, hc_mult) or pre_mix.dtype != torch.float32:
        raise ValueError('mHC fused gate/collapse pre_mix must be fp32 [tokens, hc]')
    return (num_tokens, hc_mult), (num_tokens, hidden_size)


def pre_mix_collapse(gemm_out: Tensor, sqrsum: Tensor, hc_scale: Tensor,
                     hc_base: Tensor, residual: Tensor, pre_mix: Tensor,
                     hc_hidden_size: int, rms_eps: float,
                     hc_pre_eps: float) -> tuple[Tensor, Tensor]:
    """Raw Triton entry point; returns (next_pre, layer) as new tensors."""
    (num_tokens, hc_mult), (_, hidden_size) = _shape_plan(
        gemm_out, sqrsum, hc_scale, hc_base, residual, pre_mix)
    next_pre = torch.empty(num_tokens, hc_mult, dtype=torch.float32,
                           device=gemm_out.device)
    layer = torch.empty(num_tokens, hidden_size, dtype=residual.dtype,
                        device=residual.device)
    _launch(gemm_out, sqrsum, hc_scale, hc_base, residual, pre_mix, next_pre, layer,
            hc_hidden_size, rms_eps, hc_pre_eps)
    return next_pre, layer


def pre_mix_collapse_meta(gemm_out: Tensor, sqrsum: Tensor, hc_scale: Tensor,
                          hc_base: Tensor, residual: Tensor, pre_mix: Tensor,
                          hc_hidden_size: int, rms_eps: float,
                          hc_pre_eps: float) -> tuple[Tensor, Tensor]:
    """Fake/meta implementation for the published custom op."""
    (num_tokens, hc_mult), (_, hidden_size) = _shape_plan(
        gemm_out, sqrsum, hc_scale, hc_base, residual, pre_mix)
    return (torch.empty(num_tokens, hc_mult, dtype=torch.float32,
                        device=gemm_out.device),
            torch.empty(num_tokens, hidden_size, dtype=residual.dtype,
                        device=residual.device))


__all__ = ['BLOCK_H', 'pre_mix_collapse', 'pre_mix_collapse_meta']