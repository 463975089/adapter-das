# SPDX-License-Identifier: Apache-2.0
"""Two-stage gfx938 mHC boundary with the serving BF16 storage boundary.

Adapted from SGLang mhc_boundary_hip.py (Apache-2.0), source SHA256
58ef20e278484a8d8b47f522aa742b8b1757c0fe5bf3e574e100f1bdd2b6f602. Preserve this serving implementation's distinct Post FMA
and BF16 storage boundaries; no Core or provider symbols are changed.
"""
import math

import torch
import triton
import triton.language as tl


def supported(residual, fn, scale, base, previous, x, post_in, comb_in,
              rms_eps, pre_eps, sinkhorn_eps, post_mult, repeat):
    """Bound the serving seam; unsupported callers keep their existing path."""
    tensors = (residual, fn, scale, base, x, post_in, comb_in)
    if (not all(isinstance(value, torch.Tensor) for value in tensors)
            or torch.is_grad_enabled() or torch.is_autocast_enabled('cuda')
            or torch.get_float32_matmul_precision() != 'highest'
            # This kernel shares EPS between pre and Sinkhorn; their equality
            # is an arithmetic constraint, independent of model defaults.
            or pre_eps != sinkhorn_eps
            or not all(isinstance(v, (int, float)) and math.isfinite(v)
                       for v in (rms_eps, pre_eps, sinkhorn_eps, post_mult))
            or rms_eps < 0 or pre_eps <= 0
            or type(repeat) is not int or repeat < 1
            or residual.ndim != 3 or residual.shape[1:] != (4, 5120)
            or residual.shape[0] <= 0):
        return False
    m = residual.shape[0]
    if (residual.dtype != torch.bfloat16 or x.dtype != torch.bfloat16
            or x.shape != (m, 5120) or fn.shape != (24, 20480)
            or fn.dtype != torch.float32 or scale.dtype != torch.float32
            or base.dtype != torch.float32 or scale.shape != (3,) or base.shape != (24,)
            or post_in.shape not in ((m, 4), (m, 4, 1)) or comb_in.shape != (m, 4, 4)
            or post_in.dtype != torch.float32 or comb_in.dtype != torch.float32):
        return False
    if previous is not None:
        if (not isinstance(previous, torch.Tensor) or previous.shape != (m, 4)
                or previous.dtype != torch.float32):
            return False
        tensors += (previous,)
    if not all(value.is_cuda and value.device == residual.device and value.is_contiguous()
               and not value.is_neg() and not value.is_conj() for value in tensors):
        return False
    props = torch.cuda.get_device_properties(residual.device)
    return props.gcnArchName.split(':')[0] == 'gfx938'


@triton.jit
def _mix_row(
    row,
    part_mix_ptr,
    part_sq_ptr,
    scratch_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    m,
    inv_k,
    rms_eps,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    SLICES_PAD: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
):
    """_hc_mix_reduce_sinkhorn_kernel for one row with a tree fixed by NUM_SLICES; scratch_ptr
    ([m, 32] fp32) round-trips the reduced mixes so the sinkhorn starts from a plain layout."""
    j = tl.arange(0, HC)
    jj = j[:, None]
    kk = j[None, :]

    s_idx = tl.arange(0, SLICES_PAD)
    s_ok = s_idx < NUM_SLICES
    off = (s_idx * m + row) * MIX
    a_pre = tl.sum(
        tl.load(
            part_mix_ptr + off[:, None] + j[None, :],
            mask=s_ok[:, None],
            other=0.0,
        ),
        axis=0,
    )
    a_post = tl.sum(
        tl.load(
            part_mix_ptr + off[:, None] + HC + j[None, :],
            mask=s_ok[:, None],
            other=0.0,
        ),
        axis=0,
    )
    a_comb = tl.sum(
        tl.load(
            part_mix_ptr
            + off[:, None, None]
            + 2 * HC
            + jj[None, :, :] * HC
            + kk[None, :, :],
            mask=s_ok[:, None, None],
            other=0.0,
        ),
        axis=0,
    )
    sq = tl.sum(tl.load(part_sq_ptr + s_idx * m + row, mask=s_ok, other=0.0), axis=0)
    sp = scratch_ptr + row * 32
    tl.store(sp + j, a_pre)
    tl.store(sp + HC + j, a_post)
    tl.store(sp + 2 * HC + jj * HC + kk, a_comb)
    tl.debug_barrier()
    a_pre = tl.load(sp + j)
    a_post = tl.load(sp + HC + j)
    a_comb = tl.load(sp + 2 * HC + jj * HC + kk)
    rsqrt = tl.rsqrt(sq * inv_k + rms_eps)

    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)

    pre = tl.sigmoid(a_pre * rsqrt * s0 + tl.load(base_ptr + j)) + EPS
    tl.store(pre_ptr + row * HC + j, pre)
    post = POST_MULT * tl.sigmoid(a_post * rsqrt * s1 + tl.load(base_ptr + HC + j))
    tl.store(post_ptr + row * HC + j, post)

    comb = a_comb * rsqrt * s2 + tl.load(base_ptr + 2 * HC + jj * HC + kk)
    comb = tl.exp(comb - tl.max(comb, axis=1)[:, None])
    comb = comb / tl.sum(comb, axis=1)[:, None] + EPS
    comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)
    for _ in tl.static_range(ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + EPS)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)
    tl.store(comb_ptr + row * HC * HC + jj * HC + kk, comb)


@triton.jit
def _mix_reduce(
    part_mix_ptr,
    part_sq_ptr,
    scratch_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    m,
    inv_k,
    rms_eps,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    SLICES_PAD: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
):
    """One program per row of _mix_row."""
    _mix_row(
        tl.program_id(0),
        part_mix_ptr,
        part_sq_ptr,
        scratch_ptr,
        scale_ptr,
        base_ptr,
        pre_ptr,
        post_ptr,
        comb_ptr,
        m,
        inv_k,
        rms_eps,
        MIX=MIX,
        HC=HC,
        NUM_SLICES=NUM_SLICES,
        SLICES_PAD=SLICES_PAD,
        ITERS=ITERS,
        EPS=EPS,
        POST_MULT=POST_MULT,
    )


@triton.jit
def _boundary_partials(
    x_ptr,
    res_ptr,
    post_in_ptr,
    comb_in_ptr,
    pre_prev_ptr,
    w_ptr,
    res_out_ptr,
    y_ptr,
    part_mix_ptr,
    part_sq_ptr,
    M,
    H,
    x_stride_m,
    res_stride_m,
    w_stride_n,
    HC: tl.constexpr,
    MIX: tl.constexpr,
    MIX_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    HAS_POST: tl.constexpr,
    HAS_COMBINE: tl.constexpr,
):
    """Grid (cdiv(M, BLOCK_M), H // BLOCK_K), one hidden slice of every copy per program. HAS_POST:
    res_out[k] = post[k]*x + sum_j comb[j,k]*res[j] in the serving Torch Post order, read back as bf16
    for the statistics; HAS_COMBINE: y = sum_k pre_prev[k] * copy_k in _hc_combine_kernel's order."""
    tl.static_assert(HC == 4, "the weight tiles are prefetched by name")
    pid_m = tl.program_id(0)
    pid_t = tl.program_id(1)
    # int64: offs_m * (HC * H) passes 2**31 at ~105k rows of H=5120
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    mask_m = offs_m < M
    offs_h = pid_t * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, MIX_PAD)
    mask_n = offs_n < MIX
    m2 = mask_m[:, None]
    w_off = offs_n[None, :] * w_stride_n + offs_h[:, None]
    # Prefetching only reorders memory issue; each tile still feeds its own dot.
    w0 = tl.load(w_ptr + w_off + 0 * H, mask=mask_n[None, :], other=0.0)
    w1 = tl.load(w_ptr + w_off + 1 * H, mask=mask_n[None, :], other=0.0)
    w2 = tl.load(w_ptr + w_off + 2 * H, mask=mask_n[None, :], other=0.0)
    w3 = tl.load(w_ptr + w_off + 3 * H, mask=mask_n[None, :], other=0.0)
    if HAS_POST:
        xa = tl.load(
            x_ptr + offs_m[:, None] * x_stride_m + offs_h[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        r_off = offs_m[:, None] * res_stride_m + offs_h[None, :]
        r0 = tl.load(res_ptr + r_off + 0 * H, mask=m2, other=0.0).to(tl.float32)
        r1 = tl.load(res_ptr + r_off + 1 * H, mask=m2, other=0.0).to(tl.float32)
        r2 = tl.load(res_ptr + r_off + 2 * H, mask=m2, other=0.0).to(tl.float32)
        r3 = tl.load(res_ptr + r_off + 3 * H, mask=m2, other=0.0).to(tl.float32)
        # one tile load per coefficient tensor; a one-hot sum extracts a column exactly
        cj = tl.arange(0, HC)
        post_tile = tl.load(
            post_in_ptr + offs_m[:, None] * HC + cj[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        cc = tl.arange(0, HC * HC)
        comb_tile = tl.load(
            comb_in_ptr + offs_m[:, None] * (HC * HC) + cc[None, :], mask=m2, other=0.0
        ).to(tl.float32)
    if HAS_COMBINE:
        pj = tl.arange(0, HC)
        pre_tile = tl.load(
            pre_prev_ptr + offs_m[:, None] * HC + pj[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        y = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, MIX_PAD], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k in tl.static_range(HC):
        if HAS_POST:
            pk = tl.sum(tl.where((cj == k)[None, :], post_tile, 0.0), axis=1)
            c0 = tl.sum(tl.where((cc == 0 * HC + k)[None, :], comb_tile, 0.0), axis=1)
            c1 = tl.sum(tl.where((cc == 1 * HC + k)[None, :], comb_tile, 0.0), axis=1)
            c2 = tl.sum(tl.where((cc == 2 * HC + k)[None, :], comb_tile, 0.0), axis=1)
            c3 = tl.sum(tl.where((cc == 3 * HC + k)[None, :], comb_tile, 0.0), axis=1)
            # Preserve the serving Torch bmm's +0 and forward FMA order.
            mixed = tl.fma(r0, c0[:, None], 0.)
            mixed = tl.fma(r1, c1[:, None], mixed)
            mixed = tl.fma(r2, c2[:, None], mixed)
            mixed = tl.fma(r3, c3[:, None], mixed)
            # The post product is rounded to FP32 before this separate sum.
            v = mixed + xa * pk[:, None]
            vb = v.to(res_out_ptr.dtype.element_ty)
            tl.store(
                res_out_ptr + offs_m[:, None] * (HC * H) + k * H + offs_h[None, :],
                vb,
                mask=m2,
            )
            xt = vb.to(tl.float32)
        else:
            xt = tl.load(
                x_ptr + offs_m[:, None] * x_stride_m + k * H + offs_h[None, :],
                mask=m2,
                other=0.0,
            ).to(tl.float32)
        if HAS_COMBINE:
            pp = tl.sum(tl.where((pj == k)[None, :], pre_tile, 0.0), axis=1)
            y += pp[:, None] * xt
        if k == 0:
            w_tile = w0
        elif k == 1:
            w_tile = w1
        elif k == 2:
            w_tile = w2
        else:
            w_tile = w3
        acc += tl.dot(xt, w_tile, input_precision=DOT_PRECISION)
        sq += tl.sum(xt * xt, axis=1)
    tl.store(
        part_mix_ptr + (pid_t * M + offs_m[:, None]) * MIX + offs_n[None, :],
        acc,
        mask=m2 & mask_n[None, :],
    )
    tl.store(part_sq_ptr + pid_t * M + offs_m, sq, mask=mask_m)
    if HAS_COMBINE:
        tl.store(
            y_ptr + offs_m[:, None] * H + offs_h[None, :],
            y.to(y_ptr.dtype.element_ty),
            mask=m2,
        )


@torch.inference_mode()
def boundary(residual, fn, scale, base, previous, x=None, post_in=None, comb_in=None,
             rms_eps=1.e-20,
             *, block_k=64, warps=4, post_mult=2.0, eps=1e-6, repeat=20):
    """Return updated residual,post,comb,collapsed BF16 input,and next pre."""
    m, hc, h = residual.shape
    assert residual.is_cuda and residual.is_contiguous() and residual.dtype == torch.bfloat16
    assert hc == 4 and h == 5120 and m > 0
    assert tuple(fn.shape) == (24, 20480) and fn.dtype == torch.float32 and fn.is_contiguous()
    assert scale.shape == (3,) and base.shape == (24,) and scale.dtype == base.dtype == torch.float32
    assert block_k in (64, 128, 256) and warps in (2, 4)
    assert torch.get_float32_matmul_precision() == 'highest' and not torch.is_autocast_enabled('cuda')
    props = torch.cuda.get_device_properties(residual.device)
    assert props.gcnArchName.split(':')[0] == 'gfx938'
    has_post = x is not None
    if has_post:
        assert x.shape == (m,h) and x.dtype == torch.bfloat16 and x.is_contiguous()
        assert post_in.shape in ((m,4),(m,4,1)) and comb_in.shape == (m,4,4)
        assert post_in.dtype == comb_in.dtype == torch.float32
    if previous is not None:
        assert previous.shape == (m,4) and previous.dtype == torch.float32 and previous.is_contiguous()
    tensors = [residual, fn, scale, base] + ([previous] if previous is not None else [])
    tensors += [x,post_in,comb_in] if has_post else []
    assert all(value.device == residual.device and value.is_contiguous() for value in tensors)
    slices = h // block_k
    mix = torch.empty((slices,m,24),device=residual.device,dtype=torch.float32)
    square = torch.empty((slices,m),device=residual.device,dtype=torch.float32)
    updated = torch.empty_like(residual) if has_post else residual
    layer = torch.empty((m,h),device=residual.device,dtype=torch.bfloat16)
    _boundary_partials[(triton.cdiv(m,16),slices)](
        x if has_post else residual,residual,
        post_in if has_post else square,comb_in if has_post else square,
        previous if previous is not None else square,fn,updated,layer,mix,square,
        m,h,x.stride(0) if has_post else residual.stride(0),residual.stride(0),fn.stride(0),
        HC=4,MIX=24,MIX_PAD=32,BLOCK_M=16,BLOCK_K=block_k,DOT_PRECISION='ieee',
        HAS_POST=has_post,HAS_COMBINE=previous is not None,num_warps=warps,
        num_stages=1,enable_fp_fusion=False)
    if previous is None:
        layer = updated[:,0].clone()
    scratch = torch.empty((m,32),device=residual.device,dtype=torch.float32)
    pre = torch.empty((m,4),device=residual.device,dtype=torch.float32)
    post = torch.empty((m,4,1),device=residual.device,dtype=torch.float32)
    comb = torch.empty((m,4,4),device=residual.device,dtype=torch.float32)
    _mix_reduce[(m,)](mix,square,scratch,scale,base,pre,post,comb,m,1./20480,rms_eps,
        MIX=24,HC=4,NUM_SLICES=slices,SLICES_PAD=triton.next_power_of_2(slices),
        ITERS=repeat,EPS=eps,POST_MULT=post_mult,num_warps=4,enable_fp_fusion=False)
    return updated,post,comb,layer,pre
