"""Triton M-grouped FP8 x MXFP4/FP8 contiguous GEMM (pinned DeepGEMM contract, route A fast tier).

Same numerics as hcu_adapter.kernels.gemm.m_grouped_fp8_fp4_gemm_nt_contiguous (the torch reference):
E4M3 activations and E2M1/E4M3 weights times power-of-two UE8M0 scales are exact in
BF16, the products are exact and accumulate in FP32; only the summation order differs.
The torch reference computes every group over all rows and masks (O(G * M)). Here each
program owns one (group, BLOCK_N) weight slice, reads the group's row range from the
layout on device and loops over only those rows, dequantizing weights in registers;
groups without rows exit at once and padding rows keep the zeros written up front.
Device-only (no host reads), graph-safe. Finite inputs only (E4M3 NaN codes decode as 480).
"""

import torch
import triton
import triton.language as tl

# Tiles from tests/tools/_probe_grouped_gemm_bench.py sweeps on BW1101 (V4.1 EP8 gate_up/down
# shapes): BM=32/BN=64/8 warps is best for decode-sized groups and within 25% of the best prefill tile.
_BLOCK_M = 32
_BLOCK_N = 64
_BLOCK_K = 128  # one packed int32 UE8M0 word = 4 x 32 = 128 K values
_NUM_WARPS = 8


@triton.jit
def _pow2(e):
    # 2**e for integer -126 <= e <= 127, built from the exponent field (exact).
    return ((e + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _ue8m0(byte):
    # UE8M0 exponent byte -> 2**(byte - 127); byte 0 is the 2**-127 subnormal.
    return tl.where(byte == 0, 1 << 22, byte << 23).to(tl.float32, bitcast=True)


@triton.jit
def _e4m3(u):
    # OCP E4M3 byte -> float32: (8 + m) * 2**(e - 10), subnormals m * 2**-9.
    e = (u >> 3) & 15
    m = u & 7
    mag = tl.where(e == 0, m.to(tl.float32) * 0.001953125, (8 + m).to(tl.float32) * _pow2(e - 10))
    return tl.where((u & 128) != 0, -mag, mag)


@triton.jit
def _e2m1(c):
    # E2M1 nibble -> float32: (2 + m) * 2**(e - 2), subnormal m * 0.5.
    e = (c >> 1) & 3
    m = c & 1
    mag = tl.where(e == 0, m.to(tl.float32) * 0.5, (2 + m).to(tl.float32) * _pow2(e - 2))
    return tl.where((c & 8) != 0, -mag, mag)


@triton.jit
def _m_grouped_kernel(A, A_SF, B, B_SF, D, ROW_GROUP, G_START, G_END, N, K,
                      s_am, s_ak, s_asm, s_ask,
                      s_bg, s_bn, s_bk, s_bsg, s_bsn, s_bsk,
                      s_dm, s_dn,
                      B_FP4: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr):
    g = tl.program_id(0)
    pid_n = tl.program_id(1)
    start = tl.load(G_START + g)
    end = tl.load(G_END + g)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_ok = cols < N
    kk = tl.arange(0, BLOCK_K)
    pair = tl.arange(0, BLOCK_K // 2)
    shift = ((kk // 32) * 8)[None, :]  # byte of the packed scale word for each K lane
    b_cols = B + g.to(tl.int64) * s_bg + cols[:, None].to(tl.int64) * s_bn
    b_sf_cols = B_SF + g.to(tl.int64) * s_bsg + cols.to(tl.int64) * s_bsn
    for m0 in range(start, end, BLOCK_M):
        rows = m0 + tl.arange(0, BLOCK_M)
        row_ok = rows < end
        a_rows = A + rows[:, None].to(tl.int64) * s_am
        a_sf_rows = A_SF + rows.to(tl.int64) * s_asm
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_ok = (k0 + kk) < K
            a_raw = tl.load(a_rows + (k0 + kk)[None, :] * s_ak, mask=row_ok[:, None] & k_ok[None, :],
                            other=0).to(tl.int32)
            a_word = tl.load(a_sf_rows + (k0 // BLOCK_K) * s_ask, mask=row_ok, other=0)
            a = (_e4m3(a_raw) * _ue8m0((a_word[:, None] >> shift) & 255)).to(tl.bfloat16)
            b_word = tl.load(b_sf_cols + (k0 // BLOCK_K) * s_bsk, mask=col_ok, other=0)
            b_scale = _ue8m0((b_word[:, None] >> shift) & 255)
            if B_FP4:
                packed = tl.load(b_cols + (k0 // 2 + pair)[None, :] * s_bk,
                                 mask=col_ok[:, None] & ((k0 + 2 * pair) < K)[None, :], other=0).to(tl.int32)
                b_val = tl.interleave(_e2m1(packed & 15), _e2m1(packed >> 4))  # low nibble = even K
            else:
                b_val = _e4m3(tl.load(b_cols + (k0 + kk)[None, :] * s_bk, mask=col_ok[:, None] & k_ok[None, :],
                                      other=0).to(tl.int32))
            acc += tl.dot(a, tl.trans((b_val * b_scale).to(tl.bfloat16)))
        keep = row_ok & (tl.load(ROW_GROUP + rows, mask=row_ok, other=-1) == g)
        out = D + rows[:, None].to(tl.int64) * s_dm + cols[None, :] * s_dn
        tl.store(out, acc.to(D.dtype.element_ty), mask=keep[:, None] & col_ok[None, :])


def group_ranges(grouped_layout, m, groups, use_psum_layout, align):
    """(row_group [M] int32 with -1 padding, start [G], end [G]) computed on device."""
    device = grouped_layout.device
    rows = torch.arange(m, device=device)
    if use_psum_layout:
        ends = grouped_layout.long()
        starts = torch.cat((ends.new_zeros(1), (ends[:-1] + align - 1) // align * align))
        g = torch.searchsorted(ends, rows, right=True)
        gc = g.clamp(max=groups - 1)
        valid = (g < groups) & (rows >= starts[gc]) & (rows < ends[gc])
        row_group = torch.where(valid, gc, -1).to(torch.int32)
        return row_group, starts.to(torch.int32), torch.maximum(ends, starts).to(torch.int32)
    row_group = grouped_layout.to(torch.int32).contiguous()
    slot = torch.where(row_group >= 0, row_group.long(), groups)  # padding rows go to a spare slot
    start = torch.full((groups + 1,), m, dtype=torch.long, device=device).scatter_reduce_(0, slot, rows, "amin")
    end = torch.zeros(groups + 1, dtype=torch.long, device=device).scatter_reduce_(0, slot, rows + 1, "amax")
    return row_group, start[:groups].to(torch.int32), end[:groups].to(torch.int32)


def supported(a, b, gran_k_a, gran_k_b):
    (a_x, a_sf), (b_x, b_sf) = a, b
    b_ok = b_x.dtype == torch.int8 or b_x.dtype == torch.float8_e4m3fn
    return (a_x.dtype == torch.float8_e4m3fn and a_sf.dtype == torch.int32 and b_sf.dtype == torch.int32
            and gran_k_a == 32 and gran_k_b == 32 and b_ok and a_x.shape[1] % 64 == 0)


def m_grouped_gemm(a, b, d, grouped_layout, use_psum_layout, align):
    """D[m] = A[m] @ B[g(m)].T; a/b are (data, packed UE8M0 per-32 scales); padding rows of D are zeroed."""
    (a_x, a_sf), (b_x, b_sf) = a, b
    m, k = a_x.shape
    groups, n = b_x.shape[0], b_x.shape[1]
    if m == 0:
        return
    row_group, start, end = group_ranges(grouped_layout, m, groups, use_psum_layout, align)
    d.zero_()
    a_u8 = a_x.view(torch.uint8)
    b_u8 = b_x.view(torch.uint8)
    grid = (groups, triton.cdiv(n, _BLOCK_N))
    with torch.cuda.device(a_x.device):
        _m_grouped_kernel[grid](
            a_u8, a_sf, b_u8, b_sf, d, row_group, start, end, n, k,
            a_u8.stride(0), a_u8.stride(1), a_sf.stride(0), a_sf.stride(1),
            b_u8.stride(0), b_u8.stride(1), b_u8.stride(2), b_sf.stride(0), b_sf.stride(1), b_sf.stride(2),
            d.stride(0), d.stride(1),
            B_FP4=b_x.dtype == torch.int8, BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
            num_warps=_NUM_WARPS)


def m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, grouped_layout, recipe=None, recipe_a=None, recipe_b=None,
                                         compiled_dims="nk", disable_ue8m0_cast=False, use_psum_layout=False,
                                         ensure_zero_padding=True, expected_m_for_psum_layout=None, epilogue=None):
    """Pinned DeepGEMM signature; variants outside the V4.1 per-32 packed-UE8M0 recipe use the torch reference."""
    from .gemm import _recipes
    from .gemm import m_grouped_fp8_fp4_gemm_nt_contiguous as reference
    from .runtime_config import get_mk_alignment_for_contiguous_layout

    if epilogue is not None or disable_ue8m0_cast:
        raise NotImplementedError("HCU grouped GEMM supports the default UE8M0 path without epilogues")
    (gran_m, gran_k_a), (gran_n, gran_k_b) = _recipes(recipe, recipe_a, recipe_b)
    if (gran_m != 1 or gran_n != 1 or d.dtype not in (torch.bfloat16, torch.float32)
            or not supported(a, b, gran_k_a, gran_k_b)):
        return reference(a, b, d, grouped_layout, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast,
                         use_psum_layout, ensure_zero_padding, expected_m_for_psum_layout, epilogue)
    m_grouped_gemm(a, b, d, grouped_layout, use_psum_layout, get_mk_alignment_for_contiguous_layout())
