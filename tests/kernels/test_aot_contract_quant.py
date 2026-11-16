# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""AOT contract anchors: sgl_per_token_group_quant_8bit bridge (C-class).

Contract sources (upstream sgl-kernel v0.5.22 tag):
  - python/sgl_kernel/gemm.py:28-64 — the bridge itself: destination-
    passing (output_q, output_s written in place), enable_v2=None
    auto-selects v2 for group_size in {16,32,64,128}, v1 asserts on
    fuse_silu_and_mul / masked_m, legacy aliases _fp8/_int8.
  - csrc/gemm/per_token_group_quant_8bit.cu:36-115 (v1 kernel):
    absmax floored at eps; y_s = absmax / max_8bit; q = clamp(x / y_s,
    min, max) cast to dst — the int8 cast is a C++ int8_t(float), i.e.
    TRUNCATION toward zero, not rounding. Row-major scales are written
    linearly per group; column-major is stride(0) < stride(1).
    scale_ue8m0: y_s = exp2(ceil(log2(max(y_s, 1e-10)))), stored as the
    biased exponent byte (log2(y_s)+127) packed 4-per-uint32
    (little-endian, column-major only).
  - csrc/gemm/per_token_group_quant_8bit_v2.cu (v2): eps is TORCH_CHECKed
    to 1e-10 (LOCAL_ABSMAX_ABS), min/max CHECK_EQ the dst dtype range,
    group_size switch over {16,32,64,128}; TWO scale directions
    (calculate_fp8_scales): the STORED scale_inv = amax / dst_max is the
    dequant scale, the quantized value MULTIPLIES the reciprocal quant
    scale y_scale = dst_max / amax; ue8m0 uses the DeepEP fast_log2_ceil
    bit trick (exp field + mantissa nonzero) with scale_inv = 2^exp and
    y_scale = 2^-exp, exponent byte = float bits >> 23; column-major
    keeps the LOGICAL axes (token, group) — physical address
    token * 1 + group * stride(-1); fuse_silu_and_mul computes
    silu(primary) cast BACK to the input dtype before the product (amax
    and source both low precision), input (T, 2H) -> output (T, H);
    masked_m is read ON DEVICE (kernel row guard over the static
    DeepEP (E, maxT, H) slab — CUDA-graph capturable), rows
    [masked_m[e], maxT) untouched.

Run (CPU face, any host):  pytest tests/kernels/test_aot_contract_quant.py
(fp8-dtype cases need torch CPU fp8 cast support and skip otherwise.)
"""

import math

import pytest
import torch

import sgl_kernel

has_fp8_cpu = hasattr(torch, "float8_e4m3fn")
requires_fp8 = pytest.mark.skipif(
    not has_fp8_cpu, reason="torch lacks float8_e4m3fn"
)


def _ref_v1(x_flat, group_size, eps, lo, hi):
    g = x_flat.reshape(-1, group_size).float()
    amax = g.abs().amax(dim=1).clamp_min(eps)
    y_s = amax / hi
    q = (g / y_s[:, None]).clamp_(lo, hi)
    return y_s, q


def _trunc_boundaries_ok(q_out, slab_exact, dst_min, dst_max, group_size):
    """Compare a truncating int8 quantizer against a float64-EXACT
    reference, tolerating the integer-boundary ULP only there.

    slab_exact: the quantized face of the input (row-major over tokens,
    hidden). amax is taken PER GROUP (hidden reshaped to (…, H/gs, gs))
    — a whole-slab amax silently works only when hidden == group_size
    and mis-fails everywhere else (caught by the graph-replay case).

    The v2 quantizer multiplies v * fl(dst_max/amax) in fp32. On the
    group-amax lane the exact product is exactly ±dst_max; hipcc and
    torch-CUDA may round that chain one ULP apart (NVIDIA compiles both
    sides to identical div.rn/mul.rn, HIP does not always), and the
    TRUNCATING int8 cast amplifies the ULP into ±1. Bit-exact against
    the float64 truncation everywhere except within 1e-3 of an integer,
    where ±1 is accepted (surfaced while verifying the review #80 P1
    fix on DCU)."""
    g = slab_exact.double().reshape(-1, group_size)
    amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    exact = (g * (float(dst_max) / amax)).clamp(dst_min, dst_max).reshape(-1)
    t = exact.trunc()
    k = q_out.reshape(-1).double()
    near = (exact - exact.round()).abs() < 1e-3
    ok = k == t
    ok |= near & ((k - t).abs() <= 1.0)
    return bool(ok.all())


# ------------------------------------------------------------- bridge face --

def test_bridge_signature_matches_upstream():
    import inspect

    params = list(inspect.signature(
        sgl_kernel.sgl_per_token_group_quant_8bit).parameters)
    assert params == [
        "input", "output_q", "output_s", "group_size", "eps",
        "fp8_min", "fp8_max", "scale_ue8m0", "fuse_silu_and_mul",
        "masked_m", "enable_v2",
    ]
    # legacy aliases (upstream gemm.py:67-68)
    assert sgl_kernel.sgl_per_token_group_quant_fp8 is sgl_kernel.sgl_per_token_group_quant_8bit
    assert sgl_kernel.sgl_per_token_group_quant_int8 is sgl_kernel.sgl_per_token_group_quant_8bit


def test_bridge_v1_rejects_fuse_and_masked():
    x = torch.zeros(4, 64)
    q = torch.zeros(4, 64, dtype=torch.int8)
    s = torch.zeros(4, 4)
    with pytest.raises(AssertionError, match="fuse_silu_and_mul"):
        sgl_kernel.sgl_per_token_group_quant_8bit(
            x, q, s, 64, 1e-10, -128.0, 127.0,
            fuse_silu_and_mul=True, enable_v2=False)
    with pytest.raises(AssertionError, match="masked_m"):
        sgl_kernel.sgl_per_token_group_quant_8bit(
            x, q, s, 64, 1e-10, -128.0, 127.0,
            masked_m=torch.tensor([1]), enable_v2=False)


def test_bridge_v2_enforces_eps_and_range():
    x = torch.randn(4, 128)
    q = torch.zeros(4, 128, dtype=torch.int8)
    s = torch.zeros(4, 2)
    with pytest.raises(RuntimeError, match="eps"):
        sgl_kernel.sgl_per_token_group_quant_8bit(
            x, q, s, 64, 1e-6, -128.0, 127.0)  # v2 auto (64 in whitelist)
    with pytest.raises(RuntimeError, match="match the output"):
        sgl_kernel.sgl_per_token_group_quant_8bit(
            x, q, s, 64, 1e-10, -448.0, 448.0)  # int8 out, fp8 range


def test_bridge_v2_bad_group_size():
    x = torch.randn(4, 128)
    q = torch.zeros(4, 128, dtype=torch.int8)
    s = torch.zeros(4, 256 // 96)
    with pytest.raises(RuntimeError, match="group_size"):
        sgl_kernel.sgl_per_token_group_quant_8bit(
            x, q, torch.zeros(4, 2), 96, 1e-10, -128.0, 127.0,
            enable_v2=True)


# ------------------------------------------------------------------ v1 math --

@pytest.mark.parametrize("dtype_lo,dtype_hi,dst", [
    (-128.0, 127.0, torch.int8),
])
def test_quant_v1_int8_math(dtype_lo, dtype_hi, dst):
    torch.manual_seed(0)
    gs = 64
    x = (torch.randn(6, 128) * 5)
    y_s_ref, q_ref = _ref_v1(x.reshape(-1), gs, 1e-10, dtype_lo, dtype_hi)
    q = torch.zeros(6, 128, dtype=dst)
    s = torch.zeros(6, 2)  # row-major (6, groups_per_row)
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, dtype_lo, dtype_hi, enable_v2=False)
    torch.testing.assert_close(s, y_s_ref.reshape(6, 2), rtol=1e-6, atol=1e-8)
    # int8 cast is truncation (C++ int8_t(float)), mirrored by torch .to
    torch.testing.assert_close(q.float(), q_ref.to(torch.int8).reshape(6, 128).float(),
                               rtol=0, atol=0)


def test_quant_v1_eps_floor():
    # A zero group must produce scale eps/max, not 0 (v1: local_absmax=eps).
    x = torch.zeros(2, 64)
    q = torch.zeros(2, 64, dtype=torch.int8)
    s = torch.zeros(2, 1)
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, 64, 1e-10, -128.0, 127.0, enable_v2=False)
    assert s[0, 0].item() == pytest.approx(1e-10 / 127.0, rel=1e-6)


def test_quant_v1_int8_truncation_semantics():
    # x = 0.5 * scale -> x/y_s = 0.5 -> int8 cast TRUNCATES to 0 (a
    # round()-based quantizer would emit 1 or -1 here).
    gs = 64
    x = torch.zeros(1, gs)
    x[0, 0] = 0.5   # amax 0.5 -> y_s = 0.5/127; this lane = 127 exactly
    x[0, 1] = -0.5
    # craft: value 0.25 -> 0.25 / (0.5/127) = 63.5 -> truncates to 63
    x[0, 2] = 0.25
    q = torch.zeros(1, gs, dtype=torch.int8)
    s = torch.zeros(1, 1)
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, -128.0, 127.0, enable_v2=False)
    assert q[0, 0].item() == 127
    assert q[0, 1].item() == -127
    assert q[0, 2].item() == 63  # 63.5 truncated, not rounded to 64


def test_quant_v1_column_major_scales():
    # Column-major keeps the LOGICAL (token, group) axes; only the strides
    # flip (token stride 1, group stride = tokens). The legal container for
    # a (3-token, 2-group) scale matrix is empty((2, 3)).t() -> shape
    # (3, 2), stride (1, 3). Indexing the buffer as [g, t] walks the wrong
    # axes and throws IndexError (review #80 repro).
    torch.manual_seed(1)
    gs = 64
    x = torch.randn(3, 128)
    y_s_ref, _ = _ref_v1(x.reshape(-1), gs, 1e-10, -128.0, 127.0)
    q = torch.zeros(3, 128, dtype=torch.int8)
    s = torch.zeros(2, 3).t()  # shape (3, 2), stride (1, 3): column-major
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, -128.0, 127.0, enable_v2=False)
    # logical s[t, g] == y_s[t, g] — no transpose on the reference side
    torch.testing.assert_close(s, y_s_ref.reshape(3, 2), rtol=1e-6, atol=1e-8)
    # v2 col-major: same logical layout on the same legal container
    s2 = torch.zeros(2, 3).t()
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s2, gs, 1e-10, -128.0, 127.0)
    torch.testing.assert_close(s2, y_s_ref.reshape(3, 2), rtol=1e-6, atol=1e-8)


# ------------------------------------------------------------------ v2 math --

def test_quant_v2_int8_math_and_floor():
    torch.manual_seed(2)
    gs = 64
    x = torch.randn(5, 128) * 3
    q = torch.zeros(5, 128, dtype=torch.int8)
    s = torch.zeros(5, 2)
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, -128.0, 127.0)  # v2 auto (64 whitelisted)
    g = x.reshape(5, 2, gs).float()
    amax = g.abs().amax(dim=-1).clamp_min(1e-10)
    scale_inv = amax / 127.0           # stored: the DEQUANT direction
    y_scale = 127.0 / amax             # quantizer: the RECIPROCAL
    q_ref = (g * y_scale[..., None]).clamp_(-128.0, 127.0)
    torch.testing.assert_close(s, scale_inv, rtol=1e-6, atol=1e-8)
    assert _trunc_boundaries_ok(q, x.double(), -128.0, 127.0, gs)
    if not q.is_cuda:
        # torch tier == its own fp32 reference chain bit-exact (the
        # boundary helper above is what stays correct for the native
        # tier on device, where hipcc's chain may sit one ULP away)
        torch.testing.assert_close(
            q.float(), q_ref.reshape(5, 128).to(torch.int8).float(),
            rtol=0, atol=0)
    # zero group -> scale floor 1e-10/127
    x0 = torch.zeros(1, gs)
    q0 = torch.zeros(1, gs, dtype=torch.int8)
    s0 = torch.zeros(1, 1)
    sgl_kernel.sgl_per_token_group_quant_8bit(x0, q0, s0, gs, 1e-10, -128.0, 127.0)
    assert s0[0, 0].item() == pytest.approx(1e-10 / 127.0, rel=1e-6)


def test_quant_v2_direction_ones_saturate():
    # The reviewer's P1 repro: x = 1.0 must quantize to the FULL int8
    # range (multiplied by y_scale = 127/amax = 127), not collapse to 0
    # (which is what multiplying by the stored scale_inv = amax/127
    # gives after the truncating cast).
    gs = 64
    x = torch.ones(1, gs)
    q = torch.zeros(1, gs, dtype=torch.int8)
    s = torch.zeros(1, 1)
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, -128.0, 127.0)  # v2 auto (64 whitelisted)
    assert q[0, 0].item() == 127
    assert s[0, 0].item() == pytest.approx(1.0 / 127.0, rel=1e-6)
    # and the fp8 face: 448 full scale
    if has_fp8_cpu:
        q8 = torch.zeros(1, gs, dtype=torch.float8_e4m3fn)
        s8 = torch.zeros(1, 1)
        sgl_kernel.sgl_per_token_group_quant_8bit(
            x, q8, s8, gs, 1e-10, -448.0, 448.0)
        assert q8[0, 0].float().item() == 448.0
        assert s8[0, 0].item() == pytest.approx(1.0 / 448.0, rel=1e-6)


def test_quant_v2_ue8m0_packs_biased_exponents():
    # amax chosen so amax/127 lands strictly between powers of two: the
    # ue8m0 scale must round UP to the next power (ceil-log2) and pack as
    # biased exponent bytes, 4 per uint32, column-major. Legal container
    # for (tokens=2, npacks=2): empty((npacks, tokens)).t() -> stride (1, 2).
    gs = 32
    hidden = 256  # 8 groups per row -> two uint32 packs per row
    amax = 100.0  # ratio = 100/127 ~ 0.787 -> ceil-log2 -> 2^0 = 1
    x = torch.full((2, hidden), amax)
    x[1, 0] = -amax
    q = torch.zeros(2, hidden, dtype=torch.int8)
    s = torch.zeros(2, 2, dtype=torch.int32).t()  # shape (2,2), stride (1,2)
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, -128.0, 127.0, scale_ue8m0=True)
    expect_scale = 1.0  # 2^ceil(log2(0.787))
    expect_byte = int(math.log2(expect_scale)) + 127  # 127
    packed = expect_byte * 0x01010101
    # logical (token, pack): every slot of both axes
    for t in range(2):
        for p in range(2):
            assert s[t, p].item() == packed, f"s[{t},{p}]"
    # quantized: y_scale = 2^-0 = 1 -> amax * 1.0 = 100 -> int8
    assert q[0, 0].item() == 100


def test_quant_v2_ue8m0_nonsquare_column_major():
    # rows(4) != npacks(2) with per-row amax straddling a power of two:
    # pins BOTH the slot layout (logical (token, pack), never a reshape
    # or a transposed axis order — review #80) and the ue8m0 quant
    # direction (y_scale = 2^-exp, so both row kinds quantize to 100;
    # multiplying by the stored scale_inv would clamp the big rows to 127).
    gs = 32
    hidden = 256  # 8 groups per row -> two uint32 packs
    rows = 4      # rows != npacks
    amax_small, amax_big = 100.0, 200.0
    x = torch.empty(rows, hidden)
    for r in range(rows):
        x[r] = amax_big if r % 2 == 1 else amax_small
    q = torch.zeros(rows, hidden, dtype=torch.int8)
    # legal container for (tokens=4, npacks=2): (npacks, tokens).t()
    # -> shape (4, 2), stride (1, 4)
    s = torch.zeros(2, 4, dtype=torch.int32).t()
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, -128.0, 127.0, scale_ue8m0=True)
    byte_small = int(math.log2(1.0)) + 127   # amax 100 -> scale_inv 2^0
    byte_big = int(math.log2(2.0)) + 127     # amax 200 -> ratio 1.575 -> 2^1
    for r in range(rows):
        expect = byte_small * 0x01010101 if r % 2 == 0 else byte_big * 0x01010101
        for p in range(2):
            # int32 holds the uint32 bit pattern: byte 128 sets the sign
            # bit, so compare as unsigned bit patterns, not signed values.
            got = s[r, p].item() & 0xFFFFFFFF
            assert got == expect, f"s[{r},{p}]: {got:#x} != {expect:#x}"
        # small: 100 * 2^0 = 100; big: 200 * 2^-1 = 100 (int8 exact)
        assert q[r, 0].item() == 100, f"row {r}"


def test_quant_v2_fuse_silu_and_mul_low_precision():
    # v2: silu(primary) is cast back to the INPUT dtype before the
    # product; amax and quantized source both come from that product.
    torch.manual_seed(3)
    gs = 64
    dtype = torch.float16
    x = (torch.randn(4, 128) * 3).to(dtype)  # (T, 2H), H=64
    q = torch.zeros(4, 64, dtype=torch.int8)
    s = torch.zeros(4, 1)
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, -128.0, 127.0, fuse_silu_and_mul=True)
    primary = torch.nn.functional.silu(x[..., :64].float()).to(dtype)
    val = (primary * x[..., 64:]).float()
    amax = val.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    scale_inv = amax / 127.0
    torch.testing.assert_close(s, scale_inv, rtol=1e-3, atol=1e-6)
    assert _trunc_boundaries_ok(q, val, -128.0, 127.0, 64)


def test_quant_v2_masked_m_deepep_layout():
    # DeepEP layout: input (E, maxT, H); rows beyond masked_m[e] untouched.
    torch.manual_seed(4)
    gs = 64
    E, maxT, H = 3, 6, 64
    x = torch.randn(E, maxT, H) * 3
    masked = torch.tensor([4, 0, 6], dtype=torch.int32)
    q = torch.full((E, maxT, H), 99, dtype=torch.int8)
    s = torch.zeros(E, maxT, 1)
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, -128.0, 127.0, masked_m=masked)
    for e, m in enumerate([4, 0, 6]):
        if m == 0:
            assert (q[e] == 99).all()  # untouched
            continue
        val = x[e, :m]
        assert _trunc_boundaries_ok(q[e, :m], x[e, :m], -128.0, 127.0, gs)
        if m < maxT:
            assert (q[e, m:] == 99).all()  # padding rows untouched


@requires_fp8
def test_quant_v1_fp8_math():
    torch.manual_seed(5)
    gs = 128
    x = torch.randn(3, 256)
    y_s_ref, q_ref = _ref_v1(x.reshape(-1), gs, 1e-10, -448.0, 448.0)
    q = torch.zeros(3, 256, dtype=torch.float8_e4m3fn)
    s = torch.zeros(3, 2)
    sgl_kernel.sgl_per_token_group_quant_8bit(
        x, q, s, gs, 1e-10, -448.0, 448.0, enable_v2=False)
    torch.testing.assert_close(s, y_s_ref.reshape(3, 2), rtol=1e-6, atol=1e-8)
    # fp8 cast is round-to-nearest (both __nv_cvt and torch CPU)
    torch.testing.assert_close(
        q.float(), q_ref.to(torch.float8_e4m3fn).reshape(3, 256).float(),
        rtol=0, atol=0)
    # dequant sanity: fp8 e4m3 relative precision is 2^-3 per step, so the
    # round-trip error is bounded by ~6.25% relative (half ULP, RN cast)
    deq = (q.float().reshape(6, gs) * y_s_ref[:, None]).reshape(3, 256)
    rel_err = (deq - x).abs().max() / x.abs().max()
    assert rel_err < 0.07


@requires_fp8
def test_quant_v2_fp8_auto_selected():
    # group_size 128 is whitelisted -> v2 without enable_v2=True.
    torch.manual_seed(6)
    gs = 128
    x = torch.randn(2, 256) * 4
    q = torch.zeros(2, 256, dtype=torch.float8_e4m3fn)
    s = torch.zeros(2, 2)
    sgl_kernel.sgl_per_token_group_quant_8bit(x, q, s, gs, 1e-10, -448.0, 448.0)
    g = x.reshape(2, 2, gs).float()
    amax = g.abs().amax(dim=-1).clamp_min(1e-10)
    scale_inv = amax / 448.0
    q_ref = (g * (448.0 / amax)[..., None]).clamp_(-448.0, 448.0)
    torch.testing.assert_close(s, scale_inv, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(
        q.float(), q_ref.reshape(2, 256).to(torch.float8_e4m3fn).float(),
        rtol=0, atol=0)


# ------------------------------------------------------ gpu graph face --

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0),
    reason="requires HCU/CUDA device",
)


@requires_gpu
def test_quant_v2_masked_m_cuda_graph_replay():
    # The DeepEP masked path is on the decode/MTP CUDA-graph route: the
    # call must be capturable WITHOUT a host sync (masked_m read on
    # device) and a replay must re-mask against an UPDATED masked_m —
    # a host .item() read freezes the counts into the captured topology
    # (upstream MaskedLayoutScheduler reads masked_m[expert_idx] in the
    # kernel; review #80).
    E, maxT, H, gs = 3, 8, 128, 64
    torch.manual_seed(7)
    x = torch.randn(E, maxT, H, device="cuda") * 3
    masked = torch.tensor([4, 0, 6], dtype=torch.int32, device="cuda")
    q = torch.full((E, maxT, H), 99, device="cuda", dtype=torch.int8)
    s = torch.zeros(E, maxT, H // gs, device="cuda")

    # warmup on a side stream (graph-capture hygiene)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            sgl_kernel.sgl_per_token_group_quant_8bit(
                x, q, s, gs, 1e-10, -128.0, 127.0, masked_m=masked)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sgl_kernel.sgl_per_token_group_quant_8bit(
            x, q, s, gs, 1e-10, -128.0, 127.0, masked_m=masked)

    def _ref_ok(expert, m):
        return _trunc_boundaries_ok(q[expert, :m], x[expert, :m],
                                    -128.0, 127.0, gs)

    # replay 1 — original counts [4, 0, 6]
    q.fill_(99)
    graph.replay()
    torch.cuda.synchronize()
    assert (q[1] == 99).all()                      # count 0: untouched
    assert (q[0, 4:] == 99).all() and (q[2, 6:] == 99).all()
    assert _ref_ok(0, 4) and _ref_ok(2, 6)

    # replay 2 — counts UPDATED on device: [8, 5, 0]
    masked.copy_(torch.tensor([8, 5, 0], dtype=torch.int32, device="cuda"))
    q.fill_(99)
    graph.replay()
    torch.cuda.synchronize()
    assert (q[2] == 99).all()                      # newly masked-off
    assert (q[1, 5:] == 99).all()                  # new padding rows
    assert _ref_ok(1, 5) and _ref_ok(0, 8)


@requires_gpu
def test_quant_native_rejects_unaddressable_output_s():
    # The native entries validate output_s BEFORE casting its data_ptr into
    # the kernel (review #80 P3): CPU / off-layout / wrong-dtype buffers
    # must raise RuntimeError instead of faulting or silently corrupting
    # the scales, while the legal layouts (column-major .t(), sliced rows
    # with exact row stride) keep flowing through.
    from sgl_kernel._native import native_op

    v1 = native_op("sgl_per_token_group_quant_8bit")
    v2 = native_op("sgl_per_token_group_quant_8bit_v2")
    if v2 is None:
        pytest.skip("native cpp tier not built in this environment")

    x = torch.randn(4, 128, device="cuda")
    q = torch.empty(4, 128, device="cuda", dtype=torch.int8)
    rows, groups = 4, 2

    good_col = torch.zeros(groups, rows, device="cuda").t()      # (1, rows)
    good_sliced = torch.zeros(8, groups, device="cuda")[:rows]   # (2, 1)
    v2(x, q, good_col, 64, 1e-10, -128.0, 127.0, False, False, None)
    v2(x, q, good_sliced, 64, 1e-10, -128.0, 127.0, False, False, None)
    if v1 is not None:
        v1(x, q, good_col, 64, 1e-10, -128.0, 127.0, False)

    bad = [
        ("cpu scales", torch.zeros(rows, groups)),
        ("wrong dtype", torch.zeros(rows, groups, device="cuda", dtype=torch.half)),
        ("padded strides", torch.zeros(rows, 2 * groups, device="cuda")[:, ::2]),
    ]
    for tag, s in bad:
        with pytest.raises(RuntimeError):
            v2(x, q, s, 64, 1e-10, -128.0, 127.0, False, False, None)
        if v1 is not None:
            with pytest.raises(RuntimeError):
                v1(x, q, s, 64, 1e-10, -128.0, 127.0, False)


@requires_gpu
def test_quant_native_rejects_cross_row_and_overlapping_scales():
    # R8 follow-ups to the output_s validator: a last dim that cannot form
    # whole groups must fail cleanly (the size check divides by
    # groups-per-row — a SIGFPE there instead of a TORCH_CHECK), and a
    # column-major buffer with stride(-1) < rows has overlapping slots
    # ((g, r) collides with (g+1, r-s1)) — reject instead of silently
    # corrupting scales.
    from sgl_kernel._native import native_op

    v1 = native_op("sgl_per_token_group_quant_8bit")
    v2 = native_op("sgl_per_token_group_quant_8bit_v2")
    if v2 is None:
        pytest.skip("native cpp tier not built in this environment")

    # (2, 64) with group_size=128: numel divides, but the last dim cannot
    # form whole groups (cross-row grouping is out of contract).
    x = torch.randn(2, 64, device="cuda")
    q = torch.empty(2, 64, device="cuda", dtype=torch.int8)
    s = torch.zeros(1, 1, device="cuda")
    for op, extra in ((v1, ()), (v2, (False, None))):
        with pytest.raises(RuntimeError, match="group_size"):
            op(x, q, s, 128, 1e-10, -128.0, 127.0, False, *extra)

    # column-major with stride(-1)=2 < rows=4: slots alias across groups.
    x = torch.randn(4, 192, device="cuda")
    q = torch.empty(4, 192, device="cuda", dtype=torch.int8)
    s = torch.empty_strided((4, 3), (1, 2), device="cuda")
    for op, extra in ((v1, ()), (v2, (False, None))):
        with pytest.raises(RuntimeError, match=">= rows"):
            op(x, q, s, 64, 1e-10, -128.0, 127.0, False, *extra)
