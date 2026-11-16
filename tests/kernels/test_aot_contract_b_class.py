# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""AOT contract anchors, B-class promotions (emu stub -> production tiers).

  merge_state_v2      upstream csrc/attention/merge_attn_states.cu:66-105
                      (math), 184-206 (checks); python wrapper adds the
                      OPTIONAL v_merged/s_merged out params (upstream
                      aot python/sgl_kernel/attention.py) and keeps lse
                      fp32 regardless of the v dtype.
  concat_mla_k        upstream csrc/elementwise/concat_mla.cu:16-110 —
                      in-place k[T,128,192] = nope | rope-broadcast; bf16
                      3D, unit last-dim stride, 16B alignment.
  concat_mla_absorb_q upstream csrc/elementwise/concat_mla.cu:120-190 —
                      a[...,512]+b[...,64] -> allocating out[...,576]
                      (python wrapper allocates + returns, upstream
                      elementwise.py:431-440).
  copy_to_gpu_no_ce   upstream csrc/elementwise/copy.cu — CPU int32 1D,
                      N in {32,64,72} template whitelist else "unexpected
                      N"; mechanism: values carried in kernel launch
                      parameters (CUDA-runtime trick, not reproducible
                      from torch — emu tier does output.copy_(input),
                      numerically identical, mechanism pinned in the
                      BACKEND_CATALOG note).

Known deliberate divergences (anchored, see module docstrings and PR):
  - merge_state_v2 all-masked rows: this package pins (0, -inf) — the
    FA emu contract its consumers rely on; the upstream CUDA wheel has
    no special case and emits (NaN, 0-ish) there. ±inf lse lanes DO
    normalize to -inf on both (upstream convention).
  - concat/copy host checks raise RuntimeError (mirroring upstream
    TORCH_CHECK) — unlike the activation face where upstream raises
    ValueError from the python layer.

Run (CPU face, any host):  pytest tests/kernels/test_aot_contract_b_class.py
Run (GPU face):            same on an HCU/CUDA node (triton tiers vs emu;
对拍待容器 without a device).
"""

import inspect
import math

import pytest
import torch

import sgl_kernel

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0),
    reason="requires HCU/CUDA device",
)


# ---------------------------------------------------------- merge_state_v2 --

def _lse_merge_ref(v_a, s_a, v_b, s_b):
    """Hand-written exact LSE merge (merge_attn_states.cu:66-105 math)."""
    m = torch.maximum(s_a, s_b)
    m_safe = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    wa = torch.exp(s_a - m_safe)
    wb = torch.exp(s_b - m_safe)
    denom = wa + wb
    o = (v_a * wa[..., None] + v_b * wb[..., None]) / denom[..., None]
    o = torch.nan_to_num(o, nan=0.0)  # all-masked -> 0 (FA contract here)
    return o, m + torch.log(denom)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_merge_state_v2_math(dtype):
    torch.manual_seed(0)
    M, H, D = 6, 4, 128
    va = torch.randn(M, H, D).to(dtype)
    vb = torch.randn(M, H, D).to(dtype)
    sa = torch.randn(M, H) * 3
    sb = torch.randn(M, H) * 3
    o, lse = sgl_kernel.merge_state_v2(va, sa, vb, sb)
    ro, rl = _lse_merge_ref(va.float(), sa, vb.float(), sb)
    torch.testing.assert_close(o.float(), ro.to(dtype).float(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, rl, rtol=1e-5, atol=1e-6)


def test_merge_state_v2_out_params_identity():
    # Upstream python wrapper: optional destination tensors; results are
    # written INTO them and the same tensors are returned.
    torch.manual_seed(1)
    va = torch.randn(3, 2, 64, dtype=torch.bfloat16)
    vb = torch.randn(3, 2, 64, dtype=torch.bfloat16)
    sa = torch.randn(3, 2)
    sb = torch.randn(3, 2)
    vm = torch.empty(3, 2, 64, dtype=torch.bfloat16)
    sm = torch.empty(3, 2, dtype=torch.float32)
    o1, l1 = sgl_kernel.merge_state_v2(va, sa, vb, sb)
    o2, l2 = sgl_kernel.merge_state_v2(va, sa, vb, sb, v_merged=vm, s_merged=sm)
    assert o2 is vm and l2 is sm
    torch.testing.assert_close(o2.float(), o1.float(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(l2, l1, rtol=1e-6, atol=1e-7)


def test_merge_state_v2_signature_has_out_params():
    # Upstream face: (v_a, s_a, v_b, s_b, v_merged=None, s_merged=None).
    params = list(inspect.signature(sgl_kernel.merge_state_v2).parameters)
    assert params == ["v_a", "s_a", "v_b", "s_b", "v_merged", "s_merged"]
    for p in params[4:]:
        assert inspect.signature(sgl_kernel.merge_state_v2).parameters[p].default is None


def test_merge_state_v2_rejects_non_fp32_s_merged():
    # The lse output is fp32 regardless of the v dtype (upstream
    # merge_attn_states TORCH_CHECKs fp32). A `.float()` rebind would
    # silently detach the caller's buffer — the merge writes a private
    # copy, the returned tensor is a different storage, and the out=
    # contract breaks (review #80); reject the dtype instead.
    va = torch.randn(2, 2, 32, dtype=torch.float16)
    vb = torch.randn(2, 2, 32, dtype=torch.float16)
    sa = torch.zeros(2, 2)
    sb = torch.zeros(2, 2)
    sm16 = torch.empty(2, 2, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="float32"):
        sgl_kernel.merge_state_v2(va, sa, vb, sb, s_merged=sm16)


def test_merge_state_v2_all_masked_rows_pinned():
    # FA-contract anchor: both lse -inf -> (0 output, -inf lse). The
    # upstream CUDA wheel emits NaN here (no special case in
    # merge_attn_states.cu) — documented divergence, deliberately pinned.
    va = torch.randn(2, 2, 32, dtype=torch.bfloat16)
    vb = torch.randn(2, 2, 32, dtype=torch.bfloat16)
    neg = torch.full((2, 2), float("-inf"))
    o, lse = sgl_kernel.merge_state_v2(va, neg, vb, neg)
    assert torch.all(o == 0)
    assert torch.all(torch.isneginf(lse))


def test_merge_state_v2_posinf_lse_normalizes_to_masked():
    # ±inf lse lanes count as fully masked (upstream convention).
    va = torch.randn(1, 1, 16, dtype=torch.float32)
    vb = torch.randn(1, 1, 16, dtype=torch.float32)
    sa = torch.full((1, 1), float("inf"))
    sb = torch.zeros(1, 1)
    o, lse = sgl_kernel.merge_state_v2(va, sa, vb, sb)
    torch.testing.assert_close(o[0, 0], vb[0, 0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(lse, sb, rtol=1e-6, atol=1e-7)


def test_merge_state_v2_shape_guards():
    v = torch.randn(2, 3, 8)
    s_bad = torch.randn(2, 4)          # lse not matching v's prefix
    s = torch.randn(2, 3)
    with pytest.raises(RuntimeError, match="unexpected shapes"):
        sgl_kernel.merge_state_v2(v, s_bad, v, s_bad)
    with pytest.raises(RuntimeError, match="mismatched"):
        sgl_kernel.merge_state_v2(v, s, torch.randn(2, 3, 9), s)
    with pytest.raises(RuntimeError, match="v_merged"):
        sgl_kernel.merge_state_v2(v, s, v, s, v_merged=torch.empty(2, 3, 7))


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_merge_state_v2_triton_vs_emu(dtype):
    # Triton tier vs the torch reference on the same GPU inputs.
    torch.manual_seed(2)
    M, H, D = 17, 8, 128
    va = torch.randn(M, H, D, device="cuda").to(dtype)
    vb = torch.randn(M, H, D, device="cuda").to(dtype)
    sa = torch.randn(M, H, device="cuda") * 3
    sb = torch.randn(M, H, device="cuda") * 3
    o_t, l_t = sgl_kernel.merge_state_v2(va, sa, vb, sb)
    ro, rl = _lse_merge_ref(va.float(), sa, vb.float(), sb)
    torch.testing.assert_close(o_t.float(), ro.to(dtype).float().cuda(),
                               rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(l_t, rl.cuda(), rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------- concat_mla --

def test_concat_mla_k_contract():
    # concat_mla.cu: k[T,128,192] = k_nope[T,128,128] | rope broadcast;
    # k_rope is ONE vector per token shared by all 128 heads.
    torch.manual_seed(3)
    T = 5
    kn = torch.randn(T, 128, 128, dtype=torch.bfloat16)
    kr = torch.randn(T, 1, 64, dtype=torch.bfloat16)
    k = torch.zeros(T, 128, 192, dtype=torch.bfloat16)
    sgl_kernel.concat_mla_k(k, kn, kr)
    torch.testing.assert_close(k[..., :128], kn, rtol=0, atol=0)
    for t in range(T):
        for h in (0, 1, 63, 127):
            torch.testing.assert_close(k[t, h, 128:], kr[t, 0], rtol=0, atol=0)
    # in-place on k: no return, k itself carries the result
    assert sgl_kernel.concat_mla_k(k, kn, kr) is None


def test_concat_mla_k_guard_surface():
    T = 2
    k = torch.zeros(T, 128, 192, dtype=torch.bfloat16)
    kn = torch.zeros(T, 128, 128, dtype=torch.bfloat16)
    kr = torch.zeros(T, 1, 64, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="last dim"):
        sgl_kernel.concat_mla_k(torch.zeros(T, 128, 128, dtype=torch.bfloat16), kn, kr)
    with pytest.raises(RuntimeError, match="bf16"):
        sgl_kernel.concat_mla_k(
            torch.zeros(T, 128, 192, dtype=torch.float32), kn, kr)
    with pytest.raises(RuntimeError, match="head dim"):
        sgl_kernel.concat_mla_k(k, torch.zeros(T, 64, 128, dtype=torch.bfloat16), kr)
    with pytest.raises(RuntimeError, match="3-D"):
        sgl_kernel.concat_mla_k(torch.zeros(192, dtype=torch.bfloat16), kn, kr)


def test_concat_mla_absorb_q_contract():
    # concat_mla.cu:120-190: a[...,512] + b[...,64] -> out[...,576],
    # allocated and returned by the python wrapper (elementwise.py:431).
    torch.manual_seed(4)
    a = torch.randn(3, 5, 512, dtype=torch.bfloat16)
    b = torch.randn(3, 5, 64, dtype=torch.bfloat16)
    out = sgl_kernel.concat_mla_absorb_q(a, b)
    assert out.shape == (3, 5, 576)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out[..., :512], a, rtol=0, atol=0)
    torch.testing.assert_close(out[..., 512:], b, rtol=0, atol=0)


def test_concat_mla_absorb_q_guard_surface():
    a = torch.zeros(2, 3, 512, dtype=torch.bfloat16)
    b = torch.zeros(2, 3, 64, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="last dim"):
        sgl_kernel.concat_mla_absorb_q(torch.zeros(2, 3, 256, dtype=torch.bfloat16), b)
    with pytest.raises(RuntimeError, match="leading dims"):
        sgl_kernel.concat_mla_absorb_q(a, torch.zeros(2, 4, 64, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="bf16"):
        sgl_kernel.concat_mla_absorb_q(torch.zeros(2, 3, 512, dtype=torch.float16), b)


@requires_gpu
def test_concat_mla_triton_tiers_vs_emu():
    torch.manual_seed(5)
    T = 9
    kn = torch.randn(T, 128, 128, device="cuda", dtype=torch.bfloat16)
    kr = torch.randn(T, 1, 64, device="cuda", dtype=torch.bfloat16)
    k_tr = torch.zeros(T, 128, 192, device="cuda", dtype=torch.bfloat16)
    k_ref = torch.zeros(T, 128, 192, device="cuda", dtype=torch.bfloat16)
    sgl_kernel.concat_mla_k(k_tr, kn, kr)
    # force the emu tier by calling on the same device through the torch
    # ops directly (the emu face is exactly these two strided copies)
    k_ref[..., :128].copy_(kn)
    k_ref[..., 128:].copy_(kr)
    torch.testing.assert_close(k_tr, k_ref, rtol=0, atol=0)

    a = torch.randn(4, 7, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4, 7, 64, device="cuda", dtype=torch.bfloat16)
    out = sgl_kernel.concat_mla_absorb_q(a, b)
    torch.testing.assert_close(out[..., :512], a, rtol=0, atol=0)
    torch.testing.assert_close(out[..., 512:], b, rtol=0, atol=0)


@requires_gpu
def test_concat_mla_absorb_q_strided_view_falls_back():
    # a=(2,3,513)[..., :512] passes every common check (3-D, bf16, last
    # dim unit stride, 16B base) but its row strides (3*513, 513) are not
    # multiples of 8 bf16 elements — the native kernel's uint4 loop would
    # TORCH_CHECK-fail. The dispatch must fall back to the strided
    # triton/torch tiers and still produce the exact concat (review #80).
    torch.manual_seed(6)
    storage = torch.randn(2, 3, 513, device="cuda", dtype=torch.bfloat16)
    a = storage[..., :512]
    b = torch.randn(2, 3, 64, device="cuda", dtype=torch.bfloat16)
    assert a.stride(1) == 513 and a.stride(1) % 8 != 0
    out = sgl_kernel.concat_mla_absorb_q(a, b)
    assert out.shape == (2, 3, 576)
    torch.testing.assert_close(out[..., :512], a, rtol=0, atol=0)
    torch.testing.assert_close(out[..., 512:], b, rtol=0, atol=0)


# -------------------------------------------------------- copy_to_gpu_no_ce --

def test_copy_to_gpu_no_ce_whitelist_cpu_face():
    # copy.cu: N must be 32/64/72 ("unexpected N" otherwise); the CPU-side
    # guards (1-D, int32, contiguous) mirror the TORCH_CHECKs verbatim.
    x = torch.zeros(16, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="unexpected N"):
        sgl_kernel.copy_to_gpu_no_ce(x, torch.zeros(16, dtype=torch.int32))
    with pytest.raises(RuntimeError, match="1-D"):
        sgl_kernel.copy_to_gpu_no_ce(
            torch.zeros(4, 16, dtype=torch.int32),
            torch.zeros(64, dtype=torch.int32),
        )
    with pytest.raises(RuntimeError, match="dtype"):
        sgl_kernel.copy_to_gpu_no_ce(
            torch.zeros(64, dtype=torch.float32),
            torch.zeros(64, dtype=torch.int32),
        )


@requires_gpu
@pytest.mark.parametrize("n", [32, 64, 72])
def test_copy_to_gpu_no_ce_gpu_values(n):
    # Numerically identical H2D (mechanism differs: regular copy path vs
    # the upstream kernel-parameter trick — see module docstring).
    src = torch.arange(n, dtype=torch.int32) * 3 - 1
    dst = torch.zeros(n, device="cuda", dtype=torch.int32)
    sgl_kernel.copy_to_gpu_no_ce(src, dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst.cpu(), src, rtol=0, atol=0)


@requires_gpu
def test_merge_state_v2_native_rejects_malformed_out():
    # The native entry validates the out pair before writing through raw
    # pointers: wrong device/shape/layout/dtype outs must raise instead of
    # corrupting memory (review #80 R8 P2).
    from sgl_kernel._native import native_op

    # Gate on an op only OUR library defines: a foreign sgl_kernel .so
    # (e.g. a dtk wheel) may pre-register merge_state_v2 WITHOUT these
    # entry checks — exercising that one would fault the GPU instead of
    # raising. If our v2 quant op is registered, our .so owns the
    # namespace and merge_state_v2 is ours too.
    if native_op("sgl_per_token_group_quant_8bit_v2") is None:
        pytest.skip("shadow cpp tier not owning the sgl_kernel:: namespace")
    op = native_op("merge_state_v2")

    va = torch.randn(2, 2, 32, device="cuda", dtype=torch.bfloat16)
    vb = torch.randn(2, 2, 32, device="cuda", dtype=torch.bfloat16)
    sa = torch.zeros(2, 2, device="cuda")
    sb = torch.zeros(2, 2, device="cuda")

    o = torch.empty_like(va)
    l = torch.empty(2, 2, device="cuda")
    op(va, sa, vb, sb, o, l)  # sanity: a correct call passes

    bad = [
        ("cpu out", torch.empty(2, 2, 32), l),
        ("short out", torch.empty(2, 2, 16, device="cuda"), l),
        ("non-contiguous", torch.empty(2, 2, 32, 64, device="cuda")[..., :32], l),
        ("v dtype", torch.empty(2, 2, 32, device="cuda", dtype=torch.float32), l),
        ("lse dtype", o, torch.empty(2, 2, device="cuda", dtype=torch.float16)),
    ]
    for tag, bad_o, bad_l in bad:
        with pytest.raises(RuntimeError):
            op(va, sa, vb, sb, bad_o, bad_l)
