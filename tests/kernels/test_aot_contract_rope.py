# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""AOT contract anchors: rotary_embedding (pos_enc.cu).

Contract sources (upstream sgl-kernel v0.5.22 tag):
  - csrc/elementwise/pos_enc.cu
      * in-place on query/key, one row per token, row offset = positions[i]
      * cos_sin_cache is ALWAYS half-split: row = [cos(rd/2) | sin(rd/2)]
        fp32, independent of is_neox_style — only the lane pairing differs
      * is_neox=True  (NeoX/rotate_half): lane j pairs with j + rd/2;
        y1 = x1*cos - x2*sin ; y2 = x2*cos + x1*sin
      * is_neox=False (GPT-J interleaved): even lane 2j pairs with odd
        2j+1, same rotation formula on the (even, odd) pair
      * partial rotary: only the first rotary_dim lanes of each head
        rotate; lanes [rotary_dim, head_size) pass through untouched
      * positions load as int64_t (cache row addressing is 64-bit)
      * math in fp32, stored back to the activation dtype
  - python/sgl_kernel/rotary_embedding.py: signature
    (positions, query, key, head_size, cos_sin_cache, is_neox=True)

Known deliberate divergences / gaps (see PR notes):
  - upstream deepseek v3.1 norm+rope fused ops (dsv4_norm_rope.cu family)
    have no counterpart in this package — out of anchor scope.
  - R3 (fixed here): int32 positions are normalized to int64 at the
    python entry; previously the triton tier would index the cache with
    32-bit math (pos * rotary_dim overflow risk).

Run (CPU face, any host):  pytest tests/kernels/test_aot_contract_rope.py
Run (GPU face):            same on an HCU/CUDA node (对拍待容器 without one).
"""

import math

import pytest
import torch

import sgl_kernel

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0),
    reason="requires HCU/CUDA device",
)

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _make_cache(max_pos, rotary_dim):
    """sglang RotaryEmbedding cache layout: [cos(rd/2) | sin(rd/2)] fp32."""
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (max_pos, rd/2)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)  # half-split


def _neox_ref(q, k, positions, head_size, cache):
    rd = cache.shape[1]
    half = rd // 2
    cos = cache[positions.long(), :half].float()
    sin = cache[positions.long(), half:].float()

    def rot(t):
        shape = t.shape
        v = t.view(shape[0], -1, head_size)
        out = v.clone()
        for h in range(v.shape[1]):
            x1 = v[:, h, :half].float()
            x2 = v[:, h, half:rd].float()
            out[:, h, :half] = (x1 * cos - x2 * sin).to(t.dtype)
            out[:, h, half:rd] = (x2 * cos + x1 * sin).to(t.dtype)
        return out.view(shape)

    return rot(q), (rot(k) if k is not None else None)


def _gptj_ref(q, k, positions, head_size, cache):
    rd = cache.shape[1]
    half = rd // 2
    cos = cache[positions.long(), :half].float()
    sin = cache[positions.long(), half:].float()

    def rot(t):
        shape = t.shape
        v = t.view(shape[0], -1, head_size)
        out = v.clone()
        for h in range(v.shape[1]):
            x1 = v[:, h, 0:rd:2].float()   # even lanes
            x2 = v[:, h, 1:rd:2].float()   # odd lanes
            out[:, h, 0:rd:2] = (x1 * cos - x2 * sin).to(t.dtype)
            out[:, h, 1:rd:2] = (x2 * cos + x1 * sin).to(t.dtype)
        return out.view(shape)

    return rot(q), (rot(k) if k is not None else None)


# ------------------------------------------------ cpu face (torch tier) --

@pytest.mark.parametrize("dtype", DTYPES)
def test_neox_math_and_cache_half_split(dtype):
    # pos_enc.cu NeoX: y1 = x1*cos - x2*sin at lane j, y2 = x2*cos + x1*sin
    # at lane j + rd/2; the cache row is [cos | sin] (cos FIRST).
    torch.manual_seed(0)
    T, H, hs, rd = 5, 4, 64, 32
    cache = _make_cache(64, rd)
    q = torch.randn(T, H * hs).to(dtype)
    k = torch.randn(T, 2 * hs)
    positions = torch.tensor([0, 3, 7, 15, 63], dtype=torch.int64)
    q_ref, k_ref = _neox_ref(q.clone(), k.clone(), positions, hs, cache)
    ret = sgl_kernel.rotary_embedding(positions, q, k, hs, cache, is_neox=True)
    assert ret is None  # in-place contract
    tol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    torch.testing.assert_close(q.float(), q_ref.float(), rtol=tol[0], atol=tol[1])
    torch.testing.assert_close(k.float(), k_ref.float(), rtol=tol[0], atol=tol[1])


def test_cache_layout_cos_first_via_quarter_turn():
    # A quarter-turn row (angle = pi/2) pins the layout: cos = 0, sin = 1
    # everywhere -> y1 = -x2, y2 = x1 exactly. Swapping the cache halves
    # (sin-first) would produce y1 = +x2.
    T, H, hs, rd = 1, 1, 8, 4
    half = rd // 2
    cache = torch.zeros(1, rd)
    cache[0, :half] = 0.0    # cos(pi/2) = 0
    cache[0, half:] = 1.0    # sin(pi/2) = 1
    x = torch.arange(1, hs + 1, dtype=torch.float32).reshape(1, hs)
    sgl_kernel.rotary_embedding(torch.zeros(1, dtype=torch.int64), x, None, hs, cache,
                                is_neox=True)
    assert x[0, 0].item() == pytest.approx(-3.0)  # y1 = -x2
    assert x[0, 1].item() == pytest.approx(-4.0)
    assert x[0, 2].item() == pytest.approx(1.0)   # y2 = x1
    assert x[0, 3].item() == pytest.approx(2.0)
    assert x[0, 4:].tolist() == [5.0, 6.0, 7.0, 8.0]  # passthrough lanes


@pytest.mark.parametrize("dtype", DTYPES)
def test_gptj_interleaved_pairing(dtype):
    # pos_enc.cu GPT-J: even lane 2j pairs with odd 2j+1; the cache stays
    # half-split (is_neox_style does NOT change the cache layout).
    torch.manual_seed(1)
    T, H, hs, rd = 3, 2, 32, 16
    cache = _make_cache(32, rd)
    q = torch.randn(T, H * hs).to(dtype)
    positions = torch.tensor([1, 5, 31], dtype=torch.int64)
    q_ref, _ = _gptj_ref(q.clone(), None, positions, hs, cache)
    sgl_kernel.rotary_embedding(positions, q, None, hs, cache, is_neox=False)
    tol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    torch.testing.assert_close(q.float(), q_ref.float(), rtol=tol[0], atol=tol[1])


def test_partial_rotary_dim_passthrough():
    # Only lanes [0, rotary_dim) rotate; the head tail passes through
    # bit-exact (pos_enc.cu writes only the rotary window).
    torch.manual_seed(2)
    T, H, hs, rd = 4, 3, 64, 16
    cache = _make_cache(64, rd)
    q = torch.randn(T, H * hs, dtype=torch.float32)
    k = torch.randn(T, H * hs, dtype=torch.float32)
    q_tail = q.view(T, H, hs)[:, :, rd:].clone()
    k_tail = k.view(T, H, hs)[:, :, rd:].clone()
    positions = torch.tensor([2, 9, 40, 63], dtype=torch.int64)
    sgl_kernel.rotary_embedding(positions, q, k, hs, cache, is_neox=True)
    torch.testing.assert_close(q.view(T, H, hs)[:, :, rd:], q_tail, rtol=0, atol=0)
    torch.testing.assert_close(k.view(T, H, hs)[:, :, rd:], k_tail, rtol=0, atol=0)


def test_per_head_independent_and_position_indexing():
    # Every head rotates independently; each token uses its own cache row.
    T, H, hs, rd = 3, 4, 8, 8
    cache = _make_cache(16, rd)
    q0 = torch.randn(T, H * hs, dtype=torch.float32)
    q0[1] = q0[0]  # identical rows rotated at the same position must agree
    q = q0.clone()
    positions = torch.tensor([4, 4, 11], dtype=torch.int64)
    sgl_kernel.rotary_embedding(positions, q, None, hs, cache, is_neox=True)
    # tokens 0 and 1 share position 4 -> identical rows across ALL heads
    torch.testing.assert_close(q[0], q[1], rtol=0, atol=0)
    q_ref, _ = _neox_ref(q0.clone(), None, positions, hs, cache)
    torch.testing.assert_close(q, q_ref, rtol=1e-5, atol=1e-6)


def test_int32_positions_normalized():
    # R3 fix: int32 positions must behave like int64 (cache rows picked by
    # value, not by buffer offset).
    T, H, hs, rd = 2, 1, 8, 8
    cache = _make_cache(32, rd)
    q64 = torch.randn(T, hs, dtype=torch.float32)
    q32 = q64.clone()
    pos64 = torch.tensor([5, 30], dtype=torch.int64)
    pos32 = pos64.to(torch.int32)
    sgl_kernel.rotary_embedding(pos64, q64, None, hs, cache)
    sgl_kernel.rotary_embedding(pos32, q32, None, hs, cache)
    torch.testing.assert_close(q32, q64, rtol=0, atol=0)


def test_key_none_accepted():
    cache = _make_cache(8, 8)
    q = torch.randn(2, 8, dtype=torch.float32)
    sgl_kernel.rotary_embedding(
        torch.tensor([0, 1], dtype=torch.int64), q, None, 8, cache)
    assert not torch.isnan(q).any()


# ------------------------------------------------ gpu face (triton tier) --

@requires_gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_neox_triton_vs_torch_ref(dtype):
    torch.manual_seed(3)
    T, H, hs, rd = 17, 5, 64, 32
    cache = _make_cache(64, rd).cuda()
    q = torch.randn(T, H * hs, device="cuda").to(dtype)
    k = torch.randn(T, 2 * hs, device="cuda").to(dtype)
    positions = torch.randint(0, 64, (T,), device="cuda", dtype=torch.int64)
    # reference runs on CPU tensors: q/k/positions must move with the cache
    q_ref, k_ref = _neox_ref(
        q.clone().cpu(), k.clone().cpu(), positions.cpu(), hs, cache.cpu()
    )
    sgl_kernel.rotary_embedding(positions, q, k, hs, cache, is_neox=True)
    # gfx938/dtk2610 实测（nmz26 GPU6, 2026-11-12）: 全 dtype 与参考逐位一致
    # （maxabs=0，fp32 2.4e-7）。bf16 档按 1 ulp(≈0.4% rel) 的 ~25x 余量收紧
    # ——rtol<1 且旋转输出 O(1)（|ref|max≈3.9），全零输出必被拒（review #80 R6）。
    tol = (2e-3, 2e-3) if dtype != torch.bfloat16 else (2e-2, 2e-2)
    torch.testing.assert_close(q.float(), q_ref.float().cuda(),
                               rtol=tol[0], atol=tol[1])
    torch.testing.assert_close(k.float(), k_ref.float().cuda(),
                               rtol=tol[0], atol=tol[1])


@requires_gpu
def test_partial_rotary_triton_passthrough():
    torch.manual_seed(4)
    T, H, hs, rd = 9, 4, 64, 16
    cache = _make_cache(64, rd).cuda()
    q = torch.randn(T, H * hs, device="cuda", dtype=torch.float16)
    tail = q.view(T, H, hs)[:, :, rd:].clone()
    positions = torch.randint(0, 64, (T,), device="cuda", dtype=torch.int64)
    sgl_kernel.rotary_embedding(positions, q, None, hs, cache, is_neox=True)
    torch.testing.assert_close(q.view(T, H, hs)[:, :, rd:], tail, rtol=0, atol=0)


@requires_gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_gptj_interleaved_torch_on_gpu(dtype):
    # GPT-J has no triton tier — CUDA tensors go through the torch impl,
    # whose even/odd index tensors must live on the tensor's device: CUDA
    # self + CPU indices is a hard RuntimeError in advanced indexing
    # (review #80 P1; only reachable on-GPU, CPU runs never saw it).
    torch.manual_seed(5)
    T, H, hs, rd = 11, 3, 32, 16
    cache = _make_cache(64, rd).cuda()
    q = torch.randn(T, H * hs, device="cuda").to(dtype)
    k = torch.randn(T, hs, device="cuda").to(dtype)
    positions = torch.randint(0, 64, (T,), device="cuda", dtype=torch.int64)
    q_ref, k_ref = _gptj_ref(
        q.clone().cpu(), k.clone().cpu(), positions.cpu(), hs, cache.cpu()
    )
    sgl_kernel.rotary_embedding(positions, q, k, hs, cache, is_neox=False)
    tol = (2e-3, 2e-3) if dtype != torch.bfloat16 else (2e-2, 2e-2)  # 同 neox 实测
    torch.testing.assert_close(q.float(), q_ref.float().cuda(),
                               rtol=tol[0], atol=tol[1])
    torch.testing.assert_close(k.float(), k_ref.float().cuda(),
                               rtol=tol[0], atol=tol[1])
