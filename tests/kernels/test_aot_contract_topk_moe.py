# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""AOT contract anchors: MoE gate top-k (softmax/sigmoid) + moe_align_block_size.

Contract sources (upstream sgl-kernel v0.5.22 tag):
  - csrc/moe/moe_topk_softmax_kernels.cu
      * ties choose the LOWER expert index (L544-549, ordered max-k scan)
      * softmax variant: weights are softmax(logits) WITHOUT the correction
        bias; the bias only steers selection (bias added to logits)
      * sigmoid variant: bias added to the ACTIVATED sigmoid values
      * topk_weights are written fp32 regardless of the buffer dtype
      * renormalize divides by the routed sum with NO zero-guard
      * softcapping: logits = cap * tanh(logits / cap) before softmax
      * k > num_experts fails a TORCH_CHECK inside the C++ launcher
  - csrc/moe/moe_align_kernel.cu
      * +1-offset bucket convention: expert_id + 1 (0 slot = invalid)
      * holes in sorted_token_ids are filled with topk_ids.numel()
      * pad_sorted_token_ids=false leaves holes UNWRITTEN upstream (caller
        responsibility); ignore_invalid_expert=true DROPS -1 lanes
      * callers pass num_experts+1 buckets upstream

Known deliberate divergences of this shadow package (documented here,
anchored as-is; see PR notes):
  D1 — moe_align_block_size takes the REAL expert count E (not E+1) and
       builds the +1 sentinel bucket internally; blocks owned by the
       sentinel bucket are flagged expert id -1 (filter_expert zeroes
       those rows) instead of being dropped.
  D2 — pad_sorted_token_ids is accepted but holes are ALWAYS filled with
       numel (the fused-moe consumer relies on it).
  D3 — ignore_invalid_expert is accepted; both values keep the sentinel
       bucket (D1) — dropping lanes left cache rows unwritten.
  D4 — topk_weights are cast to the caller's buffer dtype (upstream writes
       fp32 unconditionally); fp32 buffers (the only kind the runner
       passes) are bit-compatible.
  D5 — renormalize guards against zero routed sums (upstream would emit
       NaN); nonzero-sum rows are identical.
  D6 — k out of range raises RuntimeError in python here (upstream
       TORT_CHECK/exceptions surface from the C++ launcher).
  The upstream fast_topk_v2 / fast_topk_transform_fused /
  fast_topk_transform_ragged_fused / deepseek_v4_topk_transform_512 ops
  have no counterpart in this package at all (not anchored here).

Run (CPU face, any host):  pytest tests/kernels/test_aot_contract_topk_moe.py
Run (GPU face):            same on an HCU/CUDA node (moe_align triton tiers
vs the torch port; 对拍待容器 without a device).
"""

import pytest
import torch

import sgl_kernel
from sgl_kernel import moe as moe_mod

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0),
    reason="requires HCU/CUDA device",
)


# ------------------------------------------------------------------ topk --

def test_topk_softmax_tie_breaks_to_lower_expert():
    # moe_topk_softmax_kernels.cu:544-549: equal scores keep the lower id.
    gating = torch.tensor([[1.0, 3.0, 3.0, 2.0]])  # experts 1 and 2 tie at 3.0
    w = torch.zeros(1, 2, dtype=torch.float32)
    ids = torch.zeros(1, 2, dtype=torch.int32)
    sgl_kernel.topk_softmax(w, ids, gating)
    assert ids[0].tolist() == [1, 2]  # lower index first on the tie
    p = torch.softmax(gating, dim=-1)
    torch.testing.assert_close(w, torch.tensor([[p[0, 1], p[0, 2]]]))


def test_topk_softmax_math_and_renormalize():
    torch.manual_seed(0)
    gating = torch.randn(5, 8)
    for renorm in (False, True):
        w = torch.zeros(5, 3, dtype=torch.float32)
        ids = torch.zeros(5, 3, dtype=torch.int32)
        sgl_kernel.topk_softmax(w, ids, gating, renormalize=renorm)
        p = torch.softmax(gating, dim=-1)
        top = torch.argsort(gating, dim=-1, descending=True, stable=True)[:, :3]
        ref_w = torch.gather(p, 1, top)
        if renorm:
            ref_w = ref_w / ref_w.sum(-1, keepdim=True)
        torch.testing.assert_close(ids.long(), top)
        torch.testing.assert_close(w, ref_w, rtol=1e-6, atol=1e-7)


def test_topk_softmax_softcap_formula():
    # softcapping: cap * tanh(x / cap) before softmax (moe_topk_softmax_kernels.cu).
    torch.manual_seed(1)
    gating = torch.randn(4, 6) * 4
    cap = 2.0
    w = torch.zeros(4, 2, dtype=torch.float32)
    ids = torch.zeros(4, 2, dtype=torch.int32)
    sgl_kernel.topk_softmax(w, ids, gating, renormalize=True, moe_softcapping=cap)
    capped = cap * torch.tanh(gating / cap)
    p = torch.softmax(capped, dim=-1)
    top = torch.argsort(capped, dim=-1, descending=True, stable=True)[:, :2]
    ref_w = torch.gather(p, 1, top)
    ref_w = ref_w / ref_w.sum(-1, keepdim=True)
    torch.testing.assert_close(w, ref_w, rtol=1e-6, atol=1e-7)


def test_topk_softmax_correction_bias_selects_but_not_weights():
    # Softmax variant: bias rides on the LOGITS for selection only; the
    # emitted weights are softmax of the unbiased logits.
    torch.manual_seed(2)
    gating = torch.randn(3, 6)
    bias = torch.randn(6)
    w = torch.zeros(3, 2, dtype=torch.float32)
    ids = torch.zeros(3, 2, dtype=torch.int32)
    sgl_kernel.topk_softmax(w, ids, gating, correction_bias=bias)
    biased = gating + bias
    top = torch.argsort(biased, dim=-1, descending=True, stable=True)[:, :2]
    p = torch.softmax(gating, dim=-1)
    torch.testing.assert_close(ids.long(), top)
    torch.testing.assert_close(w, torch.gather(p, 1, top), rtol=1e-6, atol=1e-7)


def test_topk_sigmoid_contract():
    # Sigmoid variant: weights = sigmoid(logits); bias joins AFTER the
    # activation; routed_scaling_factor multiplies; shared experts append
    # num_experts..+n slots carrying routed_sum / rsf.
    torch.manual_seed(3)
    gating = torch.randn(4, 8)
    w = torch.zeros(4, 3, dtype=torch.float32)
    ids = torch.zeros(4, 3, dtype=torch.int32)
    sgl_kernel.topk_sigmoid(w, ids, gating, renormalize=False,
                            routed_scaling_factor=1.0,
                            num_fused_shared_experts=0)
    act = torch.sigmoid(gating)
    top = torch.argsort(act, dim=-1, descending=True, stable=True)[:, :3]
    torch.testing.assert_close(ids.long(), top)
    torch.testing.assert_close(w, torch.gather(act, 1, top), rtol=1e-6, atol=1e-7)


def test_topk_sigmoid_bias_on_activation():
    torch.manual_seed(4)
    gating = torch.randn(3, 6)
    bias = torch.randn(6)
    w = torch.zeros(3, 2, dtype=torch.float32)
    ids = torch.zeros(3, 2, dtype=torch.int32)
    sgl_kernel.topk_sigmoid(w, ids, gating, correction_bias=bias)
    act = torch.sigmoid(gating)
    biased = act + bias
    top = torch.argsort(biased, dim=-1, descending=True, stable=True)[:, :2]
    torch.testing.assert_close(ids.long(), top)
    torch.testing.assert_close(w, torch.gather(act, 1, top), rtol=1e-6, atol=1e-7)


def test_topk_weights_written_in_caller_buffer_dtype():
    # D4: upstream writes fp32 unconditionally; here the buffer dtype wins.
    # fp32 buffers (the runner's only kind) are the bit-compatible anchor.
    gating = torch.randn(2, 5)
    w32 = torch.zeros(2, 2, dtype=torch.float32)
    ids = torch.zeros(2, 2, dtype=torch.int32)
    sgl_kernel.topk_softmax(w32, ids, gating)
    p = torch.softmax(gating, dim=-1)
    top = torch.argsort(gating, dim=-1, descending=True, stable=True)[:, :2]
    torch.testing.assert_close(w32, torch.gather(p, 1, top), rtol=1e-6, atol=1e-7)
    assert w32.dtype == torch.float32


def test_topk_k_out_of_range_raises():
    # D6: k > num_experts is rejected (upstream TORCH_CHECK in the launcher).
    gating = torch.randn(2, 4)
    w = torch.zeros(2, 5, dtype=torch.float32)  # k=5 > 4 experts
    ids = torch.zeros(2, 5, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="out of range"):
        sgl_kernel.topk_softmax(w, ids, gating)


def test_topk_k_equals_dim_boundary():
    # k == num_experts is legal: full permutation, renorm sums to 1.
    torch.manual_seed(5)
    gating = torch.randn(3, 4)
    w = torch.zeros(3, 4, dtype=torch.float32)
    ids = torch.zeros(3, 4, dtype=torch.int32)
    sgl_kernel.topk_softmax(w, ids, gating, renormalize=True)
    assert torch.allclose(w.sum(-1), torch.ones(3), atol=1e-6)
    for row in ids.long():
        assert sorted(row.tolist()) == [0, 1, 2, 3]


# ----------------------------------------------------------- moe_align --

def _run_align(topk_ids, num_experts, block_size, max_tokens=None, max_blocks=None):
    numel = topk_ids.numel()
    max_tokens = max_tokens or numel + num_experts * block_size + 1
    max_blocks = max_blocks or (max_tokens // block_size + 1)
    sorted_ids = torch.full((max_tokens,), -7, dtype=torch.int32)
    experts = torch.full((max_blocks,), -7, dtype=torch.int32)
    post_pad = torch.zeros(1, dtype=torch.int32)
    cumsum = torch.zeros(num_experts + 2, dtype=torch.int32)
    sgl_kernel.moe_align_block_size(
        topk_ids, num_experts, block_size,
        sorted_ids, experts, post_pad, cumsum,
    )
    return sorted_ids, experts, post_pad, cumsum


def test_moe_align_layout_and_padding_value():
    # moe_align_kernel.cu: holes carry topk_ids.numel(); every expert
    # segment is block-aligned; tokens of one expert stay in flat order
    # (stable fill); cumsum is the EXCLUSIVE block-aligned prefix.
    topk_ids = torch.tensor([[0, 2], [1, 2], [0, 0]], dtype=torch.int32)
    E, bs = 3, 4
    sorted_ids, experts, post_pad, cumsum = _run_align(topk_ids, E, bs)
    numel = topk_ids.numel()

    # counts: e0=3, e1=1, e2=2 (+0 sentinel) -> block-aligned 4,4,4(,0)
    assert int(post_pad[0]) == 12
    assert cumsum[: E + 1].tolist() == [0, 4, 8, 12]

    # expert 0 owns [0,4): flat ids {0, 4, 5} in order + 1 hole
    seg0 = sorted_ids[:4].tolist()
    assert seg0[:3] == [0, 4, 5] and seg0[3] == numel
    # expert 1 owns [4,8): {2} + holes
    seg1 = sorted_ids[4:8].tolist()
    assert seg1[0] == 2 and seg1[1:] == [numel] * 3
    # expert 2 owns [8,12): {1, 3} + holes
    seg2 = sorted_ids[8:12].tolist()
    assert seg2[:2] == [1, 3] and seg2[2:] == [numel] * 2

    # blocks 0..2 -> experts 0,1,2; tail blocks carry the bucket count
    assert experts[:3].tolist() == [0, 1, 2]
    assert (experts[3:] == E + 1).all()


def test_moe_align_valid_tokens_unique():
    # Every valid flat lane appears exactly once among non-hole slots.
    torch.manual_seed(6)
    topk_ids = torch.randint(0, 5, (17, 6), dtype=torch.int32)
    sorted_ids, experts, post_pad, cumsum = _run_align(topk_ids, 5, 8)
    numel = topk_ids.numel()
    valid = sorted_ids[sorted_ids != numel]
    assert sorted(valid.tolist()) == sorted(range(numel))


def test_moe_align_sentinel_lanes_get_dedicated_bucket():
    # D1: -1 (EP sentinel) lanes land in the LAST bucket; their blocks are
    # flagged expert -1 (upstream's ignore_invalid_expert=true DROPS them —
    # divergence anchored deliberately, see module docstring).
    topk_ids = torch.tensor([[0, -1], [-1, 2], [0, -1]], dtype=torch.int32)
    E, bs = 3, 4
    sorted_ids, experts, post_pad, cumsum = _run_align(topk_ids, E, bs)
    nb = E + 1
    # counts: e0=2, e1=0, e2=1, sentinel=3 -> aligned 4,0,4,4
    assert int(post_pad[0]) == 12
    assert cumsum[: nb + 1].tolist() == [0, 4, 4, 8, 12]
    # sentinel blocks (blocks 2..2) flagged -1
    assert experts[0].item() == 0
    assert experts[1].item() == 2
    assert experts[2].item() == -1
    assert (experts[3:] == nb).all()
    # the -1 lanes live in the sentinel segment [8,12): flat ids {1, 2, 5}
    seg = sorted_ids[8:12].tolist()
    assert seg.count(1) == 1 and seg.count(2) == 1 and seg.count(5) == 1


def test_moe_align_empty_input():
    topk_ids = torch.zeros(0, 2, dtype=torch.int32)
    sorted_ids, experts, post_pad, cumsum = _run_align(topk_ids, 3, 4)
    numel = 0
    assert int(post_pad[0]) == 0
    assert (sorted_ids == numel).all()
    assert cumsum[:4].tolist() == [0, 0, 0, 0]


def test_moe_align_cumsum_exclusive_block_aligned():
    torch.manual_seed(7)
    topk_ids = torch.randint(0, 4, (9, 3), dtype=torch.int32)
    E, bs = 4, 4
    sorted_ids, experts, post_pad, cumsum = _run_align(topk_ids, E, bs)
    flat = topk_ids.flatten().long()
    counts = torch.bincount(flat, minlength=E)
    aligned = ((counts + bs - 1) // bs) * bs
    expect = [0] + torch.cumsum(aligned, 0).tolist()
    assert cumsum[: E + 1].tolist() == expect
    assert int(post_pad[0]) == expect[-1]


# ------------------------------------------- gpu face (triton vs torch) --

@requires_gpu
@pytest.mark.parametrize("mode", ["auto", "quartet"])
def test_moe_align_triton_tiers_match_torch_port(mode, monkeypatch):
    # GPU tiers (fused single-CTA + 4-launch quartet) vs the torch port:
    # every scalar contract bit-exact; within one expert's segment the
    # token ORDER may differ (atomic cursors) — compared as multisets.
    monkeypatch.setenv("HCU_MOE_ALIGN_FUSED", "0" if mode == "quartet" else "")
    torch.manual_seed(8)
    E, bs = 6, 4
    topk_ids = torch.randint(-1, E, (33, 5), device="cuda", dtype=torch.int32)
    numel = topk_ids.numel()
    nb = E + 1
    max_tokens = numel + E * bs + 8

    def fresh():
        s = torch.full((max_tokens,), -7, device="cuda", dtype=torch.int32)
        e = torch.full((max_tokens // bs + 2,), -7, device="cuda", dtype=torch.int32)
        p = torch.zeros(1, device="cuda", dtype=torch.int32)
        c = torch.zeros(nb + 1, device="cuda", dtype=torch.int32)
        return s, e, p, c

    def run_triton():
        # triton tier under test — the public entry, mode selects the
        # fused single-CTA kernel or the 4-launch quartet
        s, e, p, c = fresh()
        sgl_kernel.moe_align_block_size(topk_ids, E, bs, s, e, p, c)
        torch.cuda.synchronize()
        return s, e, p, c

    def run_torch_port():
        # the torch port (the CPU tier) DIRECTLY on the same CUDA buffers
        # — pure torch ops, so it runs on-device. Review #80: routing both
        # sides through the same public entry compared one tier against
        # itself (run-to-run determinism only); the reference must come
        # from a DIFFERENT tier for this to be a cross-tier parity check.
        from sgl_kernel.moe import _moe_align_block_size_torch

        s, e, p, c = fresh()
        _moe_align_block_size_torch(topk_ids, E, bs, s, e, p, c, False)
        torch.cuda.synchronize()
        return s, e, p, c

    s_t, e_t, p_t, c_t = run_torch_port()
    s_g, e_g, p_g, c_g = run_triton()
    assert int(p_g[0]) == int(p_t[0])
    torch.testing.assert_close(c_g, c_t)
    torch.testing.assert_close(e_g, e_t)
    # per-expert segments compared as multisets
    bounds = [0] + torch.cumsum(
        ((torch.bincount(
            torch.where((topk_ids.flatten() >= 0) & (topk_ids.flatten() < E),
                        topk_ids.flatten().long(),
                        torch.full_like(topk_ids.flatten().long(), E)),
            minlength=nb) + bs - 1) // bs) * bs, 0).tolist()
    for lo, hi in zip(bounds, bounds[1:]):
        if hi > lo:
            a = torch.sort(s_t[lo:hi]).values
            b = torch.sort(s_g[lo:hi]).values
            torch.testing.assert_close(b, a)
