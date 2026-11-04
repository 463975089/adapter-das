# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""moe_align_block_size: padding/cumsum contract + capture-safety on GPU.

Capture/replay regression (527ad97, Qwen3.6-35B crash) is pinned by
test_moe_align_block_size_gpu_capture_safe: the kernel must run under
CUDA-graph capture without host syncs, and replay must track fresh inputs.

Run on a node with a working GPU: `pytest tests/kernels/test_moe_align.py -v`.
"""

import pytest
import torch

import sgl_kernel

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0),
    reason="requires HCU/CUDA device",
)


@requires_gpu
def test_moe_align_block_size_gpu():
    torch.manual_seed(7)
    T, K, E, B = 100, 8, 64, 32
    topk_ids = torch.randint(0, E, (T, K), device="cuda", dtype=torch.int32)
    numel = topk_ids.numel()
    max_padded = numel + (E + 1) * (B - 1)
    sorted_ids = torch.empty(max_padded, device="cuda", dtype=torch.int32)
    n_blocks = (max_padded + B - 1) // B
    expert_ids = torch.empty(n_blocks, device="cuda", dtype=torch.int32)
    post_pad = torch.empty(1, device="cuda", dtype=torch.int32)
    cumsum = torch.empty(E + 2, device="cuda", dtype=torch.int32)

    sgl_kernel.moe_align_block_size(topk_ids, E, B, sorted_ids, expert_ids,
                                    post_pad, cumsum, True)

    counts = torch.bincount(topk_ids.flatten().long(), minlength=E)
    expect_total = int(((counts + B - 1) // B * B).sum())
    assert post_pad.item() == expect_total
    valid = sorted_ids[sorted_ids < numel]
    e_of = topk_ids.flatten().long()[valid.long()]
    assert torch.equal(e_of, torch.sort(e_of).values)  # grouped by expert
    for b in range(expect_total // B):
        seg = sorted_ids[b * B: (b + 1) * B]
        real = seg[seg < numel]
        if len(real):
            assert expert_ids[b].item() == topk_ids.flatten().long()[real[0]].item()


def _align_buffers(topk_ids, E, B):
    numel = topk_ids.numel()
    max_padded = numel + (E + 1) * (B - 1)
    return (
        torch.empty(max_padded, device=topk_ids.device, dtype=torch.int32),
        torch.empty((max_padded + B - 1) // B, device=topk_ids.device, dtype=torch.int32),
        torch.empty(1, device=topk_ids.device, dtype=torch.int32),
        torch.empty(E + 2, device=topk_ids.device, dtype=torch.int32),
    )


@requires_gpu
def test_moe_align_block_size_gpu_matches_torch():
    """Triton path must reproduce the torch-port contract bit-exactly."""
    torch.manual_seed(9)
    for T, K, E, B in ((1, 8, 64, 32), (100, 8, 64, 32), (77, 6, 257, 64), (64, 4, 8, 16), (33, 1, 5, 8)):
        topk_ids = torch.randint(0, E, (T, K), device="cuda", dtype=torch.int32)
        s_t, e_t, p_t, c_t = _align_buffers(topk_ids, E, B)
        sgl_kernel.moe_align_block_size(topk_ids, E, B, s_t, e_t, p_t, c_t, True)

        cpu_ids = topk_ids.cpu()
        s_c, e_c, p_c, c_c = (t.cpu() for t in (s_t, e_t, p_t, c_t))
        sgl_kernel.moe_align_block_size(cpu_ids, E, B, s_c, e_c, p_c, c_c, True)
        assert torch.equal(p_t.cpu(), p_c)
        assert torch.equal(c_t.cpu(), c_c)
        assert torch.equal(e_t.cpu(), e_c)
        # Per-expert membership multiset must match (slot order inside one
        # expert's segment is atomic-arrival on GPU vs FIFO on CPU torch).
        flat_ref = topk_ids.flatten().long().cpu()
        for dev_sorted, ref in ((s_t.cpu(), flat_ref), (s_c, flat_ref)):
            got_experts = ref[dev_sorted[dev_sorted < T * K].long()]
            got_experts, _ = got_experts.sort()
            want_experts, _ = ref.sort()
            assert torch.equal(got_experts, want_experts)


@requires_gpu
def test_moe_align_block_size_gpu_capture_safe():
    torch.manual_seed(11)
    T, K, E, B = 128, 8, 64, 32
    topk_ids = torch.randint(0, E, (T, K), device="cuda", dtype=torch.int32)
    s, e, p, c = _align_buffers(topk_ids, E, B)
    sgl_kernel.moe_align_block_size(topk_ids, E, B, s, e, p, c, True)  # warmup/compile

    static_ids = torch.randint(0, E, (T, K), device="cuda", dtype=torch.int32)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        sgl_kernel.moe_align_block_size(static_ids, E, B, s, e, p, c, True)
    new_ids = torch.randint(0, E, (T, K), device="cuda", dtype=torch.int32)
    static_ids.copy_(new_ids)
    g.replay()
    torch.cuda.synchronize()

    ref_s, ref_e, ref_p, ref_c = _align_buffers(new_ids, E, B)
    sgl_kernel.moe_align_block_size(new_ids, E, B, ref_s, ref_e, ref_p, ref_c, True)
    torch.cuda.synchronize()
    assert torch.equal(p, ref_p)
    assert torch.equal(c, ref_c)
    assert torch.equal(e, ref_e)


# --- EP sentinel contract (bb1504f, 2026-10-13 EP-lite token-soup fix) -----
#
# EP dispatch passes topk_ids lanes of -1 ("no expert on this rank"). The
# upstream AOT kernel folds them into an internal +1-offset bucket; our
# triton path gives them their own bucket E and -- critically -- block_expert
# maps blocks from that bucket to expert id -1 so the fused_moe triton
# kernel's filter_expert branch ZEROES those output rows. Skipping the
# sentinel lanes outright would leave intermediate_cache slots unwritten and
# combine would read garbage (the original soup). These tests pin that.


def _counts_by_bucket(topk_ids, E, B):
    """Run the align kernel; return (c, e, p, n_sentinel_blocks)."""
    s, e, p, c = _align_buffers(topk_ids, E, B)
    sgl_kernel.moe_align_block_size(topk_ids, E, B, s, e, p, c, True)
    torch.cuda.synchronize()
    return s, e, p, c


@requires_gpu
def test_moe_align_ep_sentinel_bucket_membership():
    """Sentinel lanes must land in bucket E (counted! not dropped), real
    lanes in buckets 0..E-1; post_pad covers both."""
    T, K, E, B = 64, 8, 16, 16
    g = torch.Generator(device="cuda").manual_seed(13)
    topk_ids = torch.randint(0, E, (T, K), device="cuda", dtype=torch.int32,
                             generator=g)
    # poison a known subset with sentinels
    flat = topk_ids.view(-1)
    n_sent = 40
    sent_idx = torch.randperm(flat.numel(), device="cuda", generator=g)[:n_sent]
    flat[sent_idx] = -1

    s, e, p, c = _counts_by_bucket(topk_ids, E, B)

    real = topk_ids.view(-1).clone()
    real[sent_idx] = 0  # neutralize for reference counts
    ref_counts = torch.bincount(real.long(), minlength=E)
    ref_counts[0] -= n_sent  # undo the neutralization
    # c[i] is the EXCLUSIVE prefix sum of block-aligned counts over the
    # E+1 buckets (0..E-1 real experts + bucket E for sentinels); the slot
    # at c[E] holds the grand total. Bucket-E membership is therefore
    # visible as: total == sum(aligned real counts) + aligned(n_sent).
    aligned_ref = (ref_counts.to(torch.int64) + B - 1) // B * B
    total = int(p.item())
    assert total == int(aligned_ref.sum()) + (n_sent + B - 1) // B * B, (
        "padded total must cover the sentinel bucket too")
    # and the trailing cumsum slot (nb == E+1 real+sentinel buckets, slot
    # nb = grand total) mirrors the total
    assert int(c[E + 1]) == total


@requires_gpu
def test_moe_align_ep_sentinel_blocks_map_to_minus_one():
    """Blocks owned by the sentinel bucket must carry expert id -1 in
    experts_ids (NOT a valid expert, NOT num_experts padding)."""
    T, K, E, B = 64, 8, 16, 16
    g = torch.Generator(device="cuda").manual_seed(17)
    topk_ids = torch.randint(0, E, (T, K), device="cuda", dtype=torch.int32,
                             generator=g)
    flat = topk_ids.view(-1)
    flat[: B] = -1  # guarantee >=1 aligned sentinel block

    s, e, p, c = _counts_by_bucket(topk_ids, E, B)
    n_blocks = int(p.item()) // B
    assert n_blocks > 0
    sent_hits = (e[:n_blocks] == -1).sum().item()
    want_blocks = (B + B - 1) // B  # bucket E holds exactly B sentinels -> 1 block
    assert sent_hits == want_blocks, (
        f"expected {want_blocks} sentinel block(s) mapped to -1, got {sent_hits}")
    # all other in-range blocks must be a valid expert id, not -1
    other = e[:n_blocks][e[:n_blocks] != -1]
    assert ((other >= 0) & (other < E)).all()


@requires_gpu
def test_moe_align_ep_sentinel_capture_safe():
    """The sentinel contract must survive graph capture + replay with fresh
    (different) sentinel layouts, without host syncs."""
    T, K, E, B = 64, 8, 16, 16
    g = torch.Generator(device="cuda").manual_seed(19)
    topk_ids = torch.randint(0, E, (T, K), device="cuda", dtype=torch.int32,
                             generator=g)
    topk_ids.view(-1)[:8] = -1
    s, e, p, c = _align_buffers(topk_ids, E, B)
    sgl_kernel.moe_align_block_size(topk_ids, E, B, s, e, p, c, True)

    static_ids = torch.zeros((T, K), device="cuda", dtype=torch.int32)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        sgl_kernel.moe_align_block_size(static_ids, E, B, s, e, p, c, True)

    g2 = torch.Generator(device="cuda").manual_seed(23)
    new_ids = torch.randint(0, E, (T, K), device="cuda", dtype=torch.int32,
                            generator=g2)
    new_ids.view(-1)[torch.randperm(T * K, device="cuda", generator=g2)[:B]] = -1
    static_ids.copy_(new_ids)
    gr.replay()
    torch.cuda.synchronize()

    rs, re_, rp, rc = _align_buffers(new_ids, E, B)
    sgl_kernel.moe_align_block_size(new_ids, E, B, rs, re_, rp, rc, True)
    torch.cuda.synchronize()
    assert torch.equal(p, rp)
    assert torch.equal(c, rc)
    n_blocks = int(rp.item()) // B
    assert torch.equal(e[:n_blocks], re_[:n_blocks])
    assert (e[:n_blocks] == -1).sum().item() == 1
