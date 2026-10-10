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
