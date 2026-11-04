# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""flash_mla (HCU emu) contract tests against a dense einsum reference.

The emu must agree with a plain masked-GEMM reference for:
- decode (s_q == 1, any bs), the hot path
- extend (s_q > 1) with right-aligned causal window
- paged KV with shuffled page order + gaps (the real req_to_token layout)
- head-count and dtype passthrough (bf16 out)
"""

import torch

from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

PAGE = 64


def _make_paged_kv(bs, pages_per_req, d_qk=576, d_v=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    kv_pool = torch.randn(bs * pages_per_req + 7, PAGE, 1, d_qk, generator=g)
    # shuffled, gapped page assignment (like req_to_token)
    perm = torch.randperm(kv_pool.shape[0], generator=g)
    block_table = torch.stack(
        [perm[i * pages_per_req : (i + 1) * pages_per_req] for i in range(bs)]
    ).int()
    seqlens = torch.randint(
        1, pages_per_req * PAGE + 1, (bs,), generator=g
    ).int()
    return kv_pool.cuda(), block_table.cuda(), seqlens.cuda()


def _reference(q, kv_pool, block_table, seqlens, d_v, scale, causal):
    """Pure-torch dense reference; runs on CPU (einsum HIP path is bugged)
    with the exact same math the emu must reproduce."""
    bs, win, h, d_qk = q.shape
    q_cpu = q.float().cpu()
    kv_cpu = kv_pool.float().cpu()
    bt_cpu = block_table.long().cpu()
    outs, lses = [], []
    for b in range(bs):
        k_all = kv_cpu[bt_cpu[b], :, 0, :].reshape(-1, d_qk)  # [P*PAGE, d_qk]
        s = int(seqlens[b])
        k = k_all[:s]
        v = k[:, :d_v]
        qb = q_cpu[b]  # [win, h, d_qk]
        scores = torch.einsum("whd,sd->hws", qb, k) * scale
        if causal:
            qpos = torch.arange(s - win, s)[:, None]
            scores = scores.masked_fill(
                torch.arange(s)[None, :] > qpos, float("-inf")
            )
        p = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hws,sd->whd", p, v))  # [win, h, d_v]
        lses.append(torch.logsumexp(scores, dim=-1))  # [h, win]
    return (
        torch.stack(outs).to(q.dtype),  # [bs, win, h, d_v]
        torch.stack(lses),  # [bs, h, win]
    )


def _check(name, bs, win, pages_per_req, seed=0):
    torch.manual_seed(seed)
    kv_pool, block_table, seqlens = _make_paged_kv(bs, pages_per_req, seed=seed)
    if win > 1:
        # the extend window never exceeds the sequence in real sglang
        # chunked prefill; keep the generator inside that domain
        seqlens = seqlens.clamp(min=int(win))
    h, d_qk, d_v = 16, 576, 512
    q = torch.randn(bs, win, h, d_qk, generator=torch.Generator().manual_seed(seed + 1)).to(
        torch.bfloat16
    ).cuda()
    scale = d_qk**-0.5

    meta, splits = get_mla_metadata(seqlens, h, 1)
    assert meta.shape == (1, 8) and splits.shape == (bs + 1,), (
        f"{name}: metadata shapes {meta.shape} {splits.shape}"
    )
    o, lse = flash_mla_with_kvcache(
        q=q,
        k_cache=kv_pool.view(-1, PAGE, 1, d_qk),
        block_table=block_table,
        cache_seqlens=seqlens,
        head_dim_v=d_v,
        tile_scheduler_metadata=meta,
        num_splits=splits,
        softmax_scale=scale,
        causal=True,
    )
    ref, lse_ref = _reference(q, kv_pool, block_table, seqlens, d_v, scale, True)
    ref, lse_ref = ref.cuda(), lse_ref.cuda()
    # rank follows q (146a875, dsa _forward_flashmla_kv needs 4D out to
    # head-slice): 4D q in -> o [bs, win, h, d_v] and lse [bs, h, win] --
    # both match the reference layout directly, no reshape needed
    assert o.shape == ref.shape, f"{name}: out shape {o.shape} vs {ref.shape}"
    diff = (o.float() - ref.float()).abs().max().item()
    lse_diff = (lse - lse_ref).abs().max().item()
    assert diff < 2e-2, f"{name}: max diff {diff}"
    assert lse_diff < 5e-2, f"{name}: lse diff {lse_diff}"
    print(f"PASS {name}: shape={tuple(o.shape)} diff={diff:.5f} lse_diff={lse_diff:.5f}")


def test_decode_bs1():
    _check("decode bs=1", 1, 1, 8)


def test_decode_bs4_varlen():
    _check("decode bs=4 varlen", 4, 1, 16)


def test_decode_long_ctx():
    _check("decode long ctx", 2, 1, 64)


def test_extend_window5():
    _check("extend win=5", 3, 5, 12)


def test_extend_window33():
    _check("extend win=33", 2, 33, 4)


def _smoke_2d_q():
    # target-verify style: 2D q (M, H, d) reshaped by the caller path
    torch.manual_seed(3)
    kv_pool, block_table, seqlens = _make_paged_kv(2, 4, seed=3)
    h, d_qk, d_v = 8, 576, 512
    q = torch.randn(6, h, d_qk).to(torch.bfloat16).cuda()
    o, _ = flash_mla_with_kvcache(
        q=q,
        k_cache=kv_pool.view(-1, PAGE, 1, d_qk),
        block_table=block_table,
        cache_seqlens=seqlens,
        head_dim_v=d_v,
        softmax_scale=d_qk**-0.5,
        causal=True,
    )
    assert o.shape[0] == 6 and o.shape[1:] == (h, d_v)
    print("PASS 2d q smoke")


if __name__ == "__main__":
    test_decode_bs1()
    test_decode_bs4_varlen()
    test_decode_long_ctx()
    test_extend_window5()
    test_extend_window33()
    _smoke_2d_q()
    print("ALL PASS")
