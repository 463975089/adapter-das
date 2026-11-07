# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""flash_attn (HCU emu, fa3 backend) contract tests against a dense CPU
reference.

Covers the surfaces the sglang fa3 backend actually calls:
- flash_attn_with_kvcache: paged HND KV, GQA grouping, bottom-right causal
  windows (decode len-1 and extend multi-token cu_seqlens), fp8 KV cache
  with descale, window_size, softcap
- flash_attn_varlen_func: ragged prefill, equal and unequal q/k lens
- merge_state_v2: exact partial-state merge
"""

import pytest
import torch

from sgl_kernel.flash_attn import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    merge_state_v2,
)

PAGE = 16


def _make_paged_kv(bs, pages_per_req, H, Hk, D, Dv, seed=0):
    g = torch.Generator().manual_seed(seed)
    pool_k = torch.randn(bs * pages_per_req + 5, PAGE, Hk, D, generator=g)
    pool_v = torch.randn(bs * pages_per_req + 5, PAGE, Hk, Dv, generator=g)
    perm = torch.randperm(pool_k.shape[0], generator=g)
    block_table = torch.stack(
        [perm[i * pages_per_req : (i + 1) * pages_per_req] for i in range(bs)]
    ).int()
    seqlens = torch.randint(
        1, pages_per_req * PAGE + 1, (bs,), generator=g
    ).int()
    return pool_k, pool_v, block_table, seqlens


def _reference(q, pool_k, pool_v, block_table, seqlens, cu_q, scale, causal,
               window=(-1, -1), softcap=0.0):
    """Dense fp32 CPU reference: per-request, logical page order."""
    bs = seqlens.shape[0]
    H, D = q.shape[1], q.shape[2]
    Hk = pool_k.shape[2]
    Dv = pool_v.shape[3]
    G = H // Hk
    lens_q = (cu_q[1:] - cu_q[:-1]).tolist()
    outs, lses = [], []
    for b in range(bs):
        s = int(seqlens[b])
        Lq = lens_q[b]
        k_all = pool_k[block_table[b].long()].reshape(-1, Hk, D)[:s]  # [s,Hk,D]
        v_all = pool_v[block_table[b].long()].reshape(-1, Hk, Dv)[:s]
        qb = q[int(cu_q[b]) : int(cu_q[b + 1])].float()  # [Lq,H,D]
        sc = torch.zeros(Lq, H, s)
        for h in range(H):
            hk = h // G
            sc[:, h, :] = (qb[:, h, :] @ k_all[:, hk, :].t()) * scale
        qpos = torch.arange(s - Lq, s)[:, None]  # [Lq, 1]
        keep = torch.ones(Lq, 1, s, dtype=torch.bool)
        if causal:
            keep = keep & (torch.arange(s)[None, None, :] <= qpos[:, :, None])
        if window != (-1, -1):
            left, right = window
            rel = torch.arange(s)[None, None, :] - qpos[:, :, None]
            keep = keep & (rel >= (-left if left >= 0 else -(10**9)))
            keep = keep & (rel <= (right if right >= 0 else 10**9))
        sc = sc.masked_fill(~keep, float("-inf"))
        if softcap > 0:
            sc = softcap * torch.tanh(sc / softcap)
        p = torch.softmax(sc, dim=-1)
        out = torch.zeros(Lq, H, Dv)
        for h in range(H):
            hk = h // G
            out[:, h, :] = p[:, h, :] @ v_all[:, hk, :]
        outs.append(out)
        lses.append(torch.logsumexp(sc, dim=-1))  # [Lq, H]
    return torch.cat(outs), torch.cat(lses)


def _check_paged(name, bs, H, Hk, D, Dv, lens_q_list, pages_per_req, seed=0,
                 fp8=False, window=(-1, -1), softcap=0.0, tol=2e-2):
    torch.manual_seed(seed)
    pool_k, pool_v, block_table, seqlens = _make_paged_kv(
        bs, pages_per_req, H, Hk, D, Dv, seed=seed
    )
    if fp8:
        # pool stores fp8 exactly like set_kv_buffer with scale 1.0:
        # the emu receives an fp8-dtype tensor and must dequant on read
        pool_k = pool_k.to(torch.float8_e4m3fn)
        pool_v = pool_v.to(torch.float8_e4m3fn)
        ref_k = pool_k.to(torch.float32)
        ref_v = pool_v.to(torch.float32)
    else:
        ref_k, ref_v = pool_k, pool_v
    device = torch.device("cuda")
    pool_k = pool_k.to(device)
    pool_v = pool_v.to(device)
    block_table = block_table.to(device)
    seqlens_d = seqlens.to(device)
    maxL = max(lens_q_list)
    # right-aligned windows: every request's q window ends at its seqlen;
    # keep every seqlen >= the largest window (sglang's chunked-prefill
    # domain — a window longer than the prefix never happens there)
    seqlens_c = seqlens.clamp(min=maxL).int()
    seqlens_d = seqlens_c.to(device)
    g = torch.Generator().manual_seed(seed + 1)
    M = sum(lens_q_list)
    q = torch.randn(M, H, D, generator=g).to(torch.bfloat16).to(device)
    cu_q = torch.tensor(
        [0] + list(torch.tensor(lens_q_list).cumsum(0)), dtype=torch.int32
    ).to(device)
    scale = D**-0.5

    o = flash_attn_with_kvcache(
        q=q,
        k_cache=pool_k,
        v_cache=pool_v,
        page_table=block_table,
        cache_seqlens=seqlens_d,
        cu_seqlens_q=cu_q,
        softmax_scale=scale,
        causal=True,
        window_size=window,
        softcap=softcap,
    )
    ref, lse_ref = _reference(
        q.float().cpu(), ref_k.cpu(), ref_v.cpu(), block_table.cpu(),
        seqlens_c.cpu(), cu_q.cpu(), scale, True, window, softcap,
    )
    assert o.shape == (M, H, Dv), f"{name}: shape {o.shape}"
    diff = (o.float().cpu() - ref).abs().max().item()
    assert diff < tol, f"{name}: max diff {diff}"
    print(f"PASS {name}: shape={tuple(o.shape)} diff={diff:.5f}", flush=True)


def test_decode_gqa_bs1():
    _check_paged("decode GQA bs1", 1, 16, 4, 128, 128, [1], 8)


def test_decode_gqa_bs4_varlen():
    _check_paged("decode GQA bs4 varlen", 4, 16, 4, 128, 128, [1, 1, 1, 1], 16)


def test_extend_window5():
    _check_paged("extend win=5", 3, 8, 2, 64, 64, [5, 5, 5], 12)


def test_extend_unequal_windows():
    _check_paged("extend unequal", 3, 8, 2, 64, 64, [1, 7, 3], 12, seed=5)


def test_extend_face_peak_bounded():
    """R-13 #2 pit: the (rows, kv-chunk) score slabs must NOT grow with the
    padded window. Peak(8192-face) < 2.5x Peak(1024-face) holds only when
    the query-row sub-tile engages (QTILE_TOKENS); pre-tile it peaked at
    ~8x (22 GiB live on glm53f chunk-1 prefill -> serving OOM rank-kill at
    11k ctx). Card-independent scaling assertion, no absolute numbers."""
    torch.manual_seed(3)

    def _peak(M):
        bs, H, Hk, D, Dv = 1, 8, 4, 64, 64
        pages = M // 64 + 8
        pool_k = torch.randn(pages, 64, Hk, D, device="cuda").to(torch.bfloat16)
        pool_v = torch.randn(pages, 64, Hk, Dv, device="cuda").to(torch.bfloat16)
        bt = torch.arange(bs * pages, device="cuda", dtype=torch.int32).view(bs, pages)
        seqlens = torch.full((bs,), M, dtype=torch.int32, device="cuda")
        q = torch.randn(M, H, D, device="cuda").to(torch.bfloat16)
        cu = torch.tensor([0, M], dtype=torch.int32, device="cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        flash_attn_with_kvcache(q, pool_k, pool_v, page_table=bt,
                                cache_seqlens=seqlens, cu_seqlens_q=cu,
                                max_seqlen_q=M, causal=True)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    small = _peak(1024)
    big = _peak(8192)
    assert big < 2.5 * small, (
        f"extend face peak scales with window: M=8192 peak "
        f"{big/2**30:.2f}GiB vs M=1024 {small/2**30:.2f}GiB "
        f"(ratio {big/small:.1f} >= 2.5) — query-row sub-tile regressed?"
    )


def test_extend_fp8_kv():
    _check_paged(
        "extend fp8 KV", 3, 8, 2, 64, 64, [5, 5, 5], 12, seed=7, fp8=True,
        tol=6e-2,
    )


def test_window_size():
    _check_paged(
        "window (4,4)", 2, 8, 2, 64, 64, [3, 3], 10, seed=11, window=(4, 4)
    )


def test_softcap():
    _check_paged(
        "softcap 30", 2, 8, 2, 64, 64, [3, 3], 10, seed=13, softcap=30.0,
        tol=5e-2,
    )


def test_mla_split_qv():
    """FA3 MLA split form (sglang ROCm absorbed path, forward_absorb_rocm_core):
    q = rope query (H, Dr), qv = w_kc-absorbed latent query (H, Dv),
    logits = (q.k + qv.v) * scale, output = P @ v. MQA (Hk=1), K and V
    dims differ — mirrors DeepSeek MLA with page 64."""
    torch.manual_seed(41)
    bs, H, Hk, Dr, Dv = 3, 8, 1, 64, 512
    lens_q = [5, 1, 8]
    pages_per_req = 12
    pool_k, pool_v, block_table, seqlens = _make_paged_kv(
        bs, pages_per_req, H, Hk, Dr, Dv, seed=41
    )
    maxL = max(lens_q)
    seqlens_c = seqlens.clamp(min=maxL).int()
    device = torch.device("cuda")
    g = torch.Generator().manual_seed(42)
    M = sum(lens_q)
    q = torch.randn(M, H, Dr, generator=g).to(torch.bfloat16)
    qv = torch.randn(M, H, Dv, generator=g).to(torch.bfloat16)
    cu_q = torch.tensor(
        [0] + list(torch.tensor(lens_q).cumsum(0)), dtype=torch.int32
    )
    scale = Dr**-0.5

    o = flash_attn_with_kvcache(
        q=q.to(device),
        qv=qv.to(device),
        k_cache=pool_k.to(device),
        v_cache=pool_v.to(device),
        page_table=block_table.to(device),
        cache_seqlens=seqlens_c.to(device),
        cu_seqlens_q=cu_q.to(device),
        softmax_scale=scale,
        causal=True,
    )

    # dense reference: logits_ij = (q_i.k_j + qv_i.v_j) * scale
    outs = []
    G = H // Hk
    lens = lens_q
    for b in range(bs):
        s = int(seqlens_c[b])
        Lq = lens[b]
        k_all = pool_k[block_table[b].long()].reshape(-1, Hk, Dr)[:s]
        v_all = pool_v[block_table[b].long()].reshape(-1, Hk, Dv)[:s]
        qb = q[int(cu_q[b]) : int(cu_q[b + 1])].float()
        qvb = qv[int(cu_q[b]) : int(cu_q[b + 1])].float()
        sc = torch.zeros(Lq, H, s)
        for h in range(H):
            sc[:, h, :] = (
                qb[:, h, :] @ k_all[:, h // G, :].t()
                + qvb[:, h, :] @ v_all[:, h // G, :].t()
            ) * scale
        qpos = torch.arange(s - Lq, s)[:, None]
        sc = sc.masked_fill(
            torch.arange(s)[None, None, :] > qpos[:, :, None], float("-inf")
        )
        p = torch.softmax(sc, dim=-1)
        out = torch.zeros(Lq, H, Dv)
        for h in range(H):
            out[:, h, :] = p[:, h, :] @ v_all[:, h // G, :]
        outs.append(out)
    ref = torch.cat(outs)
    diff = (o.float().cpu() - ref).abs().max().item()
    assert o.shape == (M, H, Dv), f"mla qv: shape {o.shape}"
    assert diff < 2e-2, f"mla qv: max diff {diff}"
    print(f"PASS mla split qv (Dr={Dr},Dv={Dv},MQA): diff={diff:.5f}", flush=True)


def test_varlen_prefill():
    torch.manual_seed(21)
    H, Hk, D, Dv = 8, 2, 64, 64
    lens_q = [7, 3, 12]
    lens_k = [7, 3, 12]  # prefill: q == k
    g = torch.Generator().manual_seed(21)
    q = torch.randn(sum(lens_q), H, D, generator=g).to(torch.bfloat16)
    k = torch.randn(sum(lens_k), Hk, D, generator=g).to(torch.bfloat16)
    v = torch.randn(sum(lens_k), Hk, Dv, generator=g).to(torch.bfloat16)
    cu_q = torch.tensor([0] + list(torch.tensor(lens_q).cumsum(0)), dtype=torch.int32)
    cu_k = torch.tensor([0] + list(torch.tensor(lens_k).cumsum(0)), dtype=torch.int32)
    scale = D**-0.5
    o = flash_attn_varlen_func(
        q=q.cuda(), k=k.cuda(), v=v.cuda(),
        cu_seqlens_q=cu_q.cuda(), cu_seqlens_k=cu_k.cuda(),
        softmax_scale=scale, causal=True,
    )
    # reference
    G = H // Hk
    outs = []
    for b in range(3):
        qb = q[cu_q[b] : cu_q[b + 1]].float()
        kb = k[cu_k[b] : cu_k[b + 1]].float()
        vb = v[cu_k[b] : cu_k[b + 1]].float()
        s = lens_k[b]
        Lq = lens_q[b]
        sc = torch.zeros(Lq, H, s)
        for h in range(H):
            sc[:, h, :] = (qb[:, h, :] @ kb[:, h // G, :].t()) * scale
        qpos = torch.arange(s - Lq, s)[:, None]
        sc = sc.masked_fill(
            torch.arange(s)[None, None, :] > qpos[:, :, None], float("-inf")
        )
        p = torch.softmax(sc, dim=-1)
        out = torch.zeros(Lq, H, Dv)
        for h in range(H):
            out[:, h, :] = p[:, h, :] @ vb[:, h // G, :]
        outs.append(out)
    ref = torch.cat(outs)
    diff = (o.float().cpu() - ref).abs().max().item()
    assert o.shape == (sum(lens_q), H, Dv)
    assert diff < 2e-2, f"varlen prefill: max diff {diff}"
    print(f"PASS varlen prefill: diff={diff:.5f}", flush=True)


def test_merge_state_v2():
    torch.manual_seed(31)
    M, H, Dv = 9, 8, 64
    # two partial attentions over disjoint KV halves of one sequence
    k1 = torch.randn(M, H, Dv, generator=torch.Generator().manual_seed(31))
    k2 = torch.randn(M, H, Dv, generator=torch.Generator().manual_seed(32))
    w1 = torch.rand(M, H, generator=torch.Generator().manual_seed(33))
    w2 = torch.rand(M, H, generator=torch.Generator().manual_seed(34))
    # emulate lse = log(sum-of-weights): lse1=log(w1), o1=k1 (weighted avg of V=1 per unit)
    lse1 = torch.log(w1)
    lse2 = torch.log(w2)
    o1 = k1
    o2 = k2
    om, lsem = merge_state_v2(o1.cuda(), lse1.cuda(), o2.cuda(), lse2.cuda())
    # exact joint: o = (w1*o1 + w2*o2)/(w1+w2), lse = log(w1+w2)
    ref_o = (w1.unsqueeze(-1) * o1 + w2.unsqueeze(-1) * o2) / (
        w1 + w2
    ).unsqueeze(-1)
    ref_lse = torch.log(w1 + w2)
    d1 = (om.float().cpu() - ref_o).abs().max().item()
    d2 = (lsem.cpu() - ref_lse).abs().max().item()
    assert d1 < 1e-3 and d2 < 1e-4, f"merge: o {d1} lse {d2}"
    print(f"PASS merge_state_v2: o={d1:.6f} lse={d2:.6f}", flush=True)


# --- 3D packed-q decode normalization (ce53d06, #333: decode 400x fix) -----
#
# Real serving decode passes 3D packed q (n_tok, H, D); the triton kernel
# takes 4D (B, sq, H, D). The fa_vendor bridge normalizes, choosing B from
# cu_seqlens_q when present (FA3 form) or from the page table when not
# (metadata form), gating on n_tok == B (=uniform L=1). These tests drive
# the ROUTER face (sgl_kernel.flash_attn.flash_attn_with_kvcache, i.e. the
# installed bridge) with both 3D forms and cross-check against the
# CPU reference AND the explicit-4D call.


def _decode_fixture(bs, H, Hk, D, seed=41):
    torch.manual_seed(seed)
    pages = 4
    page = 16
    g = torch.Generator().manual_seed(seed)
    pool_k = torch.randn(bs * pages + 3, page, Hk, D, generator=g).cuda()
    pool_v = torch.randn(bs * pages + 3, page, Hk, D, generator=g).cuda()
    perm = torch.randperm(pool_k.shape[0], generator=g)
    page_table = torch.stack(
        [perm[i * pages:(i + 1) * pages] for i in range(bs)]
    ).int().cuda()
    cache_seqlens = torch.randint(3, pages * page + 1, (bs,), generator=g).int().cuda()
    q3d = torch.randn(bs, H, D, generator=g).to(torch.bfloat16).cuda()
    cu_no = None
    return pool_k, pool_v, page_table, cache_seqlens, q3d


def test_decode_3d_packed_q_with_cu():
    from sgl_kernel.paged_attn_triton import paged_attention_decode_triton

    pool_k, pool_v, pt, sl, q3d = _decode_fixture(4, 8, 2, 128)
    B, H, D = q3d.shape
    cu = torch.arange(0, B + 1, dtype=torch.int32).cuda()
    o_3d = flash_attn_with_kvcache(
        q=q3d, k_cache=pool_k, v_cache=pool_v, page_table=pt,
        cache_seqlens=sl, cu_seqlens_q=cu, softmax_scale=D**-0.5,
        causal=True,
    )
    o_4d = paged_attention_decode_triton(
        q3d.reshape(B, 1, H, D), pool_k, pool_v, pt, sl,
        softmax_scale=D**-0.5,
    )[0].reshape(B, H, D)
    assert o_3d.shape == o_4d.shape
    d = (o_3d.float() - o_4d.float()).abs().max().item()
    assert d <= 2e-2, f"3D(cu) vs 4D triton: {d}"
    print(f"PASS decode 3D packed (cu_seqlens form): router-vs-kernel {d:.5f}")


def test_decode_3d_packed_q_no_cu():
    from sgl_kernel.paged_attn_triton import paged_attention_decode_triton

    pool_k, pool_v, pt, sl, q3d = _decode_fixture(3, 16, 4, 64, seed=43)
    B, H, D = q3d.shape
    o_3d = flash_attn_with_kvcache(
        q=q3d, k_cache=pool_k, v_cache=pool_v, page_table=pt,
        cache_seqlens=sl, softmax_scale=D**-0.5, causal=True,
    )
    o_4d = paged_attention_decode_triton(
        q3d.reshape(B, 1, H, D), pool_k, pool_v, pt, sl,
        softmax_scale=D**-0.5,
    )[0].reshape(B, H, D)
    assert o_3d.shape == o_4d.shape
    d = (o_3d.float() - o_4d.float()).abs().max().item()
    assert d <= 2e-2, f"3D(no cu) vs 4D triton: {d}"
    print(f"PASS decode 3D packed (metadata form): router-vs-kernel {d:.5f}")


@pytest.mark.parametrize("length", [1, 3])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_paged_router_output_layout(length, packed, return_lse):
    """Check the router ABI before comparing independent dense-reference math."""
    pool_k, pool_v, pt, sl, q = _decode_fixture(3 * length, 8, 2, 64)
    # Use three requests with a uniform query window, including speculative
    # multi-token windows. Page rows remain distinct and noncontiguous.
    pt, sl = pt[:3], sl[:3].clamp(min=length)
    q4 = q.reshape(3, length, 8, 64)
    query = q if packed else q4
    cu = torch.arange(0, 3 * length + 1, length, dtype=torch.int32).cuda()
    result = flash_attn_with_kvcache(
        q=query, k_cache=pool_k, v_cache=pool_v, page_table=pt,
        cache_seqlens=sl, cu_seqlens_q=cu if packed else None,
        max_seqlen_q=length, softmax_scale=64**-0.5,
        causal=True, return_softmax_lse=return_lse,
    )
    actual, lse = result if return_lse else (result, None)
    expected_shape = (3 * length, 8, 64) if packed else (3, length, 8, 64)
    assert actual.shape == expected_shape
    ref, ref_lse = _reference(
        q.float().cpu(), pool_k.cpu(), pool_v.cpu(), pt.cpu(), sl.cpu(),
        cu.cpu(), 64**-0.5, True,
    )
    assert (actual.reshape_as(q).float().cpu() - ref).abs().max().item() < 2e-2
    if return_lse:
        assert lse.shape == expected_shape[:-1]
        assert lse.dtype == torch.float32
        torch.testing.assert_close(lse.reshape_as(ref_lse).cpu(), ref_lse,
                                   rtol=0, atol=2e-2)


@pytest.mark.parametrize("return_lse", [False, True])
def test_decode_packed_output_buffer(return_lse):
    pool_k, pool_v, pt, sl, q = _decode_fixture(3, 8, 2, 64)
    # A noncontiguous destination also checks that the supplied buffer is used.
    backing = torch.full((3, 8, 128), float("nan"), device=q.device, dtype=q.dtype)
    out = backing[..., ::2]
    result = flash_attn_with_kvcache(
        q=q, k_cache=pool_k, v_cache=pool_v, page_table=pt,
        cache_seqlens=sl, causal=True, out=out,
        return_softmax_lse=return_lse,
    )
    actual, lse = result if return_lse else (result, None)
    assert actual is out
    assert torch.isnan(backing[..., 1::2]).all()
    cu = torch.arange(4, dtype=torch.int32)
    ref, ref_lse = _reference(q.float().cpu(), pool_k.cpu(), pool_v.cpu(),
                              pt.cpu(), sl.cpu(), cu, 64**-0.5, True)
    assert actual.shape == ref.shape
    assert (actual.float().cpu() - ref).abs().max().item() < 2e-2
    if return_lse:
        assert lse.shape == ref_lse.shape
        torch.testing.assert_close(lse.cpu(), ref_lse, rtol=0, atol=2e-2)


def test_decode_3d_capture_safe():
    """The normalization runs inside graph capture: no host-device syncs,
    and replay tracks the refreshed q/pages."""
    pool_k, pool_v, pt, sl, q3d = _decode_fixture(4, 8, 2, 128, seed=47)
    B, H, D = q3d.shape
    static_q = torch.zeros_like(q3d)
    out = torch.empty_like(q3d)
    flash_attn_with_kvcache(  # warmup (compiles triton) off capture
        q=q3d, k_cache=pool_k, v_cache=pool_v, page_table=pt,
        cache_seqlens=sl, softmax_scale=D**-0.5, causal=True,
    )
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o, lse = flash_attn_with_kvcache(
            q=static_q, k_cache=pool_k, v_cache=pool_v, page_table=pt,
            cache_seqlens=sl, softmax_scale=D**-0.5, causal=True,
            out=out, return_softmax_lse=True,
        )
    g2 = torch.Generator().manual_seed(53)
    for _ in range(3):
        new_q = torch.randn(B, H, D, generator=g2).to(torch.bfloat16).cuda()
        static_q.copy_(new_q)
        pt.copy_(pt.roll(1, dims=0))
        sl.copy_(sl.roll(1, dims=0))
        g.replay()
        torch.cuda.synchronize()
        want, want_lse = flash_attn_with_kvcache(
            q=new_q, k_cache=pool_k, v_cache=pool_v, page_table=pt,
            cache_seqlens=sl, softmax_scale=D**-0.5, causal=True,
            return_softmax_lse=True,
        )
        assert o is out
        assert o.shape == want.shape == new_q.shape
        assert lse.shape == want_lse.shape == (B, H)
        torch.testing.assert_close(o, want, rtol=0, atol=0)
        torch.testing.assert_close(lse, want_lse, rtol=0, atol=0)
        ref, ref_lse = _reference(
            new_q.float().cpu(), pool_k.cpu(), pool_v.cpu(), pt.cpu(),
            sl.cpu(), torch.arange(B + 1, dtype=torch.int32), D**-0.5, True,
        )
        assert (o.float().cpu() - ref).abs().max().item() < 2e-2
        torch.testing.assert_close(lse.cpu(), ref_lse, rtol=0, atol=2e-2)
    print("PASS decode 3D capture+replay: replay tracks fresh inputs")


def test_only_qv_no_k_cache():
    """GLM-5.x DSA sparse decode path (dsa_backend flashmla_sparse over the
    fa3 emu via forward_absorb_prepare's only_qv=True form): k_cache=None,
    only the c_kv pool is passed as v_cache, qa/k scale folded into qv on
    the bridge so logits = (qv.folded . v) with zero K-side contribution.
    Exercises the line-342 fix (Hk derived from v_cache when k_cache is
    None) + src_k=None + total_tokens from page_table x v shape."""
    torch.manual_seed(43)
    bs, H, Dv = 3, 8, 512
    lens_q = [4, 2, 6]
    pages_per_req = 10
    # v-only pool: no K tensor at all
    _, pool_v, block_table, seqlens = _make_paged_kv(
        bs, pages_per_req, H, 1, 64, Dv, seed=43
    )
    maxL = max(lens_q)
    seqlens_c = seqlens.clamp(min=maxL).int()
    device = torch.device("cuda")
    g = torch.Generator().manual_seed(44)
    M = sum(lens_q)
    qv = torch.randn(M, H, Dv, generator=g).to(torch.bfloat16)
    cu_q = torch.tensor([0] + list(torch.tensor(lens_q).cumsum(0)), dtype=torch.int32)
    scale = Dv**-0.5

    o = flash_attn_with_kvcache(
        q=None,
        qv=qv.to(device),
        k_cache=None,
        v_cache=pool_v.to(device),
        page_table=block_table.to(device),
        cache_seqlens=seqlens_c.to(device),
        cu_seqlens_q=cu_q.to(device),
        softmax_scale=scale,
        causal=True,
    )

    # reference: logits_ij = qv_i . v_j * scale (K side absent by contract)
    outs = []
    for b in range(bs):
        s = int(seqlens_c[b])
        Lq = lens_q[b]
        v_all = pool_v[block_table[b].long()].reshape(-1, 1, Dv)[:s].squeeze(1)
        qvb = qv[int(cu_q[b]) : int(cu_q[b + 1])].float()
        sc = torch.einsum("lhd,sd->lhs", qvb, v_all) * scale
        qpos = torch.arange(s - Lq, s)[:, None]
        sc = sc.masked_fill(
            torch.arange(s)[None, None, :] > qpos[:, :, None], float("-inf")
        )
        p = torch.softmax(sc, dim=-1)
        outs.append(torch.einsum("lhs,sd->lhd", p, v_all))
    ref = torch.cat(outs)
    diff = (o.float().cpu() - ref).abs().max().item()
    assert o.shape == (M, H, Dv), f"only_qv: shape {o.shape}"
    assert diff < 2e-2, f"only_qv: max diff {diff}"
    print(f"PASS only_qv k_cache=None (DSA sparse decode): diff={diff:.5f}", flush=True)


if __name__ == "__main__":
    test_decode_gqa_bs1()
    test_decode_gqa_bs4_varlen()
    test_extend_window5()
    test_extend_unequal_windows()
    test_extend_fp8_kv()
    test_window_size()
    test_softcap()
    test_decode_3d_packed_q_with_cu()
    test_decode_3d_packed_q_no_cu()
    test_decode_3d_capture_safe()
    test_mla_split_qv()
    test_varlen_prefill()
    test_merge_state_v2()
    test_only_qv_no_k_cache()
    print("ALL PASS")
