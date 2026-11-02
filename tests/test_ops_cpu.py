"""Sanity tests that pass on any machine (CPU-only included).

They exercise the non-triton (pure torch) paths and the packaging surface;
triton kernels are covered by tests/kernels/ on HCU/CUDA hosts.
"""

import math

import pytest
import torch

import sgl_kernel
from sgl_kernel import moe as moe_mod


DTYPES = [torch.float32, torch.float16, torch.bfloat16]


# ---------------------------------------------------------------------------
# elementwise (torch fallback path on CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_silu_and_mul_cpu(dtype):
    x = torch.randn(4, 10, 32, dtype=torch.float32).to(dtype)
    y = sgl_kernel.silu_and_mul(x)
    ref = torch.nn.functional.silu(x[..., :16].float()) * x[..., 16:].float()
    torch.testing.assert_close(y.float(), ref.to(y.dtype).float(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gelu_and_mul_cpu(dtype):
    x = torch.randn(7, 24, dtype=torch.float32).to(dtype)
    y = sgl_kernel.gelu_and_mul(x)
    ref = torch.nn.functional.gelu(x[:, :12].float()) * x[:, 12:].float()
    torch.testing.assert_close(y.float(), ref.to(y.dtype).float(), rtol=1e-3, atol=1e-3)


def test_gelu_tanh_and_mul_cpu():
    x = torch.randn(3, 8, dtype=torch.float32)
    y = sgl_kernel.gelu_tanh_and_mul(x)
    ref = torch.nn.functional.gelu(x[:, :4], approximate="tanh") * x[:, 4:]
    torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-5)


def test_gelu_quick_cpu():
    x = torch.randn(5, 6, dtype=torch.float32)
    y = sgl_kernel.gelu_quick(x)
    ref = x * torch.sigmoid(1.702 * x)
    torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-5)


def test_silu_and_mul_out_kwarg():
    x = torch.randn(2, 6)
    out = torch.empty(2, 3)
    y = sgl_kernel.silu_and_mul(x, out)
    assert y is out


# ---------------------------------------------------------------------------
# layernorm (torch fallback path on CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_cpu(gemma, dtype):
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.float32).to(dtype)
    w = torch.randn(64, dtype=torch.float32).abs().to(dtype)
    fn = sgl_kernel.gemma_rmsnorm if gemma else sgl_kernel.rmsnorm
    y = fn(x, w, eps=1e-6)

    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = ref * (w.float() + (1.0 if gemma else 0.0))
    torch.testing.assert_close(y.float(), ref.to(y.dtype).float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_add_rmsnorm_cpu(dtype):
    torch.manual_seed(1)
    x = torch.randn(4, 32, dtype=torch.float32).to(dtype)
    residual = torch.randn(4, 32, dtype=torch.float32).to(dtype)
    w = torch.randn(32, dtype=torch.float32).abs().to(dtype)
    x0, r0 = x.clone(), residual.clone()

    sgl_kernel.fused_add_rmsnorm(x, residual, w, eps=1e-6)

    rf = (r0.float() + x0.float())
    ref_norm = rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    torch.testing.assert_close(residual.float(), rf.to(residual.dtype).float(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(x.float(), ref_norm.to(x.dtype).float(), rtol=2e-2, atol=2e-2)


def test_gemma_fused_add_rmsnorm_cpu():
    torch.manual_seed(2)
    x = torch.randn(2, 16)
    residual = torch.randn(2, 16)
    w = torch.rand(16)
    x0, r0 = x.clone(), residual.clone()
    sgl_kernel.gemma_fused_add_rmsnorm(x, residual, w)
    rf = r0 + x0
    ref = rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + w)
    torch.testing.assert_close(residual, rf, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# rope (torch paths on CPU) — NeoX against hand-computed pairs
# ---------------------------------------------------------------------------


def test_rope_neox_cpu():
    torch.manual_seed(3)
    T, H, D = 5, 3, 12
    rot = 8  # rotary covers the first 8 of each 12-dim head; 4-lane tail passes through
    cache = torch.randn(T * 7 + 4, rot, dtype=torch.float32)
    pos = torch.tensor([0, 3, 1, 9, 7])
    q = torch.randn(T, H * D, dtype=torch.float32)
    q0 = q.clone()

    sgl_kernel.rotary_embedding(pos, q, None, D, cache, is_neox=True)

    half = rot // 2
    for t in range(T):
        p = pos[t].item()
        c = cache[p, :half]
        s = cache[p, half:]
        for h in range(H):
            x1 = q0[t, h * D: h * D + half]
            x2 = q0[t, h * D + half: h * D + rot]
            assert torch.allclose(q[t, h * D: h * D + half], x1 * c - x2 * s, atol=1e-5)
            assert torch.allclose(q[t, h * D + half: h * D + rot], x1 * s + x2 * c, atol=1e-5)
            # pass-through tail untouched
            assert torch.equal(q[t, h * D + rot: (h + 1) * D], q0[t, h * D + rot: (h + 1) * D])


def test_rope_interleaved_cpu():
    torch.manual_seed(4)
    T, H, D = 2, 2, 8
    rot = 8
    cache = torch.randn(16, rot, dtype=torch.float32)
    pos = torch.tensor([2, 5])
    q = torch.randn(T, H * D, dtype=torch.float32)
    q0 = q.clone()
    sgl_kernel.rotary_embedding(pos, q, None, D, cache, is_neox=False)
    for t in range(T):
        p = pos[t].item()
        # sglang's cos_sin_cache is ALWAYS half-split [cos | sin] regardless
        # of neox style (fbc540c); lanes are interleaved, the cache is not.
        cos_i = cache[p, : rot // 2]
        sin_i = cache[p, rot // 2:]
        for h in range(H):
            x1 = q0[t, h * D: h * D + rot: 2]
            x2 = q0[t, h * D + 1: h * D + rot: 2]
            assert torch.allclose(q[t, h * D: h * D + rot: 2], x1 * cos_i - x2 * sin_i, atol=1e-5)
            assert torch.allclose(q[t, h * D + 1: h * D + rot: 2], x1 * sin_i + x2 * cos_i, atol=1e-5)


# ---------------------------------------------------------------------------
# moe gate — reference oracle implemented from the CUDA kernel spec
# ---------------------------------------------------------------------------


def _oracle_softmax_topk(logits, topk, renormalize, softcap, bias):
    logits = logits.float()
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    act = torch.softmax(logits, dim=-1, dtype=torch.float32)
    biased = logits + bias.float() if bias is not None else logits
    _, ids = torch.topk(biased, k=topk, dim=-1, sorted=False)
    w = act.gather(1, ids)
    if renormalize:
        w = w / w.sum(-1, keepdim=True)
    return w, ids.to(torch.int32)


@pytest.mark.parametrize("renorm", [False, True])
@pytest.mark.parametrize("cap", [0.0, 30.0])
@pytest.mark.parametrize("use_bias", [False, True])
def test_topk_softmax_matches_oracle(renorm, cap, use_bias):
    torch.manual_seed(5)
    M, E, K = 16, 64, 8
    logits = torch.randn(M, E) * 5
    bias = torch.randn(E) if use_bias else None
    w = torch.empty(M, K, dtype=torch.float32)
    ids = torch.empty(M, K, dtype=torch.int32)
    sgl_kernel.topk_softmax(w, ids, logits, renormalize=renorm,
                            moe_softcapping=cap, correction_bias=bias)
    rw, ri = _oracle_softmax_topk(logits, K, renorm, cap, bias)
    # same (weight, id) *set* per row; order may differ across equal values
    ws, _ = torch.sort(w, dim=-1)
    rws, _ = torch.sort(rw, dim=-1)
    torch.testing.assert_close(ws, rws, rtol=1e-4, atol=1e-5)
    assert torch.equal(torch.sort(ids, dim=-1).values, torch.sort(ri, dim=-1).values)


def test_topk_sigmoid_with_shared_expert():
    torch.manual_seed(6)
    M, E, K = 8, 32, 4
    logits = torch.randn(M, E)
    bias = torch.randn(E)
    w = torch.empty(M, K, dtype=torch.float32)
    ids = torch.empty(M, K, dtype=torch.int32)
    scale = 2.5
    sgl_kernel.topk_sigmoid(w, ids, logits, renormalize=True,
                            correction_bias=bias, routed_scaling_factor=scale,
                            num_fused_shared_experts=1)
    act = torch.sigmoid(logits)
    biased = act + bias
    _, rids = torch.topk(biased, k=K - 1, dim=-1, sorted=False)
    rw = act.gather(1, rids)
    # kernel order: shared = routed_sum/scale; renorm by ROUTED sum only; scale all
    routed_sum = rw.sum(-1, keepdim=True)
    shared = routed_sum / scale
    rw = torch.cat([rw, shared], dim=1)
    rw = rw / routed_sum * scale
    rs, _ = torch.sort(rw, dim=-1)
    ws, _ = torch.sort(w, dim=-1)
    torch.testing.assert_close(ws, rs, rtol=1e-4, atol=1e-5)
    assert (ids[:, -1] == E).all()


def test_moe_fused_gate_grouped_matches_native():
    """Grouped routing vs the torch reference in sglang topk.py (masked_fill
    + group-mask + topk), which is the semantics of the DSV3 noaux_tc kernel."""
    torch.manual_seed(7)
    M, E, K, G, KG = 12, 64, 8, 8, 4
    logits = torch.randn(M, E)
    bias = torch.randn(E)

    scores = torch.sigmoid(logits)
    biased = scores + bias
    gs = biased.view(M, G, E // G).topk(2, dim=-1).values.sum(-1)
    keep = gs.topk(KG, dim=-1, sorted=False).indices
    gm = torch.zeros_like(gs, dtype=torch.bool).scatter(1, keep, True)
    em = gm.unsqueeze(-1).expand(M, G, E // G).reshape(M, E)
    masked = biased.masked_fill(~em, float("-inf"))
    _, rids = torch.topk(masked, k=K, dim=-1, sorted=False)
    rw = scores.gather(1, rids)
    rw = rw / rw.sum(-1, keepdim=True)

    w, ids = sgl_kernel.moe_fused_gate(
        logits, bias, topk=K, scoring_func="sigmoid", renormalize=True,
        num_expert_group=G, topk_group=KG,
    )
    assert w.dtype == torch.float32 and ids.dtype == torch.int32
    torch.testing.assert_close(
        torch.sort(w, dim=-1).values, torch.sort(rw, dim=-1).values, rtol=1e-4, atol=1e-5
    )
    # group constraint holds
    for m in range(M):
        groups = ids[m] // (E // G)
        assert len(set(groups.tolist())) >= 2 or KG == 1
    assert torch.equal(torch.sort(ids, dim=-1).values, torch.sort(rids, dim=-1).values)


def test_moe_fused_gate_legacy_signature():
    torch.manual_seed(8)
    M, E, K = 4, 16, 2
    logits = torch.randn(M, E)
    w1, i1 = sgl_kernel.moe_fused_gate(logits, None, 1, 1, K, renormalize=False,
                                       scoring_func="softmax",
                                       routed_scaling_factor=1.0,
                                       apply_routed_scaling_factor_on_output=False)
    w2, i2 = sgl_kernel.moe_fused_gate(logits, None, topk=K, scoring_func="softmax",
                                       renormalize=False, num_expert_group=1, topk_group=1)
    torch.testing.assert_close(w1, w2)
    assert torch.equal(i1, i2)


# ---------------------------------------------------------------------------
# moe_align_block_size — contract test from the CUDA kernel spec
# ---------------------------------------------------------------------------


def test_moe_align_block_size_contract():
    topk_ids = torch.tensor([[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], dtype=torch.int32)
    E, B = 4 + 1, 4  # +1 sentinel expert like sglang's callers pass
    numel = topk_ids.numel()
    max_padded = numel + (E + 1) * (B - 1)
    sorted_ids = torch.empty(max_padded, dtype=torch.int32)
    n_blocks = (max_padded + B - 1) // B
    expert_ids = torch.empty(n_blocks, dtype=torch.int32)
    post_pad = torch.empty(1, dtype=torch.int32)
    cumsum = torch.empty(E + 2, dtype=torch.int32)

    moe_mod.moe_align_block_size(topk_ids, E, B, sorted_ids, expert_ids,
                                 post_pad, cumsum, True)

    counts = torch.bincount(topk_ids.flatten().long(), minlength=E)
    blk_counts = (counts + B - 1) // B * B
    assert post_pad.item() == int(blk_counts.sum())

    valid = sorted_ids[sorted_ids < numel]
    # tokens grouped by expert, stable within expert
    e_of = topk_ids.flatten().long()[valid.long()]
    assert torch.equal(e_of, torch.sort(e_of).values)
    # padding fills everything else
    assert int((sorted_ids == numel).sum()) == len(sorted_ids) - len(valid)
    # block -> expert map
    for b in range(int(post_pad.item()) // B):
        s = b * B
        exps = expert_ids[b]
        seg = sorted_ids[s: s + B]
        real = seg[seg < numel]
        if len(real):
            assert exps == topk_ids.flatten().long()[real[0]]
        else:
            assert exps == E


def test_moe_align_invalid_expert_sentinel_bucket():
    topk_ids = torch.tensor([[0, -1], [2, 1]], dtype=torch.int32)
    E, B = 4, 2
    numel = 4
    sorted_ids = torch.full((32,), -1, dtype=torch.int32)
    expert_ids = torch.empty(16, dtype=torch.int32)
    post_pad = torch.empty(1, dtype=torch.int32)
    cumsum = torch.empty(E + 2, dtype=torch.int32)
    moe_mod.moe_align_block_size(topk_ids, E, B, sorted_ids, expert_ids,
                                 post_pad, cumsum, True, ignore_invalid_expert=True)
    # AOT/sentinel contract (bb1504f EP-lite fix; pinned GPU-side in
    # tests/kernels/test_moe_align.py): the -1 lane is NOT dropped -- it
    # lands in the dedicated last bucket so its cache row still gets
    # written. experts [0,2,1] + the sentinel each own one block -> 4*2 = 8
    assert post_pad.item() == 8
    valid = sorted_ids[sorted_ids < numel]
    assert sorted(valid.tolist()) == [0, 1, 2, 3]
    used_blocks = post_pad.item() // B
    # the sentinel-owned block carries expert id -1 (filter_expert zeroes it)
    assert expert_ids[:used_blocks].tolist() == [0, 1, 2, -1]
    assert (expert_ids[used_blocks:] == E + 1).all()


def test_moe_sum_and_reduce():
    x = torch.randn(4, 3, 8)
    out = torch.empty(4, 8)
    sgl_kernel.moe_sum(x, out)
    torch.testing.assert_close(out, x.sum(1), rtol=1e-5, atol=1e-6)

    out2 = torch.empty(4, 8)
    sgl_kernel.moe_sum_reduce(x, out2, 2.0)
    torch.testing.assert_close(out2, x.sum(1) * 2.0, rtol=1e-5, atol=1e-6)


def test_import_surface():
    for name in sgl_kernel.__all__:
        assert hasattr(sgl_kernel, name), name
    assert sgl_kernel.__version__


# ---------------------------------------------------------------------------
# kvcacheio hcu_* passthrough surface (vendor dtk wheel ops)
# ---------------------------------------------------------------------------

KVCACHEIO_HCU_OPS = [
    "hcu_align_evict_mask_to_page_size",
    "hcu_alloc_decode_kernel",
    "hcu_alloc_extend_kernel",
    "hcu_assign_extend_cache_locs",
    "hcu_assign_req_to_token_pool",
    "hcu_create_chunked_prefix_cache_kv_indices",
    "hcu_create_extend_after_decode_spec_info",
    "hcu_get_last_loc",
]


def test_kvcacheio_hcu_op_surface():
    """Shadow kvcacheio mirrors the dtk wheel's hcu_* allocation wrappers.

    The torch ops themselves are registered by the vendor sglang-kernel
    library; only the python surface is ours. Losing these wrappers broke
    consumers at runtime (AttributeError on sgl_kernel.kvcacheio, raised
    only when the matching scheduling path runs), so pin the surface here.
    """
    from sgl_kernel import kvcacheio

    for name in KVCACHEIO_HCU_OPS:
        assert name in kvcacheio.__all__, name
        assert callable(getattr(kvcacheio, name)), name
        # where the vendor library is loaded, every wrapper must resolve to
        # its registered op (no stale/renamed mirror); skip on clean hosts
        op_ns = torch.ops.sgl_kernel
        if hasattr(op_ns, name):
            assert getattr(op_ns, name) is not None, name
