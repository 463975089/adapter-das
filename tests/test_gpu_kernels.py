"""GPU (HCU/CUDA) kernel tests: triton paths vs the same-op torch reference.

Run on a node with a working GPU: `pytest tests/test_gpu_kernels.py -v`.
Skipped automatically when no GPU is visible.
"""

import pytest
import torch

import sgl_kernel

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0),
    reason="requires HCU/CUDA device",
)

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@requires_gpu
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("act", ["silu", "gelu", "gelu_tanh"])
def test_gated_act_gpu(dtype, act):
    torch.manual_seed(0)
    D = 512
    x = (torch.randn(33, D * 2, device="cuda") * 3).to(dtype)
    fn = {
        "silu": sgl_kernel.silu_and_mul,
        "gelu": sgl_kernel.gelu_and_mul,
        "gelu_tanh": sgl_kernel.gelu_tanh_and_mul,
    }[act]
    y = fn(x)
    gate, up = x[:, :D].float(), x[:, D:].float()
    if act == "silu":
        ref = gate * torch.sigmoid(gate)
    elif act == "gelu":
        ref = torch.nn.functional.gelu(gate)
    else:
        ref = torch.nn.functional.gelu(gate, approximate="tanh")
    ref = ref * up
    # fp32 accumulate + single final round vs torch's double-rounded bf16 ref
    # can differ by 1 ULP (bf16 ULP = 1/128 relative at exponent 32..64).
    tol = {"torch.float32": (2e-3, 2e-3), torch.float32: (2e-3, 2e-3),
           torch.float16: (2e-3, 2e-3), torch.bfloat16: (1.0, 0.3)}[dtype]
    torch.testing.assert_close(y.float(), ref, rtol=tol[0], atol=tol[1])


@requires_gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_gated_act_gpu_3d_out_kwarg(dtype):
    torch.manual_seed(1)
    x = torch.randn(2, 7, 256, device="cuda").to(dtype)
    out = torch.empty(2, 7, 128, device="cuda", dtype=dtype)
    y = sgl_kernel.silu_and_mul(x, out)
    ref = torch.nn.functional.silu(x[..., :128].float()) * x[..., 128:].float()
    tol = (2e-3, 2e-3) if dtype != torch.bfloat16 else (1.0, 0.3)
    torch.testing.assert_close(y.float(), ref, rtol=tol[0], atol=tol[1])
    assert y.data_ptr() == out.data_ptr()


@requires_gpu
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_gpu(gemma, dtype):
    torch.manual_seed(2)
    N = 4096
    x = torch.randn(17, N, device="cuda").to(dtype)
    w = torch.randn(N, device="cuda").abs().to(dtype)
    fn = sgl_kernel.gemma_rmsnorm if gemma else sgl_kernel.rmsnorm
    y = fn(x, w, eps=1e-6)

    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = ref * (w.float() + (1.0 if gemma else 0.0))
    tol = (2e-2, 2e-2) if dtype != torch.bfloat16 else (6e-2, 4e-2)
    torch.testing.assert_close(y.float(), ref, rtol=tol[0], atol=tol[1])


@requires_gpu
def test_rmsnorm_out_kwarg_and_big_hidden():
    torch.manual_seed(3)
    N = 7168  # deepseek-wide hidden, exercises the row loop
    x = torch.randn(5, N, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, device="cuda").abs().to(torch.bfloat16)
    y = sgl_kernel.rmsnorm(x, w, 1e-6)
    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    torch.testing.assert_close(y.float(), ref, rtol=2e-2, atol=2e-2)


@requires_gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_add_rmsnorm_gpu(dtype):
    torch.manual_seed(4)
    N = 512
    x = torch.randn(9, N, device="cuda").to(dtype)
    residual = torch.randn(9, N, device="cuda").to(dtype)
    w = torch.randn(N, device="cuda").abs().to(dtype)
    x0, r0 = x.clone(), residual.clone()

    ret = sgl_kernel.fused_add_rmsnorm(x, residual, w, eps=1e-6)
    assert ret is None

    rf = r0.float() + x0.float()
    ref = rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    tol = (1e-3, 1e-3) if dtype != torch.bfloat16 else (1.0, 0.5)
    torch.testing.assert_close(residual.float(), rf, rtol=tol[0], atol=tol[1])
    torch.testing.assert_close(x.float(), ref, rtol=tol[0], atol=tol[1])


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rope_neox_gpu(dtype):
    torch.manual_seed(5)
    T, qH, kvH, D, rot = 37, 8, 4, 128, 64
    cache = torch.randn(128, rot, device="cuda", dtype=torch.float32)
    pos = torch.randint(0, 128, (T,), device="cuda")
    q = torch.randn(T, qH * D, device="cuda").to(dtype)
    k = torch.randn(T, kvH * D, device="cuda").to(dtype)
    q0, k0 = q.clone(), k.clone()

    sgl_kernel.rotary_embedding(pos, q, k, D, cache, is_neox=True)

    for t in (0, 5, 36):
        p = pos[t].item()
        c = cache[p, : rot // 2]
        s = cache[p, rot // 2:]
        for base0, nH, t0 in ((q0, qH, q), (k0, kvH, k)):
            for h in range(nH):
                off = h * D
                x1 = base0[t, off: off + rot // 2].float()
                x2 = base0[t, off + rot // 2: off + rot].float()
                torch.testing.assert_close(
                    t0[t, off: off + rot // 2].float(), x1 * c - x2 * s,
                    rtol=2e-2, atol=2e-2,
                )
                torch.testing.assert_close(
                    t0[t, off + rot // 2: off + rot].float(), x1 * s + x2 * c,
                    rtol=2e-2, atol=2e-2,
                )


@requires_gpu
def test_topk_softmax_gpu_matches_cpu():
    torch.manual_seed(6)
    M, E, K = 64, 256, 8
    logits = (torch.randn(M, E, device="cuda") * 4).to(torch.bfloat16)
    bias = torch.randn(E, device="cuda", dtype=torch.float32)
    w = torch.empty(M, K, device="cuda", dtype=torch.float32)
    ids = torch.empty(M, K, device="cuda", dtype=torch.int32)
    sgl_kernel.topk_softmax(w, ids, logits, renormalize=True,
                            moe_softcapping=0.0, correction_bias=bias)

    lf = logits.float()
    act = torch.softmax(lf, dim=-1)
    _, rids = torch.topk(lf + bias, k=K, dim=-1, sorted=False)
    rw = act.gather(1, rids)
    rw = rw / rw.sum(-1, keepdim=True)
    torch.testing.assert_close(
        torch.sort(w, dim=-1).values, torch.sort(rw, dim=-1).values, rtol=1e-3, atol=1e-4
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


@requires_gpu
def test_moe_fused_gate_gpu_grouped():
    torch.manual_seed(8)
    M, E, K, G, KG = 32, 64, 8, 8, 4
    logits = torch.randn(M, E, device="cuda")
    bias = torch.randn(E, device="cuda")
    w, ids = sgl_kernel.moe_fused_gate(
        logits, bias, topk=K, scoring_func="sigmoid", renormalize=True,
        num_expert_group=G, topk_group=KG,
    )
    assert w.shape == (M, K) and ids.shape == (M, K)
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
    torch.testing.assert_close(
        torch.sort(w, dim=-1).values, torch.sort(rw, dim=-1).values, rtol=1e-4, atol=1e-5
    )
    assert torch.equal(torch.sort(ids, dim=-1).values, torch.sort(rids, dim=-1).values)
