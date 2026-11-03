"""int8_scaled_mm (per-token x per-channel INT8 GEMM), GPU contracts.

Reference math: fp32 dequant (matmul in int8->i32 then scale) — same style
as the fp8 twin. Unblocks CompressedTensorsW8A8Int8 (Channel-INT8-w8a8
checkpoint family) on HCU: the scheme's apply_weights calls
sgl_kernel.int8_scaled_mm directly.

Run: pytest tests/kernels/test_gemm_int8.py -v (HCU node).
"""

import pytest
import torch

import sgl_kernel

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires HCU/CUDA device",
)

# (M, N, K, bias): decode M, graph-pad M, odd dims, TP slice
SHAPES = [
    (1, 512, 512, False),
    (17, 1024, 512, False),
    (64, 2048, 1024, True),
    (129, 512, 4096, False),
    (1, 128, 128, True),
    (33, 256, 448, False),
]
OUT_DTYPES = [torch.bfloat16, torch.float16]


def _ref(a, b, sa, sb, out_dtype, bias):
    o = torch.matmul(a.float(), b.float())
    o = o * sa.view(-1, 1) * sb.view(1, -1)
    o = o.to(out_dtype)
    if bias is not None:
        o = o + bias.view(1, -1)
    return o


def _mk(M, N, K, wave=0):
    torch.manual_seed(M * 5 + N * 3 + K + wave * 100003)
    a = torch.randint(-127, 128, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(-127, 128, (N, K), device="cuda", dtype=torch.int8).t()
    sa = (torch.randn(M, device="cuda") * 0.01).float()
    sb = (torch.randn(N, device="cuda") * 0.01).float()
    return a, b, sa, sb


@requires_gpu
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("M,N,K,with_bias", SHAPES)
def test_int8_scaled_mm_contract(out_dtype, M, N, K, with_bias):
    a, b, sa, sb = _mk(M, N, K)
    bias = torch.randn(N, device="cuda", dtype=out_dtype) if with_bias else None
    got = sgl_kernel.int8_scaled_mm(a, b, sa, sb, out_dtype, bias)
    ref = _ref(a, b, sa, sb, out_dtype, bias)
    assert got.shape == (M, N) and got.dtype == out_dtype
    # int8 exact dot + fp32 scales: only output-rounding noise remains;
    # abs bound 2e-2 covers near-zero rel blowups.
    torch.testing.assert_close(got.float(), ref.float(), rtol=5e-2, atol=2e-2)


@requires_gpu
def test_int8_scaled_mm_scale_shapes():
    """(M,1)/(N,1) column forms from CT parameters must work like flat."""
    M, N, K = 9, 256, 512
    a, b, sa, sb = _mk(M, N, K)
    got = sgl_kernel.int8_scaled_mm(a, b, sa.view(-1, 1), sb.view(-1, 1),
                                    torch.bfloat16)
    ref = _ref(a, b, sa, sb, torch.bfloat16, None)
    torch.testing.assert_close(got.float(), ref.float(), rtol=5e-2, atol=2e-2)


@requires_gpu
def test_int8_scaled_mm_graph_replay_tracks_fresh_inputs():
    M, N, K = 32, 1024, 512
    a, b, sa, sb = _mk(M, N, K)
    for _ in range(2):
        sgl_kernel.int8_scaled_mm(a, b, sa, sb, torch.bfloat16)
    torch.cuda.synchronize()

    static_a = a.clone()
    static_sa = sa.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = sgl_kernel.int8_scaled_mm(static_a, b, static_sa, sb,
                                        torch.bfloat16)
    previous = out.clone()
    for wave in range(1, 4):
        new_a, _, new_sa, _ = _mk(M, N, K, wave=wave)
        assert not torch.equal(new_a.float(), static_a.float())
        static_a.copy_(new_a)
        static_sa.copy_(new_sa)
        g.replay()
        torch.cuda.synchronize()
        ref = _ref(new_a, b, new_sa, sb, torch.bfloat16, None)
        assert not torch.allclose(previous.float(), ref.float(), rtol=5e-2, atol=2e-2)
        torch.testing.assert_close(out.float(), ref.float(), rtol=5e-2, atol=2e-2)
        previous = out.clone()
