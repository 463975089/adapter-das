"""fp8_scaled_mm (row-scaled A x col-scaled B FP8 GEMM), GPU contracts.

Official-op parity reference follows sglang/kernels/aot/tests/test_fp8_gemm.py
torch_scaled_mm (fp32 dequant math). Pinning this op is what unblocks the
v0.5.21 fp8 linear chain on HCU: Fp8ScaledMMOp.forward_aot ->
sgl_kernel.fp8_scaled_mm is the only backend reachable on HIP (TORCH needs
SM90), and apply_fp8_linear's CHANNEL path lands here.

Run: pytest tests/kernels/test_gemm_fp8.py -v (HCU node).
"""

import pytest
import torch

import sgl_kernel

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires HCU/CUDA device",
)

FP8 = torch.float8_e4m3fn

# (M, N, K, bias) — decode M, graph-pad M, prefill M; odd N/K; TP slices
SHAPES = [
    (1, 512, 512, False),
    (17, 1024, 512, False),
    (64, 2048, 1024, True),
    (129, 512, 4096, False),
    (1, 128, 128, True),
    (33, 256, 448, False),
]
OUT_DTYPES = [torch.bfloat16, torch.float16]


def _ref(a_fp8, b_fp8, sa, sb, out_dtype, bias):
    o = torch.matmul(a_fp8.float(), b_fp8.float())
    o = o * sa.view(-1, 1) * sb.view(1, -1)
    o = o.to(out_dtype)
    if bias is not None:
        o = o + bias.view(1, -1)
    return o


def _mk_case(M, N, K, wave=0):
    torch.manual_seed(M * 7 + N * 3 + K + wave * 100003)
    fp8_max = torch.finfo(FP8).max
    a = ((torch.rand(M, K, device="cuda") - 0.5) * 2 * fp8_max).to(FP8)
    b = ((torch.rand(N, K, device="cuda") - 0.5) * 2 * fp8_max).to(FP8).t()
    sa = (torch.randn(M, device="cuda") * 0.001).float()
    sb = (torch.randn(N, device="cuda") * 0.001).float()
    return a, b, sa, sb


@requires_gpu
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("M,N,K,with_bias", SHAPES)
def test_fp8_scaled_mm_contract(out_dtype, M, N, K, with_bias):
    a, b, sa, sb = _mk_case(M, N, K)
    bias = torch.randn(N, device="cuda", dtype=out_dtype) if with_bias else None
    got = sgl_kernel.fp8_scaled_mm(a, b, sa, sb, out_dtype, bias)
    ref = _ref(a, b, sa, sb, out_dtype, bias)
    assert got.shape == (M, N) and got.dtype == out_dtype
    # bf16/fp16 output rounding dominates: values ~1e-3*randn*K-col-scales
    # land near zero where rel error explodes — bound by abs 2e-2 (one bf16
    # ULP at magnitude ~2) plus rel for the bulk.
    torch.testing.assert_close(got.float(), ref.float(), rtol=5e-2, atol=2e-2)


@requires_gpu
def test_fp8_scaled_mm_graph_replay_tracks_fresh_inputs():
    """Decode capture/replay must follow updated A/scales (no baking)."""
    M, N, K = 32, 1024, 512
    a, b, sa, sb = _mk_case(M, N, K)
    for _ in range(2):
        sgl_kernel.fp8_scaled_mm(a, b, sa, sb, torch.bfloat16)  # warmup/compile
    torch.cuda.synchronize()

    static_a = a.clone()
    static_sa = sa.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = sgl_kernel.fp8_scaled_mm(
            static_a, b, static_sa, sb, torch.bfloat16)
    previous = out.clone()
    for wave in range(1, 4):
        new_a, _, new_sa, _ = _mk_case(M, N, K, wave=wave)
        assert not torch.equal(new_a.float(), static_a.float())
        static_a.copy_(new_a)
        static_sa.copy_(new_sa)
        g.replay()
        torch.cuda.synchronize()
        ref = _ref(new_a, b, new_sa, sb, torch.bfloat16, None)
        assert not torch.allclose(previous.float(), ref.float(), rtol=5e-2, atol=1e-3)
        torch.testing.assert_close(out.float(), ref.float(), rtol=5e-2, atol=1e-3)
        previous = out.clone()
