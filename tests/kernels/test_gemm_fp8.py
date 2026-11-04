# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

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


def _block_dequant(q, scale):
    """Expand the logical DeepSeek 128x128 weight-scale contract in FP32."""
    return q.float() * scale.repeat_interleave(128, -2).repeat_interleave(128, -1)


@requires_gpu
@pytest.mark.parametrize("layout", ["masked", "contiguous"])
@pytest.mark.parametrize("n,k,strided_scales", [(128, 128, False), (256, 256, True)])
def test_fp8_grouped_expert_contract(layout, n, k, strided_scales):
    """Signed grouped GEMM, non-unit scales, padding and changing graph metadata.

    Tests the adapter Triton implementation directly. It does not claim a
    distributed DeepEP route or change its runtime selection.
    """
    from hcu_adapter.deep_gemm_fp8_triton import (
        fp8_m_grouped_gemm_nt_masked, m_grouped_fp8_gemm_nt_contiguous,
    )
    torch.manual_seed(910 + n)
    experts, capacity, rows = 3, 32, 129
    a_shape = (experts, capacity, k) if layout == "masked" else (rows, k)
    a = torch.empty(a_shape, device="cuda", dtype=FP8)
    b = torch.empty((experts, n, k), device="cuda", dtype=FP8)

    def scales(shape):
        if strided_scales:
            return torch.empty(
                (*shape[:-2], shape[-1], shape[-2]), device="cuda"
            ).transpose(-1, -2)
        return torch.empty(shape, device="cuda")

    sa = scales((*a_shape[:-1], k // 128))
    sb = scales((experts, n // 128, k // 128))
    out_shape = (experts, capacity, n) if layout == "masked" else (rows, n)
    output = torch.empty(out_shape, device="cuda", dtype=torch.bfloat16)
    metadata = torch.empty(experts if layout == "masked" else rows,
                           device="cuda", dtype=torch.int32)
    counts_by_wave = [(0, 1, 17), (1, 32, 5), (32, 0, 7)]

    def change(wave):
        a.copy_((torch.randn(a_shape, device="cuda") * (wave + 1)).to(FP8))
        b.copy_((torch.randn(b.shape, device="cuda") * 2).to(FP8))
        sa.copy_(torch.rand_like(sa) * 0.08 + 0.02)
        sb.copy_(torch.rand_like(sb) * 0.04 + 0.01)
        if layout == "masked":
            metadata.copy_(torch.tensor(counts_by_wave[wave], device="cuda"))
        else:
            # Each 64-row tile belongs to one expert; the final tile has one row.
            ids = (torch.arange(rows, device="cuda") // 64 + wave + 1) % experts
            metadata.copy_(ids)

    def run():
        output.fill_(123)
        if layout == "masked":
            return fp8_m_grouped_gemm_nt_masked(
                (a, sa), (b, sb), output, metadata, expected_m=17,
            )
        return m_grouped_fp8_gemm_nt_contiguous(
            (a, sa), (b, sb), output, metadata,
        )

    def verify():
        aq = a.float() * sa.repeat_interleave(128, -1)
        bq = _block_dequant(b, sb)
        expected = torch.full_like(output, 123, dtype=torch.float32)
        if layout == "masked":
            for expert, count in enumerate(metadata.cpu().tolist()):
                if count:
                    expected[expert, :count] = aq[expert, :count] @ bq[expert].t()
                assert torch.equal(
                    output[expert, count:],
                    torch.full_like(output[expert, count:], 123),
                )
            live = torch.arange(capacity, device="cuda")[None, :] < metadata[:, None]
            actual_live, expected_live = output[live].float(), expected[live]
        else:
            for row, expert in enumerate(metadata.cpu().tolist()):
                expected[row] = aq[row] @ bq[expert].t()
            actual_live, expected_live = output.float(), expected
        torch.testing.assert_close(actual_live, expected_live, rtol=0.05, atol=0.002)
        cosine = torch.nn.functional.cosine_similarity(
            actual_live.flatten(), expected_live.flatten(), dim=0,
        )
        assert cosine.item() >= 0.9999
        maxrel = (actual_live - expected_live).abs().max() / expected_live.abs().max()
        assert maxrel.item() <= 0.05
        return expected

    change(0)
    for _ in range(2):
        run()
    verify()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    assert captured is output
    previous = None
    for wave in range(3):
        change(wave)
        graph.replay()
        torch.cuda.synchronize()
        expected = verify()
        if previous is not None:
            assert not torch.allclose(previous, expected, rtol=0.05, atol=0.002)
        previous = expected.clone()
