# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm-family kernels (rmsnorm / gemma / fused-add), GPU contracts.

Run on a node with a working GPU: `pytest tests/kernels/test_layernorm.py -v`.
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


# --------------------------------------------------------------- lightop ----

def _reset_fused_add_cache(ln):
    ln._FUSED_ADD_IMPL = None
    ln._LIGHTOP_FUSED_ADD = None


def _fused_add_reference(
    x, residual, weight, eps, gemma, cast_x_before_out_mul=False
):
    """Independent CPU arithmetic with the fused kernel's storage boundary."""
    total = x.cpu().double() + residual.cpu().double()
    stored = total.to(residual.dtype)
    multiplier = weight.cpu().double() + (1.0 if gemma else 0.0)
    denominator = (total.square().mean(-1, keepdim=True) + eps).sqrt()
    if cast_x_before_out_mul:
        normalized = (total / denominator).to(x.dtype).double() * multiplier
    else:
        normalized = stored.double() / denominator * multiplier
    return normalized.to(x.dtype), stored


@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_native_preserves_unrounded_variance(gemma, dtype):
    import sgl_kernel.layernorm as ln

    # The exact sum lies between adjacent activation values. Computing the
    # variance after storing the sum would incorrectly normalize it to one.
    increment = torch.finfo(dtype).eps
    x = torch.ones(3, 128, dtype=dtype)
    residual = torch.full_like(x, 1.0 + increment)
    weight = torch.ones(128, dtype=dtype)
    expected, expected_residual = _fused_add_reference(
        x, residual, weight, 1e-6, gemma
    )
    assert not torch.equal(
        expected, torch.full_like(expected, 2.0 if gemma else 1.0)
    )
    ln._fused_add_rmsnorm_native(x, residual, weight, 1e-6, gemma)
    torch.testing.assert_close(residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(x, expected, rtol=0, atol=0)


@requires_gpu
@pytest.mark.parametrize("backend", ["torch", "triton"])
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [128, 12289])
def test_fused_add_variance_fresh_input_graph(backend, gemma, dtype, width, monkeypatch):
    import sgl_kernel.layernorm as ln

    monkeypatch.setenv("HCU_FUSED_ADD_RMS_IMPL", backend)
    monkeypatch.setattr(ln, "_FUSED_ADD_IMPL", None)
    monkeypatch.setattr(ln, "_LIGHTOP_FUSED_ADD", None)
    fn = ln.gemma_fused_add_rmsnorm if gemma else ln.fused_add_rmsnorm
    # The larger width spans multiple 8192-element tiles and a masked tail.
    x = torch.ones(3, width, device="cuda", dtype=dtype)
    residual = torch.ones_like(x)
    weight = torch.ones(width, device="cuda", dtype=dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn(x, residual, weight)
    torch.cuda.current_stream().wait_stream(stream)
    assert ln._FUSED_ADD_IMPL == backend
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(x, residual, weight)
    for wave in range(3):
        x0 = torch.full((3, width), 1.0 + wave, dtype=dtype)
        r0 = torch.full_like(x0, 1.0 + torch.finfo(dtype).eps)
        w0 = torch.full((width,), 1.0 + 0.25 * wave, dtype=dtype)
        expected, expected_residual = _fused_add_reference(
            x0, r0, w0, 1e-6, gemma
        )
        for replay in (False, True):
            x.copy_(x0)
            residual.copy_(r0)
            weight.copy_(w0)
            if replay:
                graph.replay()
            else:
                fn(x, residual, weight)
            torch.testing.assert_close(
                residual.cpu(), expected_residual, rtol=0, atol=0
            )
            torch.testing.assert_close(x.cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(weight.cpu(), w0, rtol=0, atol=0)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [128, 12289])
@pytest.mark.parametrize("cast_before_weight", [False, True])
def test_community_fused_add_storage_modes_fresh_input_graph(
    dtype, width, cast_before_weight, monkeypatch
):
    import sgl_kernel.layernorm as ln
    from sglang.kernels.ops.layernorm import norm

    assert getattr(norm.fused_add_rmsnorm, "_hcu_adapter_route", False)
    monkeypatch.setenv("HCU_FUSED_ADD_RMS_IMPL", "triton")
    monkeypatch.setattr(ln, "_FUSED_ADD_IMPL", None)
    monkeypatch.setattr(ln, "_LIGHTOP_FUSED_ADD", None)
    x = torch.ones(3, width, device="cuda", dtype=dtype)
    residual = torch.ones_like(x)
    weight = torch.ones(width, device="cuda", dtype=dtype)

    def run():
        norm.fused_add_rmsnorm(
            x, residual, weight, eps=1e-6,
            cast_x_before_out_mul=cast_before_weight,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for wave in range(3):
        x0 = torch.full((3, width), 1.0 + wave, dtype=dtype)
        r0 = torch.full_like(x0, 1.0 + torch.finfo(dtype).eps)
        w0 = torch.full((width,), 1.0 + 0.25 * wave, dtype=dtype)
        expected, expected_residual = _fused_add_reference(
            x0, r0, w0, 1e-6, False, cast_before_weight
        )
        if wave == 0:
            other, _ = _fused_add_reference(
                x0, r0, w0, 1e-6, False, not cast_before_weight
            )
            assert not torch.equal(expected, other), "storage modes were not distinguished"
        for replay in (False, True):
            x.copy_(x0)
            residual.copy_(r0)
            weight.copy_(w0)
            graph.replay() if replay else run()
            torch.testing.assert_close(x.cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(residual.cpu(), expected_residual, rtol=0, atol=0)
            torch.testing.assert_close(weight.cpu(), w0, rtol=0, atol=0)


def test_fused_add_backend_default_is_triton(monkeypatch):
    """auto (=default) must stay on the triton path — the lightop vendor
    tier is opt-in until it passes the contract + graph-replay matrix."""
    monkeypatch.delenv("HCU_FUSED_ADD_RMS_IMPL", raising=False)
    import sgl_kernel.layernorm as ln

    _reset_fused_add_cache(ln)  # re-probe with the cleaned env
    try:
        assert ln._resolve_fused_add_impl() == "triton"
        assert ln._LIGHTOP_FUSED_ADD is None
    finally:
        _reset_fused_add_cache(ln)


def test_fused_add_backend_unknown_value_falls_to_triton(monkeypatch):
    monkeypatch.setenv("HCU_FUSED_ADD_RMS_IMPL", "wat")
    import sgl_kernel.layernorm as ln

    _reset_fused_add_cache(ln)
    try:
        assert ln._resolve_fused_add_impl() == "triton"
    finally:
        _reset_fused_add_cache(ln)


@requires_gpu
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_lightop_parity(gemma, dtype, monkeypatch):
    """HCU_FUSED_ADD_RMS_IMPL=lightop: same in-place contract as the triton
    path (2D and 3D leading dims; fp32 falls back by design and is covered
    by test_fused_add_rmsnorm_gpu above)."""
    monkeypatch.setenv("HCU_FUSED_ADD_RMS_IMPL", "lightop")
    import sgl_kernel.layernorm as ln

    _reset_fused_add_cache(ln)
    try:
        if ln._resolve_fused_add_impl() != "lightop":
            pytest.skip(
                "lightop backend not active (no wheel on this host "
                "or probe failed — traceback printed to stderr)")
        for shape in ((9, 512), (3, 5, 1024), (4096, 7168)):
            torch.manual_seed(4)
            N = shape[-1]
            x = torch.randn(*shape, device="cuda").to(dtype)
            residual = torch.randn(*shape, device="cuda").to(dtype)
            w = torch.randn(N, device="cuda").abs().to(dtype)
            x0, r0 = x.clone(), residual.clone()

            fn = (sgl_kernel.gemma_fused_add_rmsnorm if gemma
                  else sgl_kernel.fused_add_rmsnorm)
            assert fn(x, residual, w, eps=1e-6) is None

            rf = r0.float() + x0.float()
            ref = rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + 1e-6)
            ref = ref * (w.float() + (1.0 if gemma else 0.0))
            tol = (1e-3, 1e-3) if dtype != torch.bfloat16 else (1.0, 0.5)
            torch.testing.assert_close(
                residual.float(), rf, rtol=tol[0], atol=tol[1])
            torch.testing.assert_close(
                x.float(), ref, rtol=tol[0], atol=tol[1])
    finally:
        _reset_fused_add_cache(ln)


@requires_gpu
def test_fused_add_lightop_one_strike_fallback(monkeypatch):
    """A raising vendor call must not escape: tier disabled for the process,
    this call (and later ones) run on triton."""
    import sgl_kernel.layernorm as ln

    _reset_fused_add_cache(ln)
    try:
        saved = ln._LIGHTOP_FUSED_ADD
        ln._FUSED_ADD_IMPL = "lightop"  # force the vendor branch armed
        ln._LIGHTOP_FUSED_ADD = (
            lambda x, r, w, e: (_ for _ in ()).throw(RuntimeError("boom")),
            lambda x, r, w, e: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        torch.manual_seed(5)
        x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(256, device="cuda", dtype=torch.bfloat16).abs()
        x0, r0 = x.clone(), residual.clone()

        ln._do_fused_add(x, residual, w, 1e-6, False)  # must not raise

        assert ln._LIGHTOP_FUSED_ADD is None  # one strike → disabled
        rf = r0.float() + x0.float()
        ref = rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
        torch.testing.assert_close(
            residual.float(), rf, rtol=1.0, atol=0.5)
        torch.testing.assert_close(x.float(), ref, rtol=1.0, atol=0.5)
        ln._LIGHTOP_FUSED_ADD = saved
    finally:
        _reset_fused_add_cache(ln)
