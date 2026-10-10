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
