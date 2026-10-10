"""Gated-activation kernels (silu/gelu family), GPU: contract vs torch ref.

Run on a node with a working GPU: `pytest tests/kernels/test_elementwise.py -v`.
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
