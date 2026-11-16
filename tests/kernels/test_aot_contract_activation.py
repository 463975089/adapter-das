# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""AOT contract anchors: gated activations + quick-gelu.

Contract sources (upstream sgl-kernel v0.5.22 tag, exported under
tmp/aot-v0522/stripped/):
  - csrc/elementwise/activation.cu:56-83   math constants:
        silu       = x / (1 + exp(-x))
        gelu(erf)  = x * 0.5 * (1 + erf(x * M_SQRT1_2)),  M_SQRT1_2 = 0.7071067811865476
        gelu_quick = x / (1 + exp(-1.702 * x))
        gelu_tanh  = 0.5x(1 + tanh(0.7978845608028654 * (x + 0.044715 x^3)))
  - include/utils.h:306-318   DISPATCH_..._FP16: fp32/fp16/bf16 only;
    every other dtype TORCH_CHECK-fails on the CUDA path
  - python/sgl_kernel/elementwise.py:248-255  _check_shape (ndim>=1, out
    prefix-compatible, gated input has even last dim)
  - python/sgl_kernel/elementwise.py:259-260 / 274 / 289 / 312  16-byte tail
    alignment: shape[-1] * itemsize % 16 == 0 else ValueError("The pointers
    must be multiple of 16 bytes.")
  - python/sgl_kernel/elementwise.py:262-266  out= is validated first,
    written in place, and returned (out identity)

Known deliberate divergences (documented, not re-litigated here):
  - out= shape mismatch: upstream raises RuntimeError (TORCH_CHECK via
    _check_shape), this shadow package raises ValueError — same rejection
    point, different exception type.
  - fp64 input: upstream CUDA path TORCH_CHECK-fails (utils.h:306-318);
    the torch emu tier here computes it. GPU triton tier keeps the
    fp32/fp16/bf16 dispatch (see _gated_mul_impl use_triton guard).
  - gelu_quick is a HIP-only export upstream (elementwise.py:303); here it
    is always exported (host triton/torch tiers are platform-agnostic).

Run (CPU face, any host):  pytest tests/kernels/test_aot_contract_activation.py
Run (GPU face):            same command on an HCU/CUDA node.
The GPU section is skipped automatically without a visible device; the
triton-vs-formula parity marks 待容器 (awaiting a DTK container) on hosts
without a device.
"""

import pytest
import torch

import sgl_kernel

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0),
    reason="requires HCU/CUDA device",
)

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

GATED_FNS = [
    ("silu", sgl_kernel.silu_and_mul),
    ("gelu", sgl_kernel.gelu_and_mul),
    ("gelu_tanh", sgl_kernel.gelu_tanh_and_mul),
]


# ----------------------------------------------------------- math constants --

def _silu_formula(x):  # activation.cu:56-58
    return x / (1.0 + torch.exp(-x))


def _gelu_erf_formula(x):  # activation.cu:60-63 (M_SQRT1_2 as float)
    return x * 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


def _gelu_tanh_formula(x):  # activation.cu:65-83, constants verbatim
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + torch.tanh(inner))


def _gelu_quick_formula(x):  # activation.cu (quick path) / elementwise.py:312
    return x / (1.0 + torch.exp(-1.702 * x))


_ACT_FORMULAS = {
    "silu": _silu_formula,
    "gelu": _gelu_erf_formula,
    "gelu_tanh": _gelu_tanh_formula,
}


# ------------------------------------------------ cpu face (any host, emu) --

@pytest.mark.parametrize("act,fn", GATED_FNS)
def test_gated_16byte_alignment_rejected(act, fn):
    # elementwise.py:259-260/274/289: last dim * itemsize must be a
    # multiple of 16 — fp32 tail 6 (=24B) and fp16 tail 12 (=24B) both fail.
    for dtype, last_dim in [(torch.float32, 6), (torch.float16, 12), (torch.bfloat16, 10)]:
        x = torch.randn(3, last_dim, dtype=dtype)
        with pytest.raises(ValueError, match="16 bytes"):
            fn(x)


def test_gelu_quick_16byte_alignment_rejected():
    # elementwise.py:312: same guard on the quick-gelu entry.
    x = torch.randn(3, 6, dtype=torch.float32)  # 24B tail
    with pytest.raises(ValueError, match="16 bytes"):
        sgl_kernel.gelu_quick(x)


@pytest.mark.parametrize("act,fn", GATED_FNS)
def test_gated_alignment_boundary_exact_16b_ok(act, fn):
    # Exactly 16 bytes (fp32 x4, fp16/bf16 x8) is the smallest legal tail.
    for dtype, last_dim in [(torch.float32, 4), (torch.float16, 8), (torch.bfloat16, 8)]:
        d = last_dim // 2
        x = torch.randn(3, last_dim, dtype=torch.float32).to(dtype)
        y = fn(x)
        gate = x[..., :d].float()
        ref = _ACT_FORMULAS[act](gate) * x[..., d:].float()
        torch.testing.assert_close(
            y.float(), ref.to(y.dtype).float(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("act,fn", GATED_FNS)
def test_gated_out_kwarg_identity_and_shape_guard(act, fn):
    # elementwise.py:262-266: out validated first, written in place,
    # returned. Shape guard diverges on exception type only (ValueError
    # here vs RuntimeError upstream) — see module docstring.
    x = torch.randn(3, 8, dtype=torch.float32)
    out = torch.empty(3, 4, dtype=torch.float32)
    y = fn(x, out)
    assert y is out
    gate, up = x[:, :4], x[:, 4:]
    torch.testing.assert_close(y, _ACT_FORMULAS[act](gate) * up, rtol=1e-5, atol=1e-5)

    bad = torch.empty(3, 5, dtype=torch.float32)
    with pytest.raises(ValueError, match="shape"):
        fn(x, bad)


def test_gelu_quick_out_kwarg_identity():
    x = torch.randn(3, 8, dtype=torch.float32)
    out = torch.empty(3, 8, dtype=torch.float32)
    y = sgl_kernel.gelu_quick(x, out)
    assert y is out
    torch.testing.assert_close(y, _gelu_quick_formula(x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("act,fn", GATED_FNS)
def test_gated_math_constants_cpu(dtype, act, fn):
    # activation.cu:56-83 constants, verified against the torch emu tier.
    torch.manual_seed(0)
    x = (torch.randn(5, 32, dtype=torch.float32) * 3).to(dtype)
    y = fn(x)
    gate, up = x[:, :16].float(), x[:, 16:].float()
    ref = (_ACT_FORMULAS[act](gate) * up).to(dtype).float()
    torch.testing.assert_close(y.float(), ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gelu_quick_math_constants_cpu(dtype):
    torch.manual_seed(1)
    x = (torch.randn(5, 32, dtype=torch.float32) * 3).to(dtype)
    y = sgl_kernel.gelu_quick(x)
    ref = _gelu_quick_formula(x.float()).to(dtype).float()
    torch.testing.assert_close(y.float(), ref, rtol=1e-3, atol=1e-3)


# ---------------------------------------------- gpu face (triton tier vs cu) --

@requires_gpu
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("act,fn", GATED_FNS)
def test_gated_triton_vs_activation_cu_constants(dtype, act, fn):
    # Triton tier vs the verbatim activation.cu:56-83 formulas (fp32 math,
    # single round at the store). NOT run on this dev host (no DTK): the
    # requires_gpu skip marks it 对拍待容器.
    torch.manual_seed(2)
    d = 256
    x = (torch.randn(33, d * 2, device="cuda") * 3).to(dtype)
    y = fn(x)
    gate, up = x[:, :d].float(), x[:, d:].float()
    ref = _ACT_FORMULAS[act](gate) * up
    tol = (2e-3, 2e-3) if dtype != torch.bfloat16 else (1.0, 0.3)
    torch.testing.assert_close(y.float(), ref, rtol=tol[0], atol=tol[1])


@requires_gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_gelu_quick_triton_vs_formula(dtype):
    torch.manual_seed(3)
    x = (torch.randn(33, 512, device="cuda") * 3).to(dtype)
    y = sgl_kernel.gelu_quick(x)
    ref = _gelu_quick_formula(x.float())
    tol = (2e-3, 2e-3) if dtype != torch.bfloat16 else (1.0, 0.3)
    torch.testing.assert_close(y.float(), ref, rtol=tol[0], atol=tol[1])


@requires_gpu
def test_gated_16byte_alignment_rejected_on_gpu():
    # Guard fires pre-launch on CUDA tensors too (upstream checks in the
    # python wrapper before any kernel dispatch).
    x = torch.randn(3, 6, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="16 bytes"):
        sgl_kernel.silu_and_mul(x)
