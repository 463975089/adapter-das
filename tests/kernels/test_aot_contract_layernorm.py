# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""AOT contract anchors: RMSNorm family (rmsnorm / fused_add_rmsnorm / gemma).

Contract sources (upstream sgl-kernel v0.5.22 tag):
  - vendored flashinfer norm.cuh (FetchContent pin bc29697, CMakeLists.txt:75-81):
      * rmsnorm: fp32 accumulate, rstd = rsqrtf(mean(x^2) + eps)  — eps
        joins AFTER the mean, not per-element
      * gemma variants: weight_bias = 1.f  (y = x*rstd*(w+1))
      * fused_add ordering: residual += input FIRST, then
        input = rmsnorm(residual) * w  (residual is the new sum)
      * out= aliasing input is safe (two-pass kernel: sum pass, then
        reload+store pass)
      * norm.cuh:219,223: upstream normalizes the UNROUNDED fp32 sum
  - python/sgl_kernel/layernorm.py: rmsnorm(input, weight, eps=1e-6,
    out=None, enable_pdl=None); enable_pdl feeds PDL on NVIDIA builds only.

Known deliberate divergence (kept since ee305e5, do not silently "fix"):
  N1 — the community fused_add numerator uses the ACTIVATION-DTYPE-STORED
  residual (t rounded to bf16/fp16, reloaded) while its variance uses the
  fp32 sum; upstream norm.cuh:219,223 uses the fp32 sum for both. The
  anchor below asserts BOTH references agree within one activation-dtype
  ULP budget, pinning the divergence as bounded, not absent.

Run (CPU face, any host):  pytest tests/kernels/test_aot_contract_layernorm.py
Run (GPU face):            same command on an HCU/CUDA node (对拍待容器
without a device; the requires_gpu markers auto-skip).
"""

import inspect

import pytest
import torch

import sgl_kernel

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0),
    reason="requires HCU/CUDA device",
)

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


# ------------------------------------------------------------- references --

def _rms_ref(x, weight, eps, gemma):
    """norm.cuh: x * rsqrtf(mean(x^2) + eps) * (w + gemma?1:0), fp32 math."""
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf * rstd * (weight.float() + (1.0 if gemma else 0.0))


def _fused_add_ref_upstream(input_, residual, weight, eps, gemma):
    """Upstream semantics (norm.cuh:219,223): numerator uses the fp32 sum."""
    total = input_.float() + residual.float()
    new_res = total.to(residual.dtype)
    rstd = torch.rsqrt(total.pow(2).mean(-1, keepdim=True) + eps)
    new_x = (total * rstd * (weight.float() + (1.0 if gemma else 0.0))).to(input_.dtype)
    return new_x, new_res


def _fused_add_ref_stored(input_, residual, weight, eps, gemma):
    """Community semantics (ee305e5, this package): numerator reloads the
    activation-dtype-stored residual."""
    total = input_.float() + residual.float()
    new_res = total.to(residual.dtype)
    stored = new_res.float()
    rstd = torch.rsqrt(total.pow(2).mean(-1, keepdim=True) + eps)
    new_x = (stored * rstd * (weight.float() + (1.0 if gemma else 0.0))).to(input_.dtype)
    return new_x, new_res


# ------------------------------------------------ cpu face (any host, emu) --

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("gemma", [False, True])
def test_rmsnorm_eps_after_mean_cpu(dtype, gemma):
    # eps joins after the mean: with x = c * ones, RMS = |c| exactly, so
    # out = c / sqrt(c^2 + eps) — distinguishing it from mean(x^2 + eps).
    fn = sgl_kernel.gemma_rmsnorm if gemma else sgl_kernel.rmsnorm
    torch.manual_seed(0)
    x = (torch.randn(4, 64) * 2).to(dtype)
    w = (torch.randn(64) * 0.5).to(dtype)
    eps = 1e-6
    y = fn(x, w, eps)
    ref = _rms_ref(x, w, eps, gemma).to(dtype)
    torch.testing.assert_close(y.float(), ref.float(), rtol=1e-3, atol=1e-3)

    # eps placement, made visible: tiny rows flip the denominator.
    c = torch.full((1, 64), 1e-4, dtype=torch.float32)
    w1 = torch.ones(64, dtype=torch.float32)
    y_small = sgl_kernel.rmsnorm(c, w1, eps=0.5)
    expect = c / torch.sqrt(c.pow(2).mean(-1, keepdim=True) + 0.5)
    torch.testing.assert_close(y_small, expect, rtol=1e-6, atol=1e-8)


def test_gemma_weight_plus_one_cpu():
    # norm.cuh gemma: weight_bias = 1.f — zero weights must not zero the row.
    x = torch.randn(3, 128, dtype=torch.float32)
    w = torch.zeros(128, dtype=torch.float32)
    y = sgl_kernel.gemma_rmsnorm(x, w)
    ref = _rms_ref(x, w, 1e-6, gemma=True)
    torch.testing.assert_close(y, ref, rtol=1e-6, atol=1e-6)
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(y, x * rstd, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_add_ordering_cpu(dtype):
    # residual += input FIRST; input = rmsnorm(residual) * w SECOND —
    # after the call, residual is exactly the (rounded) sum.
    torch.manual_seed(1)
    x = (torch.randn(4, 64) * 2).to(dtype)
    r = (torch.randn(4, 64)).to(dtype)
    w = (torch.randn(64) * 0.5).to(dtype)
    x0, r0 = x.clone(), r.clone()
    ret = sgl_kernel.fused_add_rmsnorm(x, r, w)
    assert ret is None  # in-place contract, no return value
    total = x0.float() + r0.float()
    torch.testing.assert_close(r.float(), total.to(dtype).float(), rtol=1e-3, atol=1e-3)


def test_fused_add_n1_divergence_is_bounded_cpu():
    # The N1 divergence anchor: our stored-residual numerator must sit
    # within one activation-dtype ULP of the upstream fp32 numerator.
    torch.manual_seed(2)
    dtype = torch.bfloat16
    x = (torch.randn(6, 128) * 3).to(dtype)
    r = (torch.randn(6, 128) * 3).to(dtype)
    w = (torch.randn(128) * 0.5).to(dtype)
    x_ours, r_ours = x.clone(), r.clone()
    sgl_kernel.fused_add_rmsnorm(x_ours, r_ours, w)
    x_up, r_up = _fused_add_ref_upstream(x, r, w, 1e-6, False)
    x_st, r_st = _fused_add_ref_stored(x, r, w, 1e-6, False)
    torch.testing.assert_close(r_ours.float(), r_up.float(), rtol=0, atol=0)
    torch.testing.assert_close(r_ours.float(), r_st.float(), rtol=0, atol=0)
    # both references describe our output within a bf16-ULP budget
    torch.testing.assert_close(x_ours.float(), x_st.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(x_ours.float(), x_up.float(), rtol=2e-2, atol=2e-2)


def test_rmsnorm_out_kwarg_identity_and_aliasing_cpu():
    # out= is validated, written in place, returned; aliasing out IS input
    # is safe (two-pass structure: variance pass never re-reads Y).
    x = torch.randn(3, 64, dtype=torch.float32)
    w = torch.randn(64, dtype=torch.float32)
    out = torch.empty_like(x)
    y = sgl_kernel.rmsnorm(x, w, out=out)
    assert y is out
    torch.testing.assert_close(y, _rms_ref(x, w, 1e-6, False), rtol=1e-6, atol=1e-6)

    bad = torch.empty(3, 32, dtype=torch.float32)
    with pytest.raises(ValueError, match="shape"):
        sgl_kernel.rmsnorm(x, w, out=bad)

    x_orig = x.clone()
    aliased = sgl_kernel.rmsnorm(x, w, out=x)  # x is the out buffer now
    fresh = sgl_kernel.rmsnorm(x_orig, w)
    torch.testing.assert_close(aliased, fresh, rtol=1e-6, atol=1e-6)


def test_rmsnorm_enable_pdl_accepted_cpu():
    # enable_pdl exists for API parity (NVIDIA PDL); accepted and ignored.
    sig = inspect.signature(sgl_kernel.rmsnorm)
    assert "enable_pdl" in sig.parameters
    assert "enable_pdl" in inspect.signature(sgl_kernel.fused_add_rmsnorm).parameters
    x = torch.randn(2, 64, dtype=torch.float32)
    w = torch.randn(64, dtype=torch.float32)
    sgl_kernel.rmsnorm(x, w, enable_pdl=True)      # kwarg form
    sgl_kernel.rmsnorm(x, w, 1e-6, None, False)     # positional form


# ------------------------------------------------ gpu face (triton tier) --

@requires_gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_triton_vs_norm_cuh(dtype):
    torch.manual_seed(3)
    x = (torch.randn(17, 512, device="cuda") * 2).to(dtype)
    w = (torch.randn(512, device="cuda") * 0.5).to(dtype)
    y = sgl_kernel.rmsnorm(x, w)
    ref = _rms_ref(x, w, 1e-6, False).to(dtype)
    torch.testing.assert_close(y.float(), ref.float(), rtol=2e-3, atol=2e-3)


@requires_gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_add_triton_vs_refs(dtype):
    # Triton tier vs BOTH numerators (upstream fp32 + stored-residual):
    # each within one ULP budget — the N1 divergence stays bounded on GPU.
    torch.manual_seed(4)
    x = (torch.randn(17, 512, device="cuda") * 2).to(dtype)
    r = (torch.randn(17, 512, device="cuda") * 2).to(dtype)
    w = (torch.randn(512, device="cuda") * 0.5).to(dtype)
    x_t, r_t = x.clone(), r.clone()
    sgl_kernel.fused_add_rmsnorm(x_t, r_t, w)
    x_up, r_up = _fused_add_ref_upstream(x, r, w, 1e-6, False)
    x_st, r_st = _fused_add_ref_stored(x, r, w, 1e-6, False)
    # the residual output IS part of the contract: both references store
    # the identical activation-dtype sum — assert it bit-exact (review
    # #80: it was previously discarded and never checked).
    torch.testing.assert_close(r_t.float(), r_up.float(), rtol=0, atol=0)
    torch.testing.assert_close(r_t.float(), r_st.float(), rtol=0, atol=0)
    # both numerators describe the normalized output within a ULP budget
    torch.testing.assert_close(x_t.float(), x_up.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(x_t.float(), x_st.float(), rtol=2e-2, atol=2e-2)


@requires_gpu
def test_rmsnorm_out_aliases_input_triton():
    # Same-address out= on the triton tier. NOTE: out=x rewrites x in
    # place, so the reference must snapshot the ORIGINAL x first.
    torch.manual_seed(5)
    x = torch.randn(9, 512, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    x0 = x.clone()
    y = sgl_kernel.rmsnorm(x, w, out=x)
    # Pin the same-address CONTRACT itself, not just the values: out=x
    # must write in place and hand back the caller's buffer — an impl
    # that ignores out= and returns a fresh tensor would pass the value
    # check below while breaking every caller relying on x being
    # updated (the CPU face asserts `y is out`; review #80).
    assert y is x
    fresh = sgl_kernel.rmsnorm(x0, w)
    torch.testing.assert_close(y.float(), fresh.float(), rtol=2e-2, atol=2e-2)


@requires_gpu
@pytest.mark.parametrize("gemma", [False, True])
def test_gemma_triton_weight_bias(gemma):
    torch.manual_seed(6)
    fn = sgl_kernel.gemma_rmsnorm if gemma else sgl_kernel.rmsnorm
    x = torch.randn(9, 512, device="cuda", dtype=torch.bfloat16)
    # gemma=True keeps zero weights on purpose: (w+1)==1 pins the +1 bias
    # semantics (same point as test_gemma_weight_plus_one_cpu). gemma=False
    # must NOT use zeros: y = (x*rstd)*w would be identically zero on both
    # sides and assert_close(0, 0) would pass no matter what the kernel did
    # to the normalization — use randn so the output tracks it (review #80
    # R7 P3).
    if gemma:
        w = torch.zeros(512, device="cuda", dtype=torch.bfloat16)
    else:
        w = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    y = fn(x, w)
    ref = _rms_ref(x.float(), w.float(), 1e-6, gemma)
    torch.testing.assert_close(y.float(), ref, rtol=2e-2, atol=2e-2)
