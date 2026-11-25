# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl

from hcu_adapter.kernels.diffusion_numerics_triton import (
    div_rn_f32,
    mul_rn_f32,
    rsqrt_approx_f32,
)


@pytest.fixture(autouse=True)
def require_gfx936():
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("Requires a HIP device")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx936":
        pytest.skip("The diffusion numerics bridge targets gfx936")


@triton.jit
def _apply(x_ptr, y_ptr, residual_ptr, out_ptr, n, mode: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    x = tl.load(x_ptr + offsets, offsets < n, 1)
    y = tl.load(y_ptr + offsets, offsets < n, 1)
    if mode == "mul_add":
        residual = tl.load(residual_ptr + offsets, offsets < n, 0)
        result = residual + mul_rn_f32(x, y)
    elif mode == "mul":
        result = mul_rn_f32(x, y)
    elif mode == "div":
        result = div_rn_f32(x, y)
    else:
        result = rsqrt_approx_f32(x)
    tl.store(out_ptr + offsets, result, offsets < n)


@pytest.mark.parametrize("mode", ["mul_add", "mul", "div", "rsqrt"])
def test_fp32_numerics_against_reference(mode):
    rng = torch.Generator().manual_seed(47)
    x = torch.cat((
        torch.tensor([
            1 + 2**-23, 2**-149, -(2**-149), 2**-126, 0.0, -0.0,
            float("inf"), -float("inf"), float("nan"), -2.0,
        ]),
        torch.randint(-(1 << 31), 1 << 31, (4096,), generator=rng, dtype=torch.int32)
        .view(torch.float32),
    ))
    y = torch.cat((
        torch.tensor([
            1 - 2**-23, 1.0, 1.0, 0.5, 0.0, -2.0,
            float("inf"), 1.0, 1.0, float("inf"),
        ]),
        torch.randint(-(1 << 31), 1 << 31, (4096,), generator=rng, dtype=torch.int32)
        .view(torch.float32),
    ))
    residual = torch.rand(x.shape, generator=rng)
    residual[0] = -1.0
    out = torch.empty_like(x, device="cuda")
    compiled = _apply[(triton.cdiv(x.numel(), 256),)](
        x.cuda(), y.cuda(), residual.cuda(), out, x.numel(), mode, 256
    )
    assembly = compiled.asm["amdgcn"]
    actual = out.cpu()

    if mode == "rsqrt":
        expected = torch.rsqrt(x.double()).float()
        torch.testing.assert_close(actual, expected, rtol=3e-7, atol=0, equal_nan=True)
        zeros = expected == 0
        assert torch.equal(actual[zeros].signbit(), expected[zeros].signbit())
        assert "v_rsq_f32" in assembly
        return
    if mode == "div":
        expected = x / y
        assert "v_div_scale_f32" in assembly and "v_div_fixup_f32" in assembly
    else:
        expected = x * y
        assert "v_mul_f32" in assembly
        if mode == "mul_add":
            expected = residual + expected
            assert "v_add_f32" in assembly
            assert "v_fma_f32" not in assembly and "v_fmac_f32" not in assembly
            assert actual[0].item() == 0.0
    finite_or_inf = ~expected.isnan()
    assert torch.equal(actual.isnan(), expected.isnan())
    assert torch.equal(actual[finite_or_inf].view(torch.int32),
                       expected[finite_or_inf].view(torch.int32))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_official_vae_residual_kernels(monkeypatch, dtype):
    pytest.importorskip("sglang.kernels.ops.diffusion.norm.h3_vae_fused_norm")
    from sglang.kernels.ops.diffusion.modulate import scale_shift_triton
    from sglang.kernels.ops.diffusion.norm import h3_vae_fused_norm

    monkeypatch.setattr(h3_vae_fused_norm, "mul_rn_f32", mul_rn_f32)
    monkeypatch.setattr(scale_shift_triton, "mul_rn_f32", mul_rn_f32)
    monkeypatch.setattr(scale_shift_triton, "is_cuda", lambda: True)
    gen = torch.Generator(device="cuda").manual_seed(47)
    residual = torch.randn((19, 2048), device="cuda", generator=gen)
    update = torch.randn(residual.shape, dtype=dtype, device="cuda", generator=gen)
    scale, weight = torch.randn((2, 2048), device="cuda", generator=gen)
    update[0, 0] = 1 + 2**-7
    scale[0] = 1 - 2**-20
    residual[0, 0] = -(1 + 2**-7 - 2**-20)
    expected = residual + update.float() * scale
    eps = 1e-6

    actual = scale_shift_triton.try_fused_scaled_residual_add_exact(residual, update, scale)
    assert actual is not None and torch.equal(actual, expected)
    stored, normed = h3_vae_fused_norm.h3_vae_scale_add_rmsnorm(
        residual, update, scale, weight, eps
    )
    assert stored is residual and torch.equal(stored, expected)
    assert stored[0, 0].item() == 0.0
    expected_norm = expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + eps) * weight
    torch.testing.assert_close(normed, expected_norm, rtol=3e-6, atol=2e-6)
