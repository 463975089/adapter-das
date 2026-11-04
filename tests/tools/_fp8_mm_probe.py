# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Probe: what does sglang.kernels selector see as platform on HCU?"""
import torch

print("hip:", torch.version.hip)

from sglang.kernels.spec import PlatformInfo, CapabilityRequirement

p = PlatformInfo.detect()
print("device_type:", p.device_type, "| arch:", p.cuda_arch_major)

from sglang.kernels.ops.gemm import _FP8_SCALED_MM

try:
    ks = _FP8_SCALED_MM.available_backends()
    print("registered backends:", ks)
except Exception as e:
    print("available_backends err:", repr(e)[:120])

from sglang.kernels.selector import select_kernel

try:
    spec = select_kernel("gemm.fp8_scaled_mm")
    print("selected:", spec.backend, spec.target)
except Exception as e:
    print("select err:", type(e).__name__, str(e)[:200])

from sglang.kernels.ops.gemm import fp8_scaled_mm

try:
    out = fp8_scaled_mm(
        torch.zeros(4, 8, device="cuda").to(torch.float8_e4m3fn),
        torch.zeros(8, 6, device="cuda").to(torch.float8_e4m3fn),
        torch.ones(4, device="cuda"),
        torch.ones(6, device="cuda"),
        torch.bfloat16,
    )
    print("call OK:", out.shape)
except Exception as e:
    print("call err:", type(e).__name__, str(e)[:200])
