# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import sys
import traceback

sys.path.insert(0, "/adapter")

import torch

from sglang.kernels.ops.quantization import fp8_kernel as fpk

# trace the native entry to see actual shapes
_orig_native = fpk._native_dynamic_per_token_quant_fp8


def traced(output, input, scale):
    print(f"NATIVE output={tuple(output.shape)} input={tuple(input.shape)} "
          f"scale={tuple(scale.shape)}", flush=True)
    return _orig_native(output, input, scale)


fpk._native_dynamic_per_token_quant_fp8 = traced

x = torch.randn(16, 64, device="cuda").to(torch.bfloat16)
fpk._use_aiter = False  # force the native arm the bridge guards
try:
    out, s = fpk.scaled_fp8_quant(
        x, None, num_token_padding=17, use_per_token_if_dynamic=True)
    print("dynamic-pad OK", tuple(out.shape), tuple(s.shape))
    # upstream contract: num_token_padding pads BOTH output and per-token
    # scale to the same row count (neutral 1.0 tail rows on the bridge)
    assert out.shape == (17, 64), out.shape
    assert s.shape == (17, 1), s.shape
    assert s[16, 0].item() == 1.0, s[16, 0].item()
    assert torch.isfinite(s).all()
    print("dynamic-pad CONTRACT OK: out/scale both 17 rows, scale tail=1.0")
except Exception:
    traceback.print_exc()

try:
    sc = torch.ones(16, 1, device="cuda", dtype=torch.float32)
    out, s = fpk.scaled_fp8_quant(
        x, sc, num_token_padding=17, use_per_token_if_dynamic=True)
    print("static-scale-pad OK", tuple(out.shape), tuple(s.shape))
except Exception:
    traceback.print_exc()

# per-tensor dynamic scale must stay 1-D and unpadded
try:
    out, s = fpk.scaled_fp8_quant(
        x, None, num_token_padding=17, use_per_token_if_dynamic=False)
    print("per-tensor-pad OK", tuple(out.shape), tuple(s.shape), "numel", s.numel())
    assert out.shape == (17, 64), out.shape
    assert s.numel() == 1, s.shape
    print("per-tensor CONTRACT OK: scale stays scalar")
except Exception:
    traceback.print_exc()

