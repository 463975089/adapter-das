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
try:
    out, s = fpk.scaled_fp8_quant(
        x, None, num_token_padding=17, use_per_token_if_dynamic=True)
    print("dynamic-pad OK", tuple(out.shape), tuple(s.shape))
except Exception:
    traceback.print_exc()

try:
    sc = torch.ones(16, 1, device="cuda", dtype=torch.float32)
    out, s = fpk.scaled_fp8_quant(
        x, sc, num_token_padding=17, use_per_token_if_dynamic=True)
    print("static-scale-pad OK", tuple(out.shape), tuple(s.shape))
except Exception:
    traceback.print_exc()

