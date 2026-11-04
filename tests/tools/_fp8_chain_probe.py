# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""FP8 chain stage probe: quant -> scaled_mm vs fp32 reference (HCU).

Mirrors Qwen3-4B-Channel-FP8 shapes (hidden 2560, inter 9728, heads 32);
each stage scored vs dequantized fp32 reference:
  A) per-token-group-256 activation quant (per_token_group_quant_fp8)
  B) static per-channel weight quant match (checkpoint weights already fp8)
  C) triton_scaled_mm with per-token a-scale from A + weight scale from B
  D) full apply_fp8_linear through sglang fp8_utils
"""

import sys

sys.path.insert(0, "/adapter")

import torch
import torch.nn.functional as F

torch.manual_seed(0)
M, K, N = 16, 2560, 9728

x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
w_bf16 = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
ws = (w_bf16.abs().amax(dim=1, keepdim=True) / 448.0).float().clamp(min=1e-12)
w_fp8 = (w_bf16.float() / ws).clamp(-448, 448).to(torch.float8_e4m3fn)

# fp32 reference
ref = x.float() @ (w_fp8.float() * ws).t()

from sglang.kernels.ops.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    scaled_fp8_quant,
)

# A) group quant (skip first dim==1 pitfall)
q, s, x_q_bg = None, None, None
try:
    xq, xs = per_token_group_quant_fp8(x, 256)
    print(f"A group-quant: xq{tuple(xq.shape)} xs{tuple(xs.shape)} "
          f"scale_med={xs.median().item():.3e}")
    dq = xq.float() * xs.repeat_interleave(256, 1)
    cos = F.cosine_similarity(dq.flatten(), x.float().flatten(), 0).item()
    print(f"A dequant cos={cos:.6f}")
except Exception as e:
    print(f"A FAIL: {e!r}")

# B) per-token dynamic quant + pad (our bridge path)
xq2, xs2 = scaled_fp8_quant(x, None, num_token_padding=17,
                            use_per_token_if_dynamic=True)
print(f"B padded-quant: out{tuple(xq2.shape)} scale{tuple(xs2.shape)} "
      f"scale[:4]={xs2.flatten()[:4].tolist()}")
dq2 = xq2[:M].float() * xs2
cos2 = F.cosine_similarity(dq2.flatten(), x.float().flatten(), 0).item()
print(f"B dequant cos={cos2:.6f}")

# C) scaled_mm: per-token a-scale x per-channel-w quant
try:
    out = torch._scaled_mm(
        xq2[:M], w_fp8.t(), scale_a=xs2, scale_b=ws.t().expand(-1, 1)[:1],
        out_dtype=torch.bfloat16,
    )
    print("C scaled_mm per-token/per-chan unexpected OK")
except Exception as e:
    print(f"C per-token x per-channel scaled_mm: {type(e).__name__}: {e!s:.140}")

# per-tensor fallback comparison
ws_t = w_bf16.abs().amax() / 448.0
w_t = (w_bf16.float() / ws_t).clamp(-448, 448).to(torch.float8_e4m3fn)
xq3, xs3 = scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)
try:
    out3 = torch._scaled_mm(
        xq3, w_t.t(), scale_a=xs3, scale_b=ws_t, out_dtype=torch.bfloat16)
    cos3 = F.cosine_similarity(
        out3.float().flatten(), ref.flatten(), 0).item()
    ratio3 = out3.float().norm().item() / (ref.norm().item() + 1e-6)
    print(f"C per-token x per-tensor: cos={cos3:.6f} norm_ratio={ratio3:.4f}")
except Exception as e:
    print(f"C per-tensor FAIL: {e!s:.140}")
