# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Standalone check: aiter fused_clamp_act_mul stub vs torch reference."""
import torch

import sgl_kernel  # registers the aiter meta-path stub

from aiter.ops.triton.fusions.fused_clamp_act_mul import fused_clamp_act_mul

torch.manual_seed(0)
dev = "cuda:0"
M, N = 33, 256  # N = 2 * 128, any M rows
lim = 7.0
gate_up = (torch.randn(M, N, device=dev, dtype=torch.float32) * 3).to(torch.bfloat16)
g, u = gate_up[:, : N // 2].float(), gate_up[:, N // 2 :].float()
ref = (torch.nn.functional.silu(g.clamp(max=lim)) * u.clamp(-lim, lim)).to(torch.bfloat16)

out = fused_clamp_act_mul(gate_up, swiglu_limit=lim, activation="silu")
torch.cuda.synchronize()
err = (out.float() - ref.float()).abs().max().item()
print(f"bf16 path: shape={tuple(out.shape)} maxerr={err:.4f}")
assert err < 0.1, err

from aiter import dtypes

q, s = fused_clamp_act_mul(
    gate_up, swiglu_limit=lim, activation="silu", dtype_quant=dtypes.fp8
)
torch.cuda.synchronize()
deq = q.float() * s
rel = (deq - ref.float()).abs().max()
print(
    f"fp8 path: q={tuple(q.shape)} {q.dtype} s={tuple(s.shape)} {s.dtype} "
    f"maxdelta={rel.item():.4f} deq-range=({deq.min().item():.2f},{deq.max().item():.2f})"
)
assert q.dtype == torch.float8_e4m3fn and s.dtype == torch.float32
refmax = ref.float().abs().max().item()
refmax = max(refmax, 1.0)
assert (rel.item() / refmax) < 0.08, (rel.item(), refmax)
print("PASS")
