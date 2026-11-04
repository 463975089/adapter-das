# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""rmsnorm bridge sanity: sgl_kernel.rmsnorm + our fused_add bridge vs torch."""
import os
import sys

sys.path.insert(0, os.environ.get("HCU_ADAPTER_ROOT", "/adapter"))
import torch
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda"
H, eps = 6144, 1e-6

x = (torch.randn(5, H) * 0.5).to(torch.bfloat16).to(dev)
res = (torch.randn(5, H) * 0.2).to(torch.bfloat16).to(dev)
w = (torch.randn(H) * 0.1 + 1.0).to(torch.bfloat16).to(dev)

ref = F.rms_norm((x.float() + res.float()), (H,), w.float(), eps)

from sgl_kernel import rmsnorm as sk_rmsnorm

out = torch.empty_like(x)
sk_rmsnorm(x.clone(), w, eps, out=out)
ref_plain = F.rms_norm(x.float(), (H,), w.float(), eps)
cos = F.cosine_similarity(out.float().flatten(), ref_plain.flatten(), dim=0)
print(f"plain rmsnorm: cos={cos.item():.6f} "
      f"norm_k={out.float().norm():.3f} norm_ref={ref_plain.norm():.3f}")

# our sitecustomize bridge form (6-arg HIP)
def fused_add_rms_norm(out, input, residual_out, residual, weight, epsilon):
    residual_in = residual if residual_out is residual else residual_out
    residual_in.copy_(residual)
    residual_in.add_(input)
    sk_rmsnorm(residual_in, weight, epsilon, out=out)

out2 = torch.empty_like(x)
res_out = torch.empty_like(x)
fused_add_rms_norm(out2, x, res_out, res, w, eps)
cos2 = F.cosine_similarity(out2.float().flatten(), ref.flatten(), dim=0)
res_ref = x.float() + res.float()
cosr = F.cosine_similarity(res_out.float().flatten(), res_ref.flatten(), dim=0)
print(f"fused_add out: cos={cos2.item():.6f} norm={out2.float().norm():.3f} "
      f"ref={ref.norm():.3f}")
print(f"residual_out : cos={cosr.item():.6f} norm={res_out.float().norm():.3f} "
      f"ref={res_ref.norm():.3f}")

# also check what vllm._custom_ops has on this image after sitecustomize load
print("--- vllm._custom_ops state after import sgl_kernel ---")
import sgl_kernel  # noqa: F401  (loads adapter stack like server does)
import vllm._custom_ops as vco
import inspect
print("rms_norm:", vco.rms_norm)
print("fused_add_rms_norm:", vco.fused_add_rms_norm)
print("has torch.ops._C.rms_norm:", hasattr(torch.ops._C, "rms_norm"))
