# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/hcu_t2")

from safetensors import safe_open

from sgl_kernel import int8_scaled_mm
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8

MODEL = "/public/opendas/DL_DATA/llm-models/Qwen3.6-27B-Channel-INT8-w8a8"
cfg = json.load(open(os.path.join(MODEL, "config.json")))
print("quant_config:", json.dumps(cfg.get("quantization_config"), indent=1)[:600])

idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))
wm = idx["weight_map"]
wkey = next(k for k in wm if k.endswith("mlp.down_proj.weight"))
with safe_open(os.path.join(MODEL, wm[wkey]), framework="pt") as f:
    w = f.get_tensor(wkey).cuda()  # [N,K] int8
    ws = f.get_tensor(wkey.replace(".weight", ".weight_scale")).cuda()

print("w", w.shape, w.dtype, "| ws", ws.shape, ws.dtype)

# emulate CompressedTensorsW8A8Int8.CHANNEL.process_weights_after_loading
layer_weight = torch.nn.Parameter(w.t(), requires_grad=False)  # [K,N] view
layer_scale = torch.nn.Parameter(ws.float(), requires_grad=False)  # [N,1]

torch.manual_seed(0)
M, K = 64, w.shape[1]
x = (torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.05).to(torch.bfloat16)

x_q, x_scale = per_token_quant_int8(x)
print("x_q", x_q.shape, x_q.dtype, "| x_scale", x_scale.shape, x_scale.dtype)

out = int8_scaled_mm(
    x_q, layer_weight, x_scale, layer_scale, out_dtype=torch.bfloat16, bias=None
)
print("out", out.shape, out.dtype, "non-contig weight refcount ok")

# non-contiguous vs contiguous control
out_c = int8_scaled_mm(
    x_q, layer_weight.contiguous(), x_scale, layer_scale,
    out_dtype=torch.bfloat16, bias=None,
)
print("view-vs-contig maxdiff:", (out.float() - out_c.float()).abs().max().item())

w_deq = w.float() * ws.float()  # [N,K]
ref = x.float() @ w_deq.t()  # [M,N]
cos = F.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
row = F.cosine_similarity(out.float(), ref, dim=1)
print(f"flat cos={cos:.6f} row-cos min={row.min().item():.6f} mean={row.mean().item():.6f}")

rel = (out.float() - ref).abs() / (ref.abs() + 1e-3)
print(f"absmax err={(out.float()-ref).abs().max().item():.4f} rel(eps-clamped) mean={rel.mean().item():.4f}")
