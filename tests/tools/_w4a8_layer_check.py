# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Single-layer w4a8 MoE parity on HCU: packed-int4 kernel vs bf16 dequant ref.

Loads one expert's real checkpoint tensors, runs:
  A) fused_experts_w4a8 (adapter triton kernel, packed int4 in HBM)
  B) bf16 reference (unpack -> dequant -> torch F.silu(mm))
Reports relative error of the outputs.
"""

import os
import sys

sys.path.insert(0, "/adapter")

import torch

from hcu_adapter.slimquant import fused_experts_w4a8

import json
import struct

CKPT = "/public/share/GLM-5-W4A8_V2_6/"
KEY_PCIE = "model.layers.7.mlp.experts.0."
wm = json.load(open(CKPT + "model.safetensors.index.json"))["weight_map"]


def _unpack_signed(w):
    u = w.view(torch.uint8)
    hi = (u >> 4).to(torch.int16)
    lo = (u & 0xF).to(torch.int16)
    hi = torch.where(hi >= 8, hi - 16, hi)
    lo = torch.where(lo >= 8, lo - 16, lo)
    return torch.stack((hi, lo), -1).reshape(w.shape[0], w.shape[1] * 2).to(torch.int8)


def read_shard(key):
    fp = open(CKPT + wm[key], "rb")
    n = struct.unpack("<Q", fp.read(8))[0]
    hdr = json.loads(fp.read(n))
    v = hdr[key]
    s, e = v["data_offsets"]
    fp.seek(8 + n + s)
    raw = fp.read(e - s)
    fp.close()
    import numpy as np

    dt = {"I8": np.int8, "F32": np.float32, "BF16": np.uint16}[v["dtype"]]
    arr = np.frombuffer(raw, dtype=dt).reshape(v["shape"])
    t = torch.from_numpy(arr.copy())
    return t


def load_projs(prefix, rows):
    w = read_shard(KEY_PCIE + prefix + ".weight")  # I8 [N, K//2] packed
    s = read_shard(KEY_PCIE + prefix + ".weight_scale").float().view(-1) * 16.0
    assert w.shape[0] == rows, (w.shape, rows)
    return w.cuda(), s.cuda()


torch.manual_seed(0)
E, H, I = 1, 6144, 2048

w13_gate, s_gate = load_projs("gate_proj", I)
w13_up, s_up = load_projs("up_proj", I)
w2, s2 = load_projs("down_proj", H)

# fuse gate/up into w13 layout [E, 2I, K//2]
w13 = (
    torch.cat([w13_gate, w13_up], dim=0).unsqueeze(0).contiguous()
)  # [1, 2I, K//2]
w13_scale = torch.cat([s_gate, s_up]).view(E, 2 * I, 1).contiguous()
w2 = w2.unsqueeze(0).contiguous()
w2_scale = s2.view(E, H, 1).contiguous()

tokens = 5
x = (torch.randn(tokens, H) * 0.5).to(torch.bfloat16).cuda()
topk_w = torch.full((tokens, 1), 0.7, dtype=torch.float32, device="cuda")
topk_ids = torch.zeros((tokens, 1), dtype=torch.int64, device="cuda")

out_w4a8 = fused_experts_w4a8(
    x, w13, w2, w13_scale, w2_scale, topk_w, topk_ids, routed_scaling_factor=1.0
)

# bf16 reference
gate_u = _unpack_signed(w13_gate).cuda().float() * s_gate.view(-1, 1)
up_u = _unpack_signed(w13_up).cuda().float() * s_up.view(-1, 1)
w2_u = _unpack_signed(w2[0]).cuda().float() * s2.view(-1, 1)

out_ref = torch.empty_like(x, dtype=torch.float32)
for i in range(tokens):
    g = (x[i].float() @ gate_u.t()).float()
    u = (x[i].float() @ up_u.t()).float()
    act = torch.nn.functional.silu(g) * u
    out_ref[i] = act @ w2_u.t() * 0.7

out_ref = out_ref.to(torch.bfloat16).float()
out_w4a8f = out_w4a8.float()
rel = (out_w4a8f - out_ref).abs() / (out_ref.abs() + 1e-5)
print("rel err: mean=%.4f max=%.4f" % (rel.mean().item(), rel.max().item()))
print("sample emu:", out_w4a8f[0, :4].tolist())
print("sample ref:", out_ref[0, :4].tolist())
row_rel = rel.mean(dim=1)
print("per-token mean rel:", [round(v, 4) for v in row_rel.tolist()])
n_norm = (out_ref[0].abs() > 1e-4).sum().item()
print("ref row0 nonzero frac:", n_norm / out_ref.shape[1])
