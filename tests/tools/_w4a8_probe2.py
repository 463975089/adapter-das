# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Probe 2: does per-token-GROUP activation quant rescue w4a8 accuracy?

Simulates the quantized chain in pure torch (same math as the kernel):
  gemm1: int8(x) x int4(w)  (per-row scale vs per-rowxgroup scale)
  act:   silu(g)*u
  gemm2: same
Variants: scalar (current kernel) | group64 both gemms.
Reports rel error + cosine of full chain vs fp32 reference.
"""

import json
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F

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
    dt = {"I8": np.int8, "F32": np.float32, "BF16": np.uint16}[v["dtype"]]
    return torch.from_numpy(
        np.frombuffer(raw, dtype=dt).reshape(v["shape"]).copy()
    )


def stats(name, a, b):
    a, b = a.float().flatten(), b.float().flatten()
    rel = (a - b).abs() / (b.abs() + 1e-5)
    cos = F.cosine_similarity(a, b, dim=0).item()
    print(f"{name}: rel_mean={rel.mean():.5f} cos={cos:.6f}")


def quant_scalar(x):
    s = x.float().abs().amax(dim=-1, keepdim=True) / 127.0
    q = torch.round(x.float() / s.clamp(min=1e-10)).clamp(-127, 127)
    return q, s


def quant_group(x, group):
    M, K = x.shape
    xs = x.float().view(M, K // group, group)
    s = xs.abs().amax(dim=-1, keepdim=True) / 127.0
    q = torch.round(xs / s.clamp(min=1e-10)).clamp(-127, 127)
    return q.view(M, K), s.view(M, K // group)


def chain(x_f32, wg, wu, w2, mode):
    if mode == "scalar":
        gq, gs = quant_scalar(x_f32.to(torch.bfloat16))
        g = gq @ wg.t() * gs
        uq, us = quant_scalar(x_f32.to(torch.bfloat16))
        u = uq @ wu.t() * us
    else:
        g = group_gemm(x_f32, wg, 64)
        u = group_gemm(x_f32, wu, 64)
    act = F.silu(g) * u
    if mode == "scalar":
        aq, as_ = quant_scalar(act.to(torch.bfloat16))
        out = aq @ w2.t() * as_ * 0.7
    else:
        out = group_gemm(act, w2, 64) * 0.7
    return out


def group_gemm(x_f32, w, group):
    M, K = x_f32.shape
    xq, xs = quant_group(x_f32.to(torch.bfloat16), group)  # [M,K], [M,K/G]
    # dequant per group then fp32 gemm (mathematically == per-group int gemm)
    xd = (xq.view(M, K // group, group) * xs.unsqueeze(-1)).view(M, K)
    return xd @ w.t()


torch.manual_seed(0)
H, I = 6144, 2048

wg = _unpack_signed(read_shard(KEY_PCIE + "gate_proj.weight")).float() * (
    read_shard(KEY_PCIE + "gate_proj.weight_scale").float().view(-1) * 16.0
).view(-1, 1)
wu = _unpack_signed(read_shard(KEY_PCIE + "up_proj.weight")).float() * (
    read_shard(KEY_PCIE + "up_proj.weight_scale").float().view(-1) * 16.0
).view(-1, 1)
w2 = _unpack_signed(read_shard(KEY_PCIE + "down_proj.weight")).float() * (
    read_shard(KEY_PCIE + "down_proj.weight_scale").float().view(-1) * 16.0
).view(-1, 1)
wg, wu, w2 = wg.cuda(), wu.cuda(), w2.cuda()

tokens = 5
x = (torch.randn(tokens, H) * 0.5).to(torch.bfloat16).float().cuda()

# fp32 exact reference
act_ref = F.silu(x @ wg.t()) * (x @ wu.t())
out_ref = act_ref @ w2.t() * 0.7

# activation dynamic range diagnostic
amax = act_ref.abs().amax(dim=-1)
med = act_ref.abs().median()
print(f"act amax/median ratio per token: {(amax / med.clamp(min=1e-6)).tolist()}")

stats("scalar full", chain(x, wg, wu, w2, "scalar"), out_ref)
stats("group64 full", chain(x, wg, wu, w2, "group64"), out_ref)
