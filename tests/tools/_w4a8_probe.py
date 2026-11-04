# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Stage-by-stage error decomposition for fused_experts_w4a8.

A) gemm1 (packed int4 x int8-a) vs fp32 dequant ref: gate half / up half
B) activation (chunk silu(g)*u from kernel c1) vs fp32 ref act
C) gemm2 isolated: ref-act quantized same way vs fp32 ref down
D) full fused_experts_w4a8 vs full fp32 ref
Cosine + rel per stage; also sanity-check per_token_quant_int8 scale formula.
"""

import os
import sys

sys.path.insert(0, "/adapter")

import struct
import json

import torch
import torch.nn.functional as F

from hcu_adapter.slimquant import (
    _invoke_w4a8,
    _per_token_group_quant_int8,
    fused_experts_w4a8,
)

import triton

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
    return torch.from_numpy(arr.copy())


def load_projs(prefix, rows):
    w = read_shard(KEY_PCIE + prefix + ".weight")
    s = read_shard(KEY_PCIE + prefix + ".weight_scale").float().view(-1) * 16.0
    return w.cuda(), s.cuda()


def stats(name, a, b):
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    rel = diff / (b.abs() + 1e-5)
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    # cos of each row of b vs a (batch)
    print(
        f"{name}: rel_mean={rel.mean().item():.5f} rel_max={rel.max().item():.4f} "
        f"cos={cos:.6f} scale_ratio={(a.norm() / (b.norm() + 1e-9)).item():.6f}"
    )
    return rel.mean().item()


torch.manual_seed(0)
E, H, I = 1, 6144, 2048

wg, sg = load_projs("gate_proj", I)
wu, su = load_projs("up_proj", I)
w2, s2 = load_projs("down_proj", H)

w13 = torch.cat([wg, wu], dim=0).unsqueeze(0).contiguous()  # [1, 2I, H//2]
w13_scale = torch.cat([sg, su]).view(E, 2 * I, 1).contiguous()
w2b = w2.unsqueeze(0).contiguous()
w2_scale = s2.view(E, H, 1).contiguous()

tokens = 5
x = (torch.randn(tokens, H) * 0.5).to(torch.bfloat16).cuda()
topk_w = torch.full((tokens, 1), 0.7, dtype=torch.float32, device="cuda")
topk_ids = torch.zeros((tokens, 1), dtype=torch.int64, device="cuda")

# fp32 reference chain
g_ref = x.float() @ (_unpack_signed(wg).cuda().float() * sg.view(-1, 1)).t()
u_ref = x.float() @ (_unpack_signed(wu).cuda().float() * su.view(-1, 1)).t()
act_ref = F.silu(g_ref) * u_ref
w2_ref = (_unpack_signed(w2).cuda().float() * s2.view(-1, 1))
out_ref = (act_ref @ w2_ref.t()) * 0.7

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)

config = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}
compute_type = triton.language.bfloat16

# --- A) gemm1 standalone, full 2I ---
s_tid, e_ids, n_post = moe_align_block_size(topk_ids, 16, E)
a1_q, a1_s = _per_token_group_quant_int8(x)
cache1 = torch.empty((tokens, 1, 2 * I), device="cuda", dtype=torch.bfloat16)
_invoke_w4a8(
    a1_q, w13, cache1, a1_s, w13_scale, topk_w,
    s_tid, e_ids, n_post, top_k=1, config=config,
    mul_routed_weight=False, compute_type=compute_type,
)
c1 = cache1.view(tokens, 2 * I).float()
print("-- A: gemm1 halves --")
stats("  gate half", c1[:, :I], g_ref)
stats("  up   half", c1[:, I:], u_ref)

# --- B) activation from kernel c1 vs ref ---
act_k = F.silu(c1[:, :I]) * c1[:, I:]
print("-- B: activation --")
stats("  act", act_k, act_ref)
# --- C) gemm2 isolated with ref-act quantized identically ---
a2_q, a2_s = _per_token_group_quant_int8(act_k.to(torch.bfloat16))
cache2 = torch.empty((tokens, 1, H), device="cuda", dtype=torch.bfloat16)
_invoke_w4a8(
    a2_q, w2b, cache2, a2_s, w2_scale, topk_w,
    s_tid, e_ids, n_post, top_k=1, config=config,
    mul_routed_weight=True, compute_type=compute_type,
)
out_c = cache2.view(tokens, H).float()
print("-- C: gemm2 (routed x0.7) --")
stats("  down", out_c, out_ref)

# --- D) full path ---
out_d = fused_experts_w4a8(
    x, w13, w2b, w13_scale, w2_scale, topk_w, topk_ids, routed_scaling_factor=1.0
)
print("-- D: full fused_experts_w4a8 --")
stats("  full", out_d, out_ref)
