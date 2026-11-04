# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Standalone rope parity: sglang RotaryEmbedding (is_neox=False) vs ref math.

Uses the same config as GLM-5-W4A8: rope dim 64, interleave (neox off),
theta from checkpoint config.
"""
import json
import os
import sys

sys.path.insert(0, os.environ.get("HCU_ADAPTER_ROOT", "/adapter"))
import torch

CKPT = os.environ.get("HCU_CKPT", "/public/share/GLM-5-W4A8_V2_6").rstrip("/") + "/"
cfg = json.load(open(CKPT + "config.json"))
theta = cfg["rope_parameters"]["rope_theta"]
RD = cfg["qk_rope_head_dim"]
print(f"theta={theta} rope_dim={RD}")

dev = "cuda"
from sglang.srt.layers.rotary_embedding import RotaryEmbedding, get_rope

rope = get_rope(
    RD, rotary_dim=RD, max_position=64, base=float(theta),
    is_neox_style=False,
    dtype=torch.bfloat16,
)
rope = rope.to(dev)

T, NH = 5, 8
q = torch.randn(T, NH, RD, dtype=torch.bfloat16, device=dev)
k = torch.randn(T, 1, RD, dtype=torch.bfloat16, device=dev)
pos = torch.arange(T, device=dev)

import inspect

sig = inspect.signature(rope.forward)
print("rope forward sig:", list(sig.parameters))
try:
    q2, k2 = rope(pos, q.clone(), k.clone())
except TypeError:
    q2, k2 = rope(positions=pos, q=q.clone(), k=k.clone())

# reference interleave rope (same as _ref_forward)
posc = torch.arange(T).float()
inv = 1.0 / (theta ** (torch.arange(0, RD, 2).float() / RD))
ANGLE = posc[:, None] * inv[None, :]


def ref_rope(t):
    c, s = ANGLE.cos().to(dev), ANGLE.sin().to(dev)
    x1, x2 = t[..., 0::2].float(), t[..., 1::2].float()
    y1 = x1 * c[None, :, None] - x2 * s[None, :, None]
    y2 = x1 * s[None, :, None] + x2 * c[None, :, None]
    o = t.clone()
    o[..., 0::2] = y1.to(t.dtype)
    o[..., 1::2] = y2.to(t.dtype)
    return o


import torch.nn.functional as F

for name, got, want in [("q", q2, ref_rope(q)), ("k", k2, ref_rope(k))]:
    g = got.float().flatten()
    w = want.float().flatten()
    cos = F.cosine_similarity(g, w, dim=0).item()
    print(f"{name}: cos={cos:.6f} max|d|={(g - w).abs().max().item():.4f} "
          f"norm_g={g.norm():.3f} norm_w={w.norm():.3f}")


def ref_rope_neox(t):
    c, s = ANGLE.cos().to(dev), ANGLE.sin().to(dev)
    h = t.shape[-1] // 2
    x1, x2 = t[..., :h].float(), t[..., h:].float()
    y1 = x1 * c[None, :, None] - x2 * s[None, :, None]
    y2 = x1 * s[None, :, None] + x2 * c[None, :, None]
    return torch.cat([y1.to(t.dtype), y2.to(t.dtype)], -1)


print("-- vs NEOX-style reference (half-split) --")
for name, got, src in [("q", q2, q), ("k", k2, k)]:
    g = got.float().flatten()
    w = ref_rope_neox(src).float().flatten()
    cos = F.cosine_similarity(g, w, dim=0).item()
    print(f"{name}: cos_neox={cos:.6f}")
