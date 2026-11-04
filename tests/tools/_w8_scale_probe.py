# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""W8 linear-layer scale semantics probe (GLM-5-W4A8_V2_6).

Q1 (internal): dequant'd W8 weights must be bounded like a quantized tensor —
   if scales are stored divided by 16 and we FORGOT to multiply, weight rows
   become up to ~16x larger than any sane quantization error; compare row
   amax against int8 grid expectation + reconstruction residual.
Q2 (cross-check): bf16-CAST of GLM-5.2-Channel-FP8-w8a8's per-tensor fp8 +
   activation-scale weights should correlate ~1.0 row-wise with our dequant.

Runs on GPU 0 only, reads shards 1..2, prints compact verdicts.
"""

import json
import struct

import numpy as np
import torch

CKPT = "/public/share/GLM-5-W4A8_V2_6/"
wm = json.load(open(CKPT + "model.safetensors.index.json"))["weight_map"]


def read_shard(path, key):
    fp = open(path, "rb")
    n = struct.unpack("<Q", fp.read(8))[0]
    hdr = json.loads(fp.read(n))
    v = hdr[key]
    s, e = v["data_offsets"]
    fp.seek(8 + n + s)
    raw = fp.read(e - s)
    fp.close()
    dt = {"I8": np.int8, "F32": np.float32, "BF16": np.uint16, "F8_E4M3": np.uint8}[
        v["dtype"]
    ]
    return torch.from_numpy(np.frombuffer(raw, dtype=dt).reshape(v["shape"]).copy())


def get(key):
    return read_shard(CKPT + wm[key], key)


def bf16(t):
    return t.view(torch.bfloat16) if t.dtype == torch.uint16 else t


def probe_row(name, w8, scale, ref=None):
    s = scale.float().flatten()
    for mult, tag in ((16.0, "x16"), (1.0, "raw")):
        w = w8.float() * (s * mult).view(-1, 1)
        row_amax = w.abs().amax(dim=1)
        # a correct per-row int8 quant reconstructs to ~relative 1/127 steps;
        # check worst |w| row against the scale*127 bound only for x16=true scale
        print(f"{name} [{tag}]: scale[min..max]=({s.min():.6f},{s.max():.6f}) "
              f"row_amax/|scale| median={((row_amax / (s.abs()+1e-12)).median()).item():.1f}")
    if ref is not None:
        ours = (w8.float() * (s * 16.0).view(-1, 1)).to(torch.bfloat16).float()
        cos = torch.nn.functional.cosine_similarity(ours, ref.float(), dim=1)
        ratio = ours.norm(dim=1) / ref.float().norm(dim=1).clamp(min=1e-9)
        print(f"{name} vs FP8-ref: cos median={cos.median():.6f} "
              f"norm_ratio median={ratio.median():.4f}")


torch.manual_seed(0)
for L in (0, 7):
    q = f"model.layers.{L}.self_attn.q_b_proj"
    kvb = f"model.layers.{L}.self_attn.kv_b_proj"
    o = f"model.layers.{L}.self_attn.o_proj"
    probe_row(q, get(q + ".weight"), get(q + ".weight_scale"))
    probe_row(kvb, get(kvb + ".weight"), get(kvb + ".weight_scale"))

# embed/lm_head sanity (bf16, untouched)
emb = bf16(get("model.embed_tokens.weight")).float()
print("embed_tokens: std=%.5f  amax=%.3f" % (emb.std(), emb.abs().max()))

# Q2 done separately if FP8 ckpt present (same layer naming, W8A8 per-tensor)
import os

FP = "/public/share/GLM-5.2-Channel-FP8-w8a8/"
if os.path.isdir(FP) and os.listdir(FP):
    fwm = json.load(open(FP + "model.safetensors.index.json"))["weight_map"] \
        if os.path.exists(FP + "model.safetensors.index.json") else None
    print("FP8 ckpt key style sample:", list(fwm.keys())[:2] if fwm else "no index")
