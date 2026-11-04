# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""One-token MLA forward parity: dequant-to-bf16 reference vs real weights.

Loads layer-7 attention weights from GLM-5-W4A8_V2_6 (q_a/q_b/kv_a/kv_b/o_proj,
all W8 int8 + storage scale = truth), runs:
  A) pure-torch MLA (MLA absorb-free Naive: full k/v heads) fp32 reference
  B) pure-torch MLA absorbed (kv_lora 512 compressed form)
and checks A vs B agree. This validates our understanding of the checkpoint's
MLA head-shape math (qk_nope=192 + rope=64, kv_lora=512, v_head_dim=256).
No sglang involvement -- pure checkpoint semantics.
"""

import json
import struct

import numpy as np
import torch
import torch.nn.functional as F

CKPT = "/public/share/GLM-5-W4A8_V2_6/"
KEY = "model.layers.7.self_attn."
wm = json.load(open(CKPT + "model.safetensors.index.json"))["weight_map"]


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
    return torch.from_numpy(np.frombuffer(raw, dtype=dt).reshape(v["shape"]).copy())


def deq(name):
    w = read_shard(KEY + name + ".weight").float()
    if name + ".weight_scale" in wm:
        s = read_shard(KEY + name + ".weight_scale").float().view(-1)
        return w * s.view(-1, 1)
    return w


torch.manual_seed(0)
H = 6144
QK_HEAD, QK_NOPE, QK_ROPE = 256, 192, 64
V_HEAD = 256
KV_LORA = 512
Q_LORA = 2048
NHEAD = 64

q_a = deq("q_a_proj")          # [Q_LORA, H]
kv_a = deq("kv_a_proj_with_mqa")  # [KV_LORA + QK_ROPE, H]
q_b = deq("q_b_proj")          # [NHEAD*QK_HEAD, Q_LORA]
kv_b = deq("kv_b_proj")        # [NHEAD*(QK_NOPE+V_HEAD), KV_LORA]
o = deq("o_proj")              # [H, NHEAD*V_HEAD]
q_a_ln = read_shard(KEY + "q_a_layernorm.weight").view(torch.bfloat16).float()
kv_a_ln = read_shard(KEY + "kv_a_layernorm.weight").view(torch.bfloat16).float()

print("shapes ok:",
      q_a.shape == (Q_LORA, H), kv_a.shape == (KV_LORA + QK_ROPE, H),
      q_b.shape == (NHEAD * QK_HEAD, Q_LORA),
      kv_b.shape == (NHEAD * (QK_NOPE + V_HEAD), KV_LORA),
      o.shape == (H, NHEAD * V_HEAD))

T = 3
x = (torch.randn(T, H) * 0.5)

# forward (DeepSeek MLA): down-projs -> per-branch RMSNorm -> up-projs
q_latent = F.rms_norm(x.float() @ q_a.t(), (Q_LORA,), q_a_ln, 1e-5)   # [T, 2048]
kva = x.float() @ kv_a.t()                        # [T, KV_LORA+64]
c_kv = F.rms_norm(kva[:, :KV_LORA], (KV_LORA,), kv_a_ln, 1e-5)
k_rope = kva[:, KV_LORA:]                         # [T, 64]

q = (q_latent @ q_b.t()).view(T, NHEAD, QK_HEAD)
q_nope, q_rope = q[..., :QK_NOPE], q[..., QK_NOPE:]
kv = (c_kv @ kv_b.t()).view(T, NHEAD, QK_NOPE + V_HEAD)
k_nope, v = kv[..., :QK_NOPE], kv[..., QK_NOPE:]  # [T, NH, 192]/[T, NH, 256]

# structural parity with rope postponed (identity rope): scores from nope
# dims only are compared between the two formulations.

# A) naive per-head
scores = torch.einsum("thd,shd->hts", q_nope, k_nope)
mask = torch.full((T, T), float("-inf")).triu(1)
probs = (scores * (1.0 / QK_HEAD ** 0.5) + mask).softmax(-1)
out_naive = torch.einsum("hts,shd->thd", probs, v).reshape(T, NHEAD * V_HEAD)
final_naive = out_naive @ o.t()

# B) absorbed: q_nope @ W_UK -> latent space; single (192+0)-dot
W_uk = kv_b.view(NHEAD, QK_NOPE + V_HEAD, KV_LORA)[:, :QK_NOPE, :]   # [NH,192,512]
W_uv = kv_b.view(NHEAD, QK_NOPE + V_HEAD, KV_LORA)[:, QK_NOPE:, :]   # [NH,256,512]
qn_abs = torch.einsum("thd,hdk->htk", q_nope, W_uk)                  # [T,NH,512]
scores_abs = torch.einsum("htk,sk->hts", qn_abs, c_kv)
probs_abs = (scores_abs * (1.0 / QK_HEAD ** 0.5) + mask).softmax(-1)
# absorbed value: v = probs @ W_uv @ c_kv
out_abs = torch.einsum("htk,hdk->thd", probs_abs @ c_kv, W_uv)
out_abs = out_abs.reshape(T, NHEAD * V_HEAD)
final_abs = out_abs @ o.t()

for tag, a, b in (("out", out_naive, out_abs), ("final", final_naive, final_abs)):
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rel = ((a - b).abs() / (b.abs() + 1e-5)).mean().item()
    print(f"{tag}: cos={cos:.6f} rel_mean={rel:.6f}")
