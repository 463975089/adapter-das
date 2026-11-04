# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""v4: plain 4D flash_attn_func vs varlen — isolate which path is broken."""
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_func

torch.manual_seed(0)
dev = "cuda"

def ref(q4, k4, v4, causal, scale):
    a = (q4.float() @ k4.float().transpose(-1, -2)) * scale
    if causal:
        tq, tk = q4.shape[-2], k4.shape[-2]
        m = torch.ones(tq, tk, dtype=torch.bool, device=q4.device).tril(tk - tq)
        a = a.masked_fill(~m, float("-inf"))
    return F.softmax(a, dim=-1) @ v4.float()

def cos(o, r):
    return F.cosine_similarity(o.float().flatten(), r.flatten(), dim=0).item()

for hd in (64, 128):
    for T in (1, 129, 2051):
        q = torch.randn(T, 4, hd, device=dev, dtype=torch.bfloat16) * 1.5
        k = torch.randn(T, 4, hd, device=dev, dtype=torch.bfloat16) * 1.5
        v = torch.randn(T, 4, hd, device=dev, dtype=torch.bfloat16) * 1.5
        scale = 1.0 / hd ** 0.5
        q4, k4, v4 = (t.unsqueeze(0) for t in (q, k, v))

        r_gc = ref(q4, k4, v4, True, scale)[0]
        o = flash_attn_func(q4, k4, v4, causal=True, softmax_scale=scale)
        if isinstance(o, (tuple, list)):
            o = next(t for t in o if torch.is_tensor(t) and t.shape == q.shape)
        o = o.squeeze(0)
        print(f"plain  hd={hd} T={T} causal: cos={cos(o, r_gc):.6f}")

        r_nc = ref(q4, k4, v4, False, scale)[0]
        o = flash_attn_func(q4, k4, v4, causal=False, softmax_scale=scale)
        if isinstance(o, (tuple, list)):
            o = next(t for t in o if torch.is_tensor(t) and t.shape == q.shape)
        o = o.squeeze(0)
        print(f"plain  hd={hd} T={T} full  : cos={cos(o, r_nc):.6f}")

        # varlen single-seq non-causal must equal plain non-causal
        cq = torch.tensor([0, T], dtype=torch.int32, device=dev)
        o = flash_attn_varlen_func(
            q, k, v, cq, cq, T, T,
            dropout_p=0.0, softmax_scale=scale, causal=False)
        if isinstance(o, (tuple, list)):
            o = next(t for t in o if torch.is_tensor(t) and t.shape == q.shape)
        print(f"varlen hd={hd} T={T} full  : cos={cos(o, r_nc):.6f}")
