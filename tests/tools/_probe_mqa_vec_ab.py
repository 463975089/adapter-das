# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""MQA logits emu 对拍：loop（tile 循环参考）vs vec（全窗口单遍）。

2026-10-17 图瘦身第一刀：decode 图 72% 节点来自 loop 形态的 32×~12 aten
微核/层；vec 一次 gather+bmm 替代。V1 收缩走二次 bmm（不物化 (N,H,T)
fp32——DTK 大张量 bfloat16_to_float 有 ~0.8GB/s 崖，朴素 vec 曾 3.5× 慢
于 loop）。本探针在 decode 形状（N≤16、乱序 context、132 列 fp8 kv、
ragged 页表）上双实现对拍 + 计时 + top-k 一致率。

跑法：容器内 GPU 上 `python tests/tools/_probe_mqa_vec_ab.py`。
"""
import os
import time

import torch

torch.manual_seed(11)
DEV = "cuda"


def _mk(N, H=16, D=128, T=65536, PS=64, ragged=True):
    q = (torch.randn(N, H, D, device=DEV) * 0.3).to(torch.float8_e4m3fn)
    pages_total = (T // PS) * 2
    kv = torch.zeros(pages_total, PS, 1, 132, device=DEV, dtype=torch.uint8)
    kb = (torch.randn(pages_total, PS, 1, D, device=DEV) * 0.2).to(torch.float8_e4m3fn)
    kv[..., :D] = kb.view(torch.uint8)
    scale = (torch.rand(pages_total, PS, 1, 1, device=DEV) * 0.02 + 0.99).float()
    kv[..., D:] = scale.view(torch.uint8).reshape(pages_total, PS, 1, 4)
    w = torch.rand(N, H, device=DEV).float()
    ctx = torch.randint(1, T, (N, 1), device=DEV, dtype=torch.int32)
    if not ragged:
        ctx[:] = T
    bt_cols = T // PS
    bt = torch.stack([
        torch.randperm(pages_total)[:bt_cols] for _ in range(N)
    ]).to(torch.int32).to(DEV)
    return q, kv, w, ctx, bt, T


def run(q, kv, w, ctx, bt, T):
    from hcu_adapter.dsa_mqa_logits_emu import fp8_paged_mqa_logits
    return fp8_paged_mqa_logits(q, kv, w, ctx, bt, None, T, clean_logits=True)


def bench(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


os.environ["HCU_MQA_LOGITS_IMPL"] = "loop"
from hcu_adapter.hcu_env import resolve  # noqa: E402

print("resolved impl:", resolve("HCU_MQA_LOGITS_IMPL"))
TOPK = 8
for N in (1, 8, 16):
    q, kv, w, ctx, bt, T = _mk(N)
    os.environ["HCU_MQA_LOGITS_IMPL"] = "loop"
    r_loop = run(q, kv, w, ctx, bt, T)
    t_loop = bench(lambda: run(q, kv, w, ctx, bt, T))
    os.environ["HCU_MQA_LOGITS_IMPL"] = "vec"
    r_vec = run(q, kv, w, ctx, bt, T)
    t_vec = bench(lambda: run(q, kv, w, ctx, bt, T))
    a, b = r_loop.float(), r_vec.float()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    md = (a - b).abs().max().item()
    rel = ((a - b).norm() / (a.norm() + 1e-9)).item()
    # top-k agreement inside each row's valid context (what downstream eats)
    agree, tot = 0, 0
    for i in range(N):
        v = int(ctx[i, 0])
        tl = set(a[i, :v].topk(min(TOPK, v)).indices.tolist())
        tv = set(b[i, :v].topk(min(TOPK, v)).indices.tolist())
        agree += len(tl & tv)
        tot += len(tl)
    ok = rel < 1e-2 and cos > 0.9999 and agree / tot > 0.95
    print(f"N={N:2d} T={T}: cos={cos:.8f} maxdiff={md:.3e} rel={rel:.3e} "
          f"topk@{TOPK}={agree}/{tot}  loop={t_loop:.1f}ms vec={t_vec:.1f}ms "
          f"({t_loop/t_vec:.2f}x)  {'PASS' if ok else 'FAIL'}")
print("DONE")
