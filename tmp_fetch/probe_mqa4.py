# -*- coding: utf-8 -*-
"""Fingerprint both deep_gemm MQA entries with exact C4 production shapes."""
import json

import torch

torch.manual_seed(0)
dev = "cuda:0"
N, H, D = 16, 32, 128
P, BLOCK, WITH_SF = 8192, 64, 132
MAXC = 16384

import deep_gemm
from torch.profiler import ProfilerActivity, profile


def fp(tag, call):
    try:
        for _ in range(2):
            out = call()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[{tag}] FAIL {type(e).__name__}: {str(e)[:150]}")
        return
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    prof.export_chrome_trace(f"/tmp/{tag}.json")
    data = json.load(open(f"/tmp/{tag}.json"))
    from collections import Counter
    names = Counter()
    for e in data["traceEvents"]:
        if e.get("cat") == "kernel":
            names[e.get("name", "").split("<")[0][:100]] += 1
    print(f"== [{tag}]:")
    for n, c in names.most_common(26):
        print(f"   {c:3d}x {n}")


q4 = torch.randint(0, 200, (N, 1, H, D), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
w = torch.rand(N, H, dtype=torch.float32, device=dev)
ctx = torch.randint(2000, 15000, (N,), dtype=torch.int64, device=dev)
bt = torch.randint(0, P, (N, MAXC // BLOCK), dtype=torch.int32, device=dev)
kv2d = torch.randint(0, 200, (P, BLOCK * WITH_SF), dtype=torch.uint8, device=dev)

fp("paged_2d", lambda: deep_gemm.fp8_paged_mqa_logits(
    q4, kv2d, w, ctx, bt, None, MAXC, False))

# ragged entry (nonpaged path): q [N,H,D] fp8, kv=(k_fp8 [T,128], scale [T] f32)
T = 12288
q3 = q4.reshape(N, H, D)
k_t = torch.randint(0, 200, (T, D), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
sc_t = torch.rand(T, dtype=torch.float32, device=dev)
ks = torch.zeros(N, dtype=torch.int64, device=dev)
ke = torch.randint(2000, 12000, (N,), dtype=torch.int64, device=dev)
fp("ragged", lambda: deep_gemm.fp8_mqa_logits(
    q3, (k_t, sc_t), w, ks, ke, clean_logits=False, max_seqlen_k=T))
print("DONE")
