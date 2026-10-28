# -*- coding: utf-8 -*-
"""End-to-end: what does deep_gemm.fp8_paged_mqa_logits resolve to in the
SERVER env, and which kernels does the exact C4-indexer call emit?"""
import json

import torch

torch.manual_seed(0)
dev = "cuda:0"

import deep_gemm

fn = deep_gemm.fp8_paged_mqa_logits
print("deep_gemm file:", getattr(deep_gemm, "__file__", "?"))
print("fn module:", getattr(fn, "__module__", "?"), "| qualname:", getattr(fn, "__qualname__", "?"))

from hcu_adapter.hcu_env import resolve

impl, src = resolve("HCU_MQA_LOGITS_IMPL")
print(f"resolve(HCU_MQA_LOGITS_IMPL) = {impl!r} ({src})")

# exact C4 shapes: q [N,1,H,128] fp8, kv 2-D view of (P,64,1,132), w [N,H],
# ctx [N] (v0522 passes _c4sl int32/64), bt [N,L] page table
N, H, D = 16, 32, 128
P, BLOCK, WITH_SF = 8192, 64, 132
MAXC = 16384
q = torch.randint(0, 200, (N, 1, H, D), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
kv = torch.randint(0, 200, (P * BLOCK * WITH_SF,), dtype=torch.uint8, device=dev)
w = torch.rand(N, H, dtype=torch.float32, device=dev)
ctx = torch.randint(2000, 15000, (N,), dtype=torch.int64, device=dev)
bt = torch.randint(0, P, (N, MAXC // BLOCK), dtype=torch.int32, device=dev)

for _ in range(2):
    out = fn(q, kv, w, ctx, bt, None, MAXC, False)
torch.cuda.synchronize()
print("out:", tuple(out.shape), out.dtype)

from torch.profiler import ProfilerActivity, profile

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    fn(q, kv, w, ctx, bt, None, MAXC, False)
    torch.cuda.synchronize()
prof.export_chrome_trace("/tmp/mqa3.json")
with open("/tmp/mqa3.json") as f:
    data = json.load(f)
from collections import Counter

names = Counter()
for e in data["traceEvents"]:
    if e.get("cat") == "kernel":
        names[e.get("name", "")[:120]] += 1
print("== kernels for ONE C4-shaped call:")
for n, c in names.most_common(30):
    print(f"   {c:3d}x {n}")
