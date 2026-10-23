# -*- coding: utf-8 -*-
"""Fingerprint probe v2: lightop.paged_mqa_logits (H=32) + adapter emu ragged +
adapter triton paged, on nmz18 gfx938. Chrome-trace kernel parse."""
import json
import os

import torch

torch.manual_seed(0)
dev = "cuda:0"
Q, H, D = 96, 32, 128
BLOCK, WITH_SF = 64, 132
NPAGES = 4096
MAX_SEQ = 8192
T = 1 << 16

def kern_fingerprint(fn, tag, *a, **kw):
    try:
        for _ in range(2):
            out = fn(*a, **kw)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[{tag}] CALL FAIL: {type(e).__name__}: {str(e)[:180]}")
        return None
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(*a, **kw)
        torch.cuda.synchronize()
    path = f"/tmp/fp_{tag.replace(' ', '_').replace('/', '_')}.json"
    prof.export_chrome_trace(path)
    with open(path) as f:
        data = json.load(f)
    from collections import Counter
    names = Counter()
    for e in data["traceEvents"]:
        if e.get("cat") == "kernel":
            n = e.get("name", "")
            names[n.split("<")[0][:100]] += 1
    print(f"== [{tag}] kernels:")
    for n, c in names.most_common(24):
        print(f"   {c:3d}x {n}")
    return out

cache = torch.randint(0, 200, (NPAGES, BLOCK, 1, WITH_SF), dtype=torch.uint8, device=dev)
q = torch.randint(0, 200, (Q, 1, H, D), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
weights = torch.rand(Q, 1, H, dtype=torch.float32, device=dev)
seq_lens = torch.randint(1000, 8000, (Q, 1), dtype=torch.int64, device=dev)
page_table = torch.randint(0, NPAGES, (Q, MAX_SEQ // BLOCK), dtype=torch.int64, device=dev)
sl32 = seq_lens.reshape(-1).to(torch.int32).contiguous()
pt32 = page_table.to(torch.int32).contiguous()

print("== lightop gemmopt")
import lightop.gemmopt as gemmopt
kern_fingerprint(gemmopt.paged_mqa_logits, "lightop_paged",
                q, cache.view(NPAGES, BLOCK, 1, WITH_SF), weights,
                sl32, pt32, None, MAX_SEQ, False)

print("== adapter emu ragged")
from sgl_kernel._adapters import mqa_logits_emu as emu
kv1 = torch.randint(0, 200, (T, WITH_SF), dtype=torch.uint8, device=dev)
cu_s = torch.randint(0, T // 2, (Q,), dtype=torch.int64, device=dev).clamp_min(0)
cu_e = (cu_s + torch.randint(1, 8000, (Q,), device=dev)).clamp_max(T)
kern_fingerprint(emu._ragged_fp8_mqa_logits, "emu_ragged",
                 q.reshape(Q, H, D), kv1, kv1[:, 128:].contiguous().view(torch.float32).reshape(-1),
                 weights.reshape(Q, H), cu_s, cu_e)

print("== adapter triton paged (R-13 #4 default for DSA path)")
try:
    from sgl_kernel._adapters.mqa_triton import triton_paged_mqa_logits
    kern_fingerprint(triton_paged_mqa_logits, "triton_paged",
                     q.reshape(Q, H, D), cache, weights.reshape(Q, H),
                     sl32, pt32, None, MAX_SEQ, False)
except ImportError as e:
    print("triton module import fail:", e)
print("DONE")
