# -*- coding: utf-8 -*-
"""Can the DTK aiter wheel's ORIGINAL ragged fp8_mqa_logits run on gfx938?
Reload the module (restoring the pre-emu definitions), call it, check
parity vs the emu golden."""
import inspect
import json

import torch

torch.manual_seed(0)
dev = "cuda:0"

import importlib

import aiter.ops.triton.fp8_mqa_logits as afml

print("bound fp8_mqa_logits:", afml.fp8_mqa_logits.__module__, afml.fp8_mqa_logits.__qualname__)
afml = importlib.reload(afml)
orig = afml.fp8_mqa_logits
print("after reload:", orig.__module__, orig.__qualname__)
try:
    print("signature:", inspect.signature(orig))
except Exception:
    pass

N, H, D, T = 16, 32, 128, 8192
q = (torch.randn(N, H, D, device=dev) * 3).clamp(-20, 20).to(torch.float8_e4m3fn)
k = (torch.randn(T, D, device=dev) * 3).clamp(-20, 20).to(torch.float8_e4m3fn)
sc = torch.rand(T, dtype=torch.float32, device=dev) + 0.5
w = torch.rand(N, H, dtype=torch.float32, device=dev)
ks = torch.zeros(N, dtype=torch.int32, device=dev)
ke = torch.randint(1000, 8000, (N,), dtype=torch.int32, device=dev)

from sgl_kernel._adapters import mqa_logits_emu as emu

golden = emu._ragged_fp8_mqa_logits(q, k, sc, w, ks.to(torch.int64), ke.to(torch.int64))

# try the original with the aiter ABI (q, k, kv_scales, weights, ks, ke)
out = None
errs = []
for attempt in (
    lambda: orig(q, k, sc, w, ks, ke),
    lambda: orig(q, k, sc, w, ks, ke, max_seqlen_k=T),
    lambda: orig(q, k, sc, w, ks, ke, T),
):
    try:
        out = attempt()
        break
    except Exception as e:
        errs.append(f"{type(e).__name__}: {str(e)[:120]}")
if out is None:
    print("ORIGINAL FAILED:")
    for e in errs:
        print("  ", e)
else:
    print("original out:", tuple(out.shape), out.dtype)
    g = golden.float()
    o = out.float()
    m = torch.isfinite(o) & torch.isfinite(g)
    rel = ((o[m] - g[m]).abs() / g[m].abs().clamp_min(1e-6)).median()
    print(f"parity median rel = {rel:.5f}")
    from torch.profiler import ProfilerActivity, profile

    for _ in range(2):
        attempt()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        attempt()
        torch.cuda.synchronize()
    prof.export_chrome_trace("/tmp/aiter_orig.json")
    data = json.load(open("/tmp/aiter_orig.json"))
    from collections import Counter
    names = Counter()
    for e in data["traceEvents"]:
        if e.get("cat") == "kernel":
            names[e.get("name", "")[:100]] += 1
    print("== original kernels:")
    for n, c in names.most_common(12):
        print(f"   {c:3d}x {n}")
print("DONE")
