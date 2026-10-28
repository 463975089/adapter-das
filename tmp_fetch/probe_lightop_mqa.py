# -*- coding: utf-8 -*-
"""Fingerprint probe: what kernels do (a) lightop.gemmopt.paged_mqa_logits and
(b) the adapter MQA emu actually launch, at C4-indexer decode shapes?
Compare against the wave6 per-layer trace chain."""
import os
import sys

import torch

torch.manual_seed(0)
dev = torch.device("cuda:0")
Q, H, D = 96, 64, 128          # verify tokens, index heads, head dim
BLOCK, WITH_SF = 64, 132        # C4 cache block, 128 fp8 + 4B scale
NPAGES = 4096
MAX_SEQ = 8192

# C4 cache: [pages, 64, 1, 132] raw u8
cache = torch.randint(0, 200, (NPAGES, BLOCK, 1, WITH_SF), dtype=torch.uint8, device=dev)
q = torch.randint(0, 200, (Q, 1, H, D), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
weights = torch.rand(Q, 1, H, dtype=torch.float32, device=dev)
seq_lens = torch.randint(1000, 8000, (Q,), dtype=torch.int64, device=dev)
page_table = torch.randint(0, NPAGES, (Q, MAX_SEQ // BLOCK), dtype=torch.int64, device=dev)

def fingerprint(fn, tag, *a, **kw):
    for _ in range(2):  # warmup (jit/lazy init outside the profile)
        try:
            fn(*a, **kw)
        except Exception as e:
            print(f"[{tag}] warmup fail: {type(e).__name__}: {e}")
            return
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(*a, **kw)
        torch.cuda.synchronize()
    print(f"== [{tag}] kernel fingerprint:")
    for ev in prof.key_averages():
        if ev.device_type == torch.autograd.DeviceType.CUDA or "cuda" in str(ev.device_type).lower():
            continue  # skip cpu rows
    for ev in prof.events():
        if str(ev.device_type).endswith("CUDA") and ev.kind is not None:
            pass
    names = []
    for ev in prof.events():
        try:
            if ev.device_type.name == "CUDA" and ev.kind.name in ("KERNEL", "MEMCPY", "MEMSET"):
                names.append(ev.name[:110])
        except Exception:
            pass
    from collections import Counter
    for n, c in Counter(names).most_common(40):
        print(f"   {c:3d}x {n}")

print("importing lightop.gemmopt ...")
import lightop.gemmopt as gemmopt

fn = gemmopt.paged_mqa_logits
sl32 = seq_lens.to(torch.int32).contiguous()
pt32 = page_table.to(torch.int32).contiguous()
cache2 = cache.view(NPAGES * BLOCK, 1, WITH_SF)  # flat token view variant
try:
    fingerprint(fn, "lightop.paged_mqa_logits q_fp8",
                q, cache.view(NPAGES, BLOCK, 1, WITH_SF), weights,
                sl32, pt32, None, MAX_SEQ, False)
except TypeError as e:
    print("signature mismatch, trying variants:", e)

# adapter emu fingerprint for comparison (legacy block-1 shapes)
from hcu_adapter import mqa_logits_emu as emu

cache1 = torch.randint(0, 200, (1 << 16, WITH_SF), dtype=torch.uint8, device=dev)
bt1 = torch.randint(0, 1 << 16, (Q, MAX_SEQ), dtype=torch.int64, device=dev)
fingerprint(emu._paged_mqa_logits, "adapter-emu paged",
            q.view(Q, H, D).view(torch.float8_e4m3fn), cache1, weights.view(Q, H),
            seq_lens, bt1, MAX_SEQ, preshuffle=False, kv_block_size=1)
print("done")
