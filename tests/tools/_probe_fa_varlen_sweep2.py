# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
_probe_: same sweep on wide-context / narrow-extend shapes (long-ctx
(512,4096),(256,2048) and fp8kv-extend (64,520),(1,33)). Census: the
shipped defaults (BM64/BN64/W4) are the only fast arm on these shapes
too (2.8ms vs 200ms+ cliffs); BN>=256 Exhausts shared memory.

Run: docker exec trt-fa3pf bash -c 'TVM_FFI_DISABLE_TORCH_C_DLPACK=1 \n  PYTHONPATH=/work/hcu_trt_fa3:/work/sglang-v0521/python HIP_VISIBLE_DEVICES=4 \n  python -u /work/hcu_trt_fa3/tests/tools/_probe_fa_varlen_sweep2.py'
"""

import time, torch
from sgl_kernel.fa_varlen_triton import flash_attn_varlen_triton as k

torch.manual_seed(1)
def mk(seqs, H, Hk, D):
    dev = "cuda"
    cu_q = torch.tensor([0] + [s[0] for s in seqs], device=dev, dtype=torch.int32).cumsum(0).int()
    cu_k = torch.tensor([0] + [s[1] for s in seqs], device=dev, dtype=torch.int32).cumsum(0).int()
    q = (torch.randn(int(cu_q[-1]), H, D) * .5).bfloat16().to(dev)
    kb = (torch.randn(int(cu_k[-1]), Hk, D) * .4).bfloat16().to(dev)
    vb = (torch.randn(int(cu_k[-1]), Hk, D) * .4).bfloat16().to(dev)
    return q, kb, vb, cu_q, cu_k

def tm(fn, n=10):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return round((time.time() - t) / n * 1000, 3)

cases = {
    "long-ctx (512,4096)(256,2048) H8Hk4": dict(seqs=[(512, 4096), (256, 2048)], H=8, Hk=4),
    "fp8kv-extend (64,520)(1,33) H8Hk8": dict(seqs=[(64, 520), (1, 33)], H=8, Hk=8),
}
for name, c in cases.items():
    q, kb, vb, cu_q, cu_k = mk(c["seqs"], c["H"], c["Hk"], 128)
    maxL = max(s[0] for s in c["seqs"])
    print(f"== {name}")
    for bm in (16, 32, 64):
        for bn in (64, 128, 256):
            for w in (2, 4):
                try:
                    ms = tm(lambda: k(q, kb, vb, cu_q, cu_k,
                                      max_seqlen_q=maxL, BLOCK_M=bm,
                                      BLOCK_N=bn, num_warps=w))
                    print(f"  BM={bm} BN={bn} W={w}: {ms}", flush=True)
                except Exception as e:
                    print(f"  BM={bm} BN={bn} W={w}: EXC {str(e)[:40]}", flush=True)
print("DONE2")
