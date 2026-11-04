# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
_probe_: fa_varlen_triton BLOCK_M/BLOCK_N/num_warps sweep on the serving
shape (bs8 Lq512 Lk4096 H32 Hk8 D128). Census: BM=64/BN=64/W=4 optimal;
W=2 and BM<64 are 3-10x cliffs (HCU triton waveform cliff, not a kernel
choice). Perf numbers BW1101 - not transferable to gfx938.

Run: docker exec trt-fa3pf bash -c 'TVM_FFI_DISABLE_TORCH_C_DLPACK=1 \n  PYTHONPATH=/work/hcu_trt_fa3:/work/sglang-v0521/python HIP_VISIBLE_DEVICES=4 \n  python -u /work/hcu_trt_fa3/tests/tools/_probe_fa_varlen_sweep.py'
"""

import time, torch
from sgl_kernel.fa_varlen_triton import flash_attn_varlen_triton as k

torch.manual_seed(1)
B, Lq, Lk, H, Hk, D = 8, 512, 4096, 32, 8, 128
dev = "cuda"
cu_q = torch.arange(B + 1, device=dev, dtype=torch.int32) * Lq
cu_k = torch.arange(B + 1, device=dev, dtype=torch.int32) * Lk
q = (torch.randn(B * Lq, H, D) * 0.5).bfloat16().to(dev)
kb = (torch.randn(B * Lk, Hk, D) * 0.4).bfloat16().to(dev)
vb = (torch.randn(B * Lk, Hk, D) * 0.4).bfloat16().to(dev)

def tm(fn, n=8):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return round((time.time() - t) / n * 1000, 2)

print("sweep1 bs8 Lq512 Lk4096 H32 Hk8 D128")
for bm in (32, 64, 128):
    for bn in (64, 128):
        for w in (4, 8):
            try:
                ms = tm(lambda: k(q, kb, vb, cu_q, cu_k,
                                  max_seqlen_q=Lq, BLOCK_M=bm,
                                  BLOCK_N=bn, num_warps=w))
                print(f"BM={bm} BN={bn} W={w}: {ms}", flush=True)
            except Exception as e:
                print(f"BM={bm} BN={bn} W={w}: EXC {str(e)[:50]}", flush=True)
print("DONE1")
