# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Performance probe: paged prefill, triton kernel vs the emu.

Non-graph wall time per call, bs=4/H=16/Hk=4/D=128 (GQA-4, close to the
Qwen3-4B head layout), wl = prefill window width, kv = wl (fresh prefill).
Mirrors the decode probe methodology (sync + cuda events).
"""

import time

import torch

import sgl_kernel.flash_attn as emu_mod
from sgl_kernel.paged_attn_triton import paged_attention_prefill_triton


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main():
    dev = "cuda"
    B, H, Hk, D, page = 4, 16, 4, 128, 64
    for wl in (512, 2048, 4096):
        torch.manual_seed(0)
        npages = (wl // page + 1) * B + B
        kc = (torch.randn(npages, page, Hk, D, device=dev) * 0.4).to(torch.bfloat16)
        vc = (torch.randn(npages, page, Hk, D, device=dev) * 0.4).to(torch.bfloat16)
        perm = torch.randperm(npages, device=dev).to(torch.int32).reshape(B, -1)
        npt = wl // page
        pt = perm[:, :npt].contiguous()
        sl = torch.full((B,), wl, dtype=torch.int32, device=dev)
        q = (torch.randn(B, wl, H, D, device=dev) * 0.5).to(torch.bfloat16)

        t_tri = bench(lambda: paged_attention_prefill_triton(
            q, kc, vc, pt, sl))
        t_emu = bench(lambda: emu_mod.flash_attn_with_kvcache(
            q=q, k_cache=kc, v_cache=vc, page_table=pt,
            cache_seqlens=sl, causal=True, softmax_scale=D**-0.5))
        print(
            f"wl={wl:5d}: triton={t_tri:7.2f}ms emu={t_emu:7.2f}ms "
            f"speedup={t_emu / t_tri:.2f}x"
        )


if __name__ == "__main__":
    main()
