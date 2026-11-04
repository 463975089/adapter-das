# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Probe: fa3-emu (flash_attn_with_kvcache) eager-vs-graph-replay at EAGLE
verify shapes.

Serving bisection (2026-10-17, GLM-5.3-Flash TP8+MTP):
  - no-graph full spec              GREEN
  - graphs, steps1/draft2 (prefill verify) GREEN
  - graphs, steps5/draft6 (decode  verify) GREEN
  - graphs, steps5/draft6 (prefill verify) GARBAGE
=> the prefill-path verify (fa3 emu) at 6 query tokens/req diverges under
   replay. This probe reproduces that in isolation: static input buffers,
   capture, then replay with two different datasets and compare against
   eager runs of the same data.

Usage (container):
  PYTHONPATH=/sglang-v0521/python:/root/hcu_t2 python \
      /tmp/probe_fa3_emu_graph_verify.py
"""

import torch

from sgl_kernel.flash_attn import flash_attn_with_kvcache as fa

DEV = "cuda"
PAGE = 64
H, HK, D, DV = 16, 2, 128, 128
NPAGES_T = 32  # page_table width


def make_data(bs, tq, seed, n_pages_phys=4096):
    g = torch.Generator(device=DEV).manual_seed(seed)
    m = bs * tq
    q = torch.randn(m, H, D, device=DEV, dtype=torch.bfloat16, generator=g)
    k_cache = torch.randn(
        n_pages_phys, PAGE, HK, D, device=DEV, dtype=torch.bfloat16, generator=g
    ) * 0.05
    # ragged cu_q (all rows tq tokens — the chain/topk1 uniform case)
    cu_q = torch.arange(0, m + 1, tq, dtype=torch.int32, device=DEV)
    cache_seqlens = torch.randint(
        4 * PAGE, NPAGES_T * PAGE - PAGE, (bs,), device=DEV, generator=g
    ).to(torch.int32)
    pt = torch.stack(
        [
            torch.randperm(n_pages_phys, generator=g, device=DEV)[:NPAGES_T]
            for _ in range(bs)
        ]
    ).to(torch.int32)
    return dict(
        q=q, k_cache=k_cache, cu_q=cu_q, cache_seqlens=cache_seqlens,
        page_table=pt,
    )


def run_emu(d, tq, max_seqlen_q):
    return fa(
        d["q"], d["k_cache"], d["k_cache"],
        cache_seqlens=d["cache_seqlens"],
        page_table=d["page_table"],
        cu_seqlens_q=d["cu_q"],
        max_seqlen_q=max_seqlen_q,
        causal=True,
    )


def probe(tq):
    bs = 4
    print(f"=== T_q={tq} bs={bs} ===")
    d1 = make_data(bs, tq, seed=1)
    d2 = make_data(bs, tq, seed=2)

    e1 = run_emu(d1, tq, tq).float()
    e2 = run_emu(d2, tq, tq).float()
    print("eager d1 vs d2 differ:", not torch.allclose(e1, e2))

    # static buffers; capture; replay with each dataset in place
    static = {k: v.clone() for k, v in d1.items()}
    # warmup on the capture stream (allocations settle)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            out_w = run_emu(static, tq, tq)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_g = run_emu(static, tq, tq)

    for name, d in (("d1", d1), ("d2", d2)):
        for k in static:
            static[k].copy_(d[k])
        g.replay()
        torch.cuda.synchronize()
        got = out_g.float()
        ref = run_emu(d, tq, tq).float()
        diff = (got - ref).abs().max().item()
        rel = diff / ref.abs().max().clamp(min=1e-6).item()
        n_bad = (got - ref).abs().gt(0.05).sum().item()
        print(f"replay[{name}]: maxabs={diff:.3e} rel={rel:.3e} "
              f"n>0.05={n_bad}/{got.numel()}")


if __name__ == "__main__":
    torch.manual_seed(0)
    probe(2)
    probe(6)
    print("probe done")
