# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Probe: do the spec KV-alloc triton kernels miscompile on DTK/gfx938?

The sglang-das fork ships C++ hcu_ ports of exactly these kernels with
in-tree comments blaming a mixed-width int32->int64 store miscompile. We
run is_cuda semantics, so stock dispatches the triton kernels. Compare each
against a pure-torch reference on synthetic batches at the REAL call-site
dtypes (req_to_token int32, lens int64, out int64 — the mixed widths).

Usage (container):
  PYTHONPATH=/sglang-v0521/python:/root/hcu_t2 python \
      /tmp/probe_spec_kvalloc_triton.py
"""

import torch

from sglang.kernels.ops.speculative.cache_locs import (
    assign_extend_cache_locs as aec_triton,
)
from sglang.kernels.ops.memory.allocator import (
    alloc_decode_kernel,
    alloc_extend_kernel,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PAGE = 64


def torch_assign_extend(req_pool, req_to_token, start, end, out, pool_len):
    # out[cumsum(end-start)[i] + j] = req_to_token[pool_i, start_i + j]
    lens = (end - start).tolist()
    offs = [0]
    for l in lens[:-1]:
        offs.append(offs[-1] + l)
    for i, (s, e) in enumerate(zip(start.tolist(), end.tolist())):
        out[offs[i] : offs[i] + (e - s)] = req_to_token[
            req_pool[i].item(), s:e
        ]
    return out


def torch_alloc_decode(seq_lens, last_loc, free_pages, out, page):
    # out[i] = last_loc+1 if same page else free_pages[cumsum_new]*page
    ceil = lambda x: (x + page - 1) // page
    new_pages = ceil(seq_lens) - ceil(seq_lens - 1)
    starts = torch.cumsum(new_pages, 0) - new_pages
    for i in range(len(seq_lens)):
        if new_pages[i] == 0:
            out[i] = last_loc[i].item() + 1
        else:
            out[i] = free_pages[starts[i].item()].item() * page
    return out


def ceil_div(a, b):
    return (a + b - 1) // b


def torch_alloc_extend(pre, seq, last_loc, free_pages, out, page):
    # Mirrors launch_alloc_extend_kernel from the das fork (C++ port).
    ext = seq - pre
    out_start = torch.cumsum(ext, 0) - ext
    new_pages = ceil_div_seq(seq, page) - ceil_div_seq(pre, page)
    new_start = torch.cumsum(new_pages, 0) - new_pages
    for i in range(len(pre)):
        s, e = int(pre[i]), int(seq[i])
        os_ = int(out_start[i])
        ll = int(last_loc[i])
        num1 = min(e, ceil_div(s, page) * page) - s
        for o in range(num1):
            out[os_ + o] = ll + 1 + o
        if s + num1 == e:
            continue
        num2 = (e // page) * page - ceil_div(s, page) * page
        for o in range(num2):
            pi = int(new_start[i]) + o // page
            out[os_ + num1 + o] = int(free_pages[pi]) * page + o % page
        if s + num1 + num2 == e:
            continue
        num3 = e - (e // page) * page
        lp = int(new_start[i]) + int(new_pages[i]) - 1
        sl = int(free_pages[lp])
        for o in range(num3):
            out[os_ + num1 + num2 + o] = sl * page + o
    return out


def ceil_div_seq(x, page):
    return (x + page - 1) // page * 0 + torch.div(
        x, page, rounding_mode="floor"
    ).where(x % page == 0, torch.div(x, page, rounding_mode="floor") + 1)


def cmp(name, got, want):
    ok = torch.equal(got, want)
    if not ok:
        bad = (got != want).nonzero().flatten()[:6].tolist()
        print(f"[{name}] MISMATCH bad_idx={bad} got={got[bad].tolist()} "
              f"want={want[bad].tolist()}")
    else:
        print(f"[{name}] OK")
    return ok


def main():
    torch.manual_seed(0)
    bs = 8
    pool_len = 8192
    n_pool_rows = 16

    # --- assign_extend_cache_locs (move_accept_tokens path) -------------
    req_to_token = torch.randint(
        1, 2**31 - 1, (n_pool_rows, pool_len), dtype=torch.int32, device=DEV
    )
    req_pool = torch.arange(bs, dtype=torch.int64, device=DEV)
    start = torch.zeros(bs, dtype=torch.int64, device=DEV)
    lens = torch.randint(1, 200, (bs,), device=DEV).to(torch.int64)
    end = start + lens
    out_t = torch.full((int(lens.sum()),), -7, dtype=torch.int64, device=DEV)
    out_r = out_t.clone()
    from sglang.srt.utils import next_power_of_2

    aec_triton[(bs,)](
        req_pool, req_to_token, start, end, out_t, pool_len,
        next_power_of_2(bs),
    )
    torch_assign_extend(req_pool, req_to_token, start, end, out_r, pool_len)
    cmp("assign_extend_cache_locs", out_t, out_r)

    # --- alloc_decode -----------------------------------------------------
    seq = torch.randint(1, 5000, (bs,), device=DEV).to(torch.int64)
    last = (torch.div(seq - 1, PAGE, rounding_mode="floor") * PAGE - 1).clamp(
        min=0
    ).to(torch.int32)
    free = torch.arange(1000, 1000 + 4 * bs, dtype=torch.int64, device=DEV)
    od_t = torch.full((bs,), -7, dtype=torch.int64, device=DEV)
    od_r = od_t.clone()
    alloc_decode_kernel[(bs,)](
        seq, last, free, od_t, next_power_of_2(bs), PAGE
    )
    torch_alloc_decode(seq, last, free, od_r, PAGE)
    cmp("alloc_decode_kernel", od_t, od_r)

    # --- alloc_extend -----------------------------------------------------
    pre = torch.randint(0, 3000, (bs,), device=DEV).to(torch.int64)
    ext = torch.randint(1, 300, (bs,), device=DEV).to(torch.int64)
    seq2 = pre + ext
    # real chain: get_last_loc_torch returns int32 (gathers int32
    # req_to_token); HCU_SPEC_LOC_SAFE routes serving through it
    last2 = (pre - 1).clamp(min=-1).to(torch.int32)
    free2 = torch.arange(
        2000, 2000 + 4 * bs * 4, dtype=torch.int64, device=DEV
    )
    oe_t = torch.full((int(ext.sum()),), -7, dtype=torch.int64, device=DEV)
    oe_r = oe_t.clone()
    alloc_extend_kernel[(bs,)](
        pre, seq2, last2, free2, oe_t, next_power_of_2(bs), PAGE
    )
    torch_alloc_extend(pre, seq2, last2.to(torch.int64), free2, oe_r, PAGE)
    cmp("alloc_extend_kernel(int32 last_loc)", oe_t, oe_r)

    # --- bigger batch, int32 last_loc, non-pow2 bs ------------------------
    bs = 48
    pre = torch.randint(0, 3000, (bs,), device=DEV).to(torch.int64)
    ext = torch.randint(1, 700, (bs,), device=DEV).to(torch.int64)
    seq2 = pre + ext
    last2 = (pre - 1).clamp(min=-1).to(torch.int32)
    free2 = torch.arange(
        5000, 5000 + 4 * bs * 8, dtype=torch.int64, device=DEV
    )
    oe_t = torch.full((int(ext.sum()),), -7, dtype=torch.int64, device=DEV)
    oe_r = oe_t.clone()
    alloc_extend_kernel[(bs,)](
        pre, seq2, last2, free2, oe_t, next_power_of_2(bs), PAGE
    )
    torch_alloc_extend(pre, seq2, last2.to(torch.int64), free2, oe_r, PAGE)
    cmp("alloc_extend_kernel(bs=48,int32)", oe_t, oe_r)

    # --- assign_extend with int32 req_pool (another real dtype) ----------
    req_pool32 = torch.arange(bs, dtype=torch.int32, device=DEV)
    start = torch.zeros(bs, dtype=torch.int64, device=DEV)
    lens = torch.randint(1, 200, (bs,), device=DEV).to(torch.int64)
    end = start + lens
    out_t = torch.full((int(lens.sum()),), -7, dtype=torch.int64, device=DEV)
    out_r = out_t.clone()
    aec_triton[(bs,)](
        req_pool32, req_to_token, start, end, out_t, pool_len,
        next_power_of_2(bs),
    )
    torch_assign_extend(req_pool32, req_to_token, start, end, out_r, pool_len)
    cmp("assign_extend_cache_locs(int32 pool)", out_t, out_r)

    print("probe done")


if __name__ == "__main__":
    main()
