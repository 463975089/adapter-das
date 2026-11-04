# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Standalone CP-shape test for the HCU fa3 emu flash_attn_with_kvcache.

Mimics exactly what sglang v0.5.21 zigzag CP does per attention layer
(flashattention_backend._fa_cp_attn):
  - one request, seq len S, prefix P (cached), local chunk rows ordered
    [prev_chunk_tokens..., next_chunk_tokens...]
  - KV pool (paged, page=1) holds prefix + full seq (written by
    materialize_full_kv BEFORE attention)
  - TWO calls: q_prev with cache_seqlens=kv_len_prev, q_next with
    cache_seqlens=kv_len_next (cumulative through each chunk's end)
Compare vs a pure-torch full-sequence causal attention reference.

PASS = emu handles CP chunk shapes; FAIL = emu is the CP garbage source.
"""
import os
import sys

import torch

os.environ.setdefault("HIP_VISIBLE_DEVICES", "5")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

from sgl_kernel.flash_attn import flash_attn_with_kvcache as emu_fa

torch.manual_seed(0)
dev = "cuda"
H, Hk, D = 32, 4, 128  # GQA 8
scale = D**-0.5
dtype = torch.bfloat16


def reference(q_full, k_full, v_full, P, S):
    # full causal attention over prefix+seq; return rows for seq tokens
    rep = H // Hk
    kf = k_full.repeat_interleave(rep, dim=1).to(torch.float32)  # (P+S,H,D)
    vf = v_full.repeat_interleave(rep, dim=1).to(torch.float32)
    qf = q_full.to(torch.float32)  # (S, H, D)
    att = torch.einsum("thd,shd->hts", qf, kf) * scale  # (H, S, P+S)
    S_tot = P + S
    # query global index = P + t; token s visible to t iff s <= P + t
    q_idx = torch.arange(P, S_tot, device=dev)[:, None]
    k_idx = torch.arange(0, S_tot, device=dev)[None, :]
    att = att.masked_fill(~(k_idx <= q_idx), float("-inf"))
    p = att.softmax(-1)
    o = torch.einsum("hts,shd->thd", p, vf)
    return o


def main():
    for S, P in ((6, 0), (130, 64), (258, 100)):
        # full-sequence K/V; pool slots hold prefix then seq tokens in order
        k_full = torch.randn(P + S, Hk, D, device=dev, dtype=dtype)
        v_full = torch.randn(P + S, Hk, D, device=dev, dtype=dtype)
        q_full = torch.randn(S, H, D, device=dev, dtype=dtype)

        # zigzag split with cp_size=2: chunks sized ceil/floor halves
        half = (S + 1) // 2
        chunk_sizes = [half, S - half]  # local rank0: chunks 0 and 2 of 4-way
        # 4-way zigzag on local chunk list: [c0, c2] on rank0
        # emulate what rank0 sees: prev = first half chunk, next = last half
        prev_len = half
        next_len = S - half

        # paged pool page=1: (num_blocks, page, H, D)
        pool_k = k_full.view(P + S, 1, Hk, D).clone()
        pool_v = v_full.view(P + S, 1, Hk, D).clone()
        page_table = torch.arange(P + S, device=dev, dtype=torch.int32).view(1, -1)

        # local q rows in zigzag order [prev..., next...]
        q_local = torch.cat([q_full[:prev_len], q_full[prev_len:]], 0)
        q_prev = q_local[:prev_len]
        q_next = q_local[prev_len:]

        # cache_seqlens cumulative through each chunk end (prefix included)
        kv_len_prev = P + prev_len
        kv_len_next = P + S

        def call(qc, kv_len):
            return emu_fa(
                q=qc,
                k_cache=pool_k,
                v_cache=pool_v,
                page_table=page_table,
                cache_seqlens=torch.tensor([kv_len], device=dev, dtype=torch.int32),
                cu_seqlens_q=torch.tensor(
                    [0, qc.shape[0]], device=dev, dtype=torch.int32
                ),
                cu_seqlens_k_new=torch.tensor(
                    [0, kv_len], device=dev, dtype=torch.int32
                ),
                max_seqlen_q=qc.shape[0],
                softmax_scale=scale,
                causal=True,
            )

        o_prev = call(q_prev, kv_len_prev)
        o_next = call(q_next, kv_len_next)

        ref = reference(q_full, k_full, v_full, P, S)
        ref_prev = ref[:prev_len]
        ref_next = ref[prev_len:]

        def rel(a, b):
            return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-6)).item()

        r_prev, r_next = rel(o_prev, ref_prev), rel(o_next, ref_next)
        ok = r_prev < 0.05 and r_next < 0.05
        print(
            f"S={S} P={P}: prev rel={r_prev:.4f} next rel={r_next:.4f} "
            f"{'OK' if ok else 'FAIL'}",
            flush=True,
        )
        if not ok:
            print("FAIL DETAIL emu prev[0,:2,:4]=", o_prev[0, :2, :4].float(), flush=True)
            print("FAIL DETAIL ref prev[0,:2,:4]=", ref_prev[0, :2, :4].float(), flush=True)
            print("FAIL DETAIL emu next[-1,:2,:4]=", o_next[-1, :2, :4].float(), flush=True)
            print("FAIL DETAIL ref next[-1,:2,:4]=", ref_next[-1, :2, :4].float(), flush=True)
            return 1
    print("ALL OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
