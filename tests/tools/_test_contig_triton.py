# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Unit gate for the triton contiguous grouped GEMM (DeepEP normal, bf16).

Contract: d[m,:] = a[m,:] @ b[m_indices[m]].T with fp32 accumulate, bf16 io.
Rows are grouped by expert with 128-aligned segments (deep_ep normal
dispatch + ep_scatter contract); pad rows carry a valid expert id over
garbage activations.

Run: HIP_VISIBLE_DEVICES=6 python tests/tools/_test_contig_triton.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hcu_adapter.deep_gemm_contig_triton import (  # noqa: E402
    get_mk_alignment_for_contiguous_layout,
    m_grouped_bf16_gemm_nt_contiguous,
)

ALIGN = get_mk_alignment_for_contiguous_layout()
assert ALIGN == 128, ALIGN
torch.manual_seed(7)
FAILS = []


def make_case(E, N, K, group_counts, pad_garbage=True):
    """group_counts: list of 128-aligned real counts per expert."""
    m_idx = []
    for e in range(E):
        seg = [e] * group_counts[e]
        m_idx.extend(seg)
    m_indices = torch.tensor(m_idx, dtype=torch.int32, device="cuda")
    M = len(m_idx)
    assert M % ALIGN == 0
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    if pad_garbage:
        # sprinkle garbage into pad rows (segment tails): simulates the
        # torch.empty slots ep_scatter never writes
        for e in range(E):
            n = group_counts[e]
            if n > ALIGN:
                a[e * n + (n - ALIGN) // 2 : e * n + n] = float("nan")
    b = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.1
    d = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    return a, b, d, m_indices


def ref(a, b, m_indices):
    """fp32 reference per expert segment; pad rows are checked only for
    finiteness-independence (real rows must not see their NaNs)."""
    M, K = a.shape
    out = torch.empty(M, b.shape[1], dtype=torch.float32, device="cuda")
    for e in torch.unique(m_indices).tolist():
        rows = (m_indices == e).nonzero(as_tuple=True)[0]
        # real rows only: exclude the NaN pad block
        rows = rows[torch.isfinite(a[rows, :].float()).all(dim=1)]
        out[rows] = a[rows].float() @ b[e].float().T
    return out


def run(tag, E, N, K, counts):
    a, b, d, m_indices = make_case(E, N, K, counts)
    m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices)
    torch.cuda.synchronize()
    r = ref(a, b, m_indices)
    fin = torch.isfinite(a.float()).all(dim=1)
    got = d.float()[fin]
    want = r[fin]
    md = (got - want).abs().max().item()
    scale = want.abs().max().item() + 1e-6
    ok = md / scale < 3e-2
    print(f"{tag}: E={E} N={N} K={K} M={sum(counts)} "
          f"maxdiff={md:.4f} rel={md/scale:.2e} {'OK' if ok else 'FAIL'}",
          flush=True)
    if not ok:
        FAILS.append(tag)


# small sanity
run("small", E=4, N=256, K=512,
    counts=[ALIGN * 1, ALIGN * 3, ALIGN * 1, ALIGN * 2])
# Qwen3-30B-A3B gateup: E=128 local(64? EP2 -> 64) use 8 experts for speed
run("gateup-like", E=8, N=3072, K=2048,
    counts=[ALIGN * (1 + (i % 7)) for i in range(8)])
run("down-like", E=8, N=2048, K=1536,
    counts=[ALIGN * (1 + ((i * 3) % 5)) for i in range(8)])
# ragged K/N tails
run("tails", E=3, N=1000, K=700,
    counts=[ALIGN, ALIGN * 2, ALIGN])

# perf smoke (prefill-ish): M=32*128=4096, E=32, gateup dims
E, N, K = 32, 3072, 2048
counts = [ALIGN * 4 for _ in range(E)]
a, b, d, m_indices = make_case(E, N, K, counts, pad_garbage=False)
for _ in range(5):
    m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 20
flops = 2 * sum(counts) * N * K
print(f"perf: M={sum(counts)} E={E} N={N} K={K} {dt*1e3:.2f} ms "
      f"({flops/dt/1e12:.1f} TFLOP/s bf16)", flush=True)

print("RESULT:", "PASS" if not FAILS else f"FAIL {FAILS}")
sys.exit(0 if not FAILS else 1)
