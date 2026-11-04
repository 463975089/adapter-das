# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""fa3 varlen prefill face — homegrown triton kernel vs fp32 hand math + emu.

sglang's fa3 backend routes ragged dense K/V (prefill extend, target
verify, the kvcache-facade varlen mode) through flash_attn_varlen_func:
packed q (M,H,D), packed k/v (N,Hk,D), cu_seqlens_q/k, per-sequence
bottom-right causal (extend windows Lq<Lk on a radix hit), optional swa
window and fp8-KV descale. The torch emu (python loop + host syncs) is
the serving gold standard; sgl_kernel.fa_varlen_triton is the graph-safe
tier-③ arm (HCU_FA_IMPL=triton).

Arms per case (three-stage scaffold per skills/hcu-adapter-dev):
  reference   fp32 hand math per sequence (authoritative), padded-free:
              s=q·kᵀ*scale, bottom-right keep mask, fp32 softmax, p@v.
  test A      triton vs hand ref: cos>=0.9999 AND maxrel<=5e-2 on o
              (bf16 family per test-suite.md; fp8-KV carries the e4m3
              grid: maxrel<=1e-1, still cos-gated), lse rel<=2e-3.
  test B      emu vs the same hand ref (contract cross-fix, tolerance as A).
  test C      triton vs emu direct pairwise (looser: 1e-2 maxrel).
  pits        zero-length q sequence (equal cu entries); zero-length k
              sequence (documented divergence: kernel writes zeros/lse=-inf,
              emu raises — pinned here on purpose).
  graph-safety capture -> replay x2 with fresh inputs; outputs must follow
              (grid static via max_seqlen_q, no host syncs in the wrapper).
  bench       python test_fa_varlen_triton.py adds timing vs emu (not
              collected by pytest; perf numbers are BW1101 — not transferable
              to gfx938 without a BW1102 rerun).

Run: pytest tests/kernels/test_fa_varlen_triton.py -v (HCU node)
Bench: HIP_VISIBLE_DEVICES=<id> python tests/kernels/test_fa_varlen_triton.py
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sgl_kernel.flash_attn import flash_attn_varlen_func as emu_varlen  # noqa: E402

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires HCU/CUDA device"
)


def _kernel():
    from sgl_kernel.fa_varlen_triton import flash_attn_varlen_triton

    return flash_attn_varlen_triton


# ---------------------------------------------------------------------------
# segment 2: fp32 hand reference (the authoritative one)
# ---------------------------------------------------------------------------


def _hand_ref(q, k, v, cu_q, cu_k, scale, causal, window,
              k_descale=None, v_descale=None, want_lse=False):
    """Per-sequence fp32 SDPA with the FA bottom-right frame.

    Local kv indices 0..Lk-1; row j (local q) sits at qpos = Lk-Lq+j;
    causal keep col<=qpos; FA window rel=col-qpos in [-left, right].
    Zero-Lk rows: o=0, lse=-inf (the kernel's documented divergence).
    """
    left, right = window
    outs, lses = [], []
    H, Hk = q.shape[-2], k.shape[-2]
    for b in range(cu_q.numel() - 1):
        Lq = int(cu_q[b + 1] - cu_q[b])
        Lk = int(cu_k[b + 1] - cu_k[b])
        if Lq == 0:
            continue
        qb = q[cu_q[b]:cu_q[b + 1]].float()
        kb = k[cu_k[b]:cu_k[b + 1]].float()
        vb = v[cu_k[b]:cu_k[b + 1]].float()
        if k_descale is not None:
            kb = kb * k_descale.float().reshape(1, Hk, 1)
        if v_descale is not None:
            vb = vb * v_descale.float().reshape(1, Hk, 1)
        # GQA grouping via grouped bmm (q heads H = Hk * G, same math as
        # the emu's _grouped_scores without the padding scaffolding):
        # q (Lq,H,D) -> (Hk,G,Lq,D); k/v (Lk,Hk,D) -> (Hk,D,Lk)/(Hk,Lk,D)
        D = qb.shape[-1]
        G = H // Hk
        qg = qb.view(Lq, Hk, G, D).permute(1, 2, 0, 3).reshape(
            Hk, G * Lq, D)
        s = torch.bmm(qg, kb.permute(1, 2, 0)) * scale     # (Hk, G*Lq, Lk)
        s = s.view(Hk, G, Lq, Lk)
        j = torch.arange(Lq, device=q.device)[:, None]
        col = torch.arange(Lk, device=q.device)[None, :]
        qpos = (Lk - Lq) + j
        if causal:
            keep = col <= qpos
        else:
            keep = torch.full((Lq, Lk), True, device=q.device)
        if left >= 0:
            keep = keep & ((col - qpos) >= -left)
        if right >= 0:
            keep = keep & ((col - qpos) <= right)
        s = s.masked_fill(~keep[None, None], float("-inf"))  # (Hk,G,Lq,Lk)
        p = torch.softmax(s, dim=-1).nan_to_num(0.0)
        vg = vb.permute(1, 0, 2)                              # (Hk,Lk,Dv)
        o = torch.matmul(p, vg.unsqueeze(1))                  # (Hk,G,Lq,Dv)
        o = o.permute(2, 0, 1, 3).reshape(Lq, H, v.shape[-1])
        o = o.contiguous().to(q.dtype)
        lse = torch.logsumexp(s, dim=-1).permute(2, 0, 1).reshape(
            Lq, H)                                            # (Lq, H)
        outs.append(o)
        lses.append(lse)
    if not outs:
        o = q.new_empty((0, H, v.shape[-1]), dtype=q.dtype)
        return (o, q.new_empty((0, H), dtype=torch.float32)) if want_lse else o
    o = torch.cat(outs, 0)
    lse = torch.cat(lses, 0)
    return (o, lse) if want_lse else o


def _metrics(o, ref):
    """cos + maxrel dual metric (layout bugs kill cos, noise does not)."""
    of, rf = o.float(), ref.float()
    cos = (of * rf).sum() / (of.norm() * rf.norm()).clamp_min(1e-12)
    denom = rf.abs().clamp_min(0.05)
    maxrel = ((of - rf).abs() / denom).max().item()
    maxabs = (of - rf).abs().max().item()
    return cos.item(), maxrel, maxabs


def _assert_face(name, o, ref, maxrel_tol):
    assert o.shape == ref.shape and o.dtype == ref.dtype, (
        f"{name}: shape/dtype {tuple(o.shape)}/{o.dtype} vs "
        f"{tuple(ref.shape)}/{ref.dtype}")
    cos, maxrel, maxabs = _metrics(o, ref)
    assert not torch.isnan(o).any(), f"{name}: NaN in output"
    assert cos >= 0.9999, f"{name}: cos={cos:.6f} < 0.9999 (maxabs={maxabs:.4f})"
    assert maxrel <= maxrel_tol, (
        f"{name}: maxrel={maxrel:.4f} > {maxrel_tol} (maxabs={maxabs:.4f})")


def _lse_check(name, lse, ref_lse):
    assert lse.shape == ref_lse.shape and lse.dtype == torch.float32
    fin = torch.isfinite(ref_lse)
    assert fin.any(), f"{name}: reference lse has no finite rows"
    rel = ((lse[fin] - ref_lse[fin]).abs()
           / ref_lse[fin].abs().clamp_min(1.0)).max().item()
    assert rel <= 2e-3, f"{name}: lse rel={rel:.4f} > 2e-3"


# ---------------------------------------------------------------------------
# segment 1: shape matrix — pure prefill / extend (radix hit, Lq<Lk) /
# odd lens / long ctx / swa / non-causal / fp8-KV / GQA ratios
# ---------------------------------------------------------------------------
CASES = [
    ("mha-eq", dict(seqs=[(17, 17), (33, 33)], H=16, Hk=16, D=128,
                    causal=True, window=(-1, -1))),
    ("gqa4-eq", dict(seqs=[(128, 128), (301, 301)], H=8, Hk=2, D=128,
                     causal=True, window=(-1, -1))),
    ("mha-extend", dict(seqs=[(64, 520), (1, 33), (129, 129), (5, 137)],
                        H=8, Hk=8, D=128, causal=True, window=(-1, -1))),
    ("mha-odd", dict(seqs=[(65, 65), (127, 127), (3, 3)], H=16, Hk=16,
                     D=128, causal=True, window=(-1, -1))),
    ("long-ctx", dict(seqs=[(512, 4096), (256, 2048)], H=8, Hk=4, D=128,
                      causal=True, window=(-1, -1))),
    ("swa-left", dict(seqs=[(97, 700), (200, 200)], H=8, Hk=4, D=128,
                      causal=True, window=(8, -1))),
    ("swa-both", dict(seqs=[(129, 400), (33, 33)], H=8, Hk=8, D=128,
                      causal=True, window=(16, 16))),
    ("noncausal", dict(seqs=[(32, 64), (9, 17)], H=8, Hk=8, D=128,
                       causal=False, window=(-1, -1))),
    ("gqa16-fp8kv", dict(seqs=[(300, 300), (77, 653)], H=32, Hk=8, D=128,
                         causal=True, window=(-1, -1), fp8_kv=True)),
    ("fp8kv-extend", dict(seqs=[(64, 520), (1, 33)], H=8, Hk=8, D=128,
                          causal=True, window=(-1, -1), fp8_kv=True)),
]


def _make_case(seqs, H, Hk, D, fp8_kv, seed):
    g = torch.Generator().manual_seed(seed)
    dev = "cuda"
    cu_q = torch.tensor([0] + list(
        torch.tensor(seqs)[:, 0].cumsum(0)), dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0] + list(
        torch.tensor(seqs)[:, 1].cumsum(0)), dtype=torch.int32, device=dev)
    M = int(cu_q[-1]); N = int(cu_k[-1])
    q = (torch.randn(M, H, D, generator=g) * 0.5).to(
        torch.bfloat16).to(dev)
    k = (torch.randn(N, Hk, D, generator=g) * 0.4).to(
        torch.bfloat16).to(dev)
    v = (torch.randn(N, Hk, D, generator=g) * 0.4).to(
        torch.bfloat16).to(dev)
    kd = vd = None
    if fp8_kv:
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)
        kd = (torch.rand(Hk, generator=g) * 0.5 + 0.75).float().to(dev)
        vd = (torch.rand(Hk, generator=g) * 0.5 + 0.75).float().to(dev)
    return q, k, v, cu_q, cu_k, kd, vd


def _seed_of(name):
    import zlib

    return zlib.crc32(name.encode()) % (2 ** 31)


@requires_gpu
@pytest.mark.parametrize("name,kw", CASES, ids=[c[0] for c in CASES])
def test_fa_varlen_triton_vs_hand(name, kw):
    """Arm A: triton vs fp32 hand ref (authoritative)."""
    flash_attn_varlen_triton = _kernel()
    seqs, H, Hk, D = kw["seqs"], kw["H"], kw["Hk"], kw["D"]
    torch.manual_seed(7)
    q, k, v, cu_q, cu_k, kd, vd = _make_case(
        seqs, H, Hk, D, kw.get("fp8_kv", False), seed=_seed_of(name))
    maxL = max(lq for lq, _ in seqs)
    got = flash_attn_varlen_triton(
        q, k, v, cu_q, cu_k, max_seqlen_q=maxL, causal=kw["causal"],
        window_size=kw["window"], k_descale=kd, v_descale=vd,
        return_softmax_lse=True)
    o_tri, lse_tri = got
    ref = _hand_ref(q, k, v, cu_q, cu_k, D ** -0.5, kw["causal"],
                    kw["window"], kd, vd, want_lse=True)
    o_ref, lse_ref = ref
    tol = 1e-1 if kw.get("fp8_kv") else 5e-2
    _assert_face(f"{name}/triton-vs-hand", o_tri, o_ref, tol)
    _lse_check(f"{name}/triton-vs-hand", lse_tri, lse_ref)


@requires_gpu
@pytest.mark.parametrize("name,kw", CASES, ids=[c[0] for c in CASES])
def test_fa_varlen_emu_vs_hand(name, kw):
    """Arm B: emu (gold) vs the same hand ref — contract cross-fix.

    The emu raises on any zero-Lk sequence; sweep-only via CASES (no
    zero-length entries in the numeric matrix)."""
    seqs, H, Hk, D = kw["seqs"], kw["H"], kw["Hk"], kw["D"]
    torch.manual_seed(7)
    q, k, v, cu_q, cu_k, kd, vd = _make_case(
        seqs, H, Hk, D, kw.get("fp8_kv", False), seed=_seed_of(name))
    maxL = max(lq for lq, _ in seqs)
    o_emu, lse_emu = emu_varlen(
        q, k, v, cu_q, cu_k, max_seqlen_q=maxL, max_seqlen_k=maxL * 8,
        causal=kw["causal"], window_size=kw["window"],
        k_descale=kd, v_descale=vd, return_softmax_lse=True)
    o_ref, lse_ref = _hand_ref(q, k, v, cu_q, cu_k, D ** -0.5,
                               kw["causal"], kw["window"], kd, vd,
                               want_lse=True)
    tol = 1e-1 if kw.get("fp8_kv") else 5e-2
    _assert_face(f"{name}/emu-vs-hand", o_emu, o_ref, tol)
    _lse_check(f"{name}/emu-vs-hand", lse_emu, lse_ref)


@requires_gpu
@pytest.mark.parametrize("name,kw", CASES, ids=[c[0] for c in CASES])
def test_fa_varlen_triton_vs_emu(name, kw):
    """Arm C: triton vs emu direct pairwise — the serving swap contract."""
    flash_attn_varlen_triton = _kernel()
    seqs, H, Hk, D = kw["seqs"], kw["H"], kw["Hk"], kw["D"]
    torch.manual_seed(7)
    q, k, v, cu_q, cu_k, kd, vd = _make_case(
        seqs, H, Hk, D, kw.get("fp8_kv", False), seed=_seed_of(name))
    maxL = max(lq for lq, _ in seqs)
    kw_common = dict(softmax_scale=None, causal=kw["causal"],
                     window_size=kw["window"],
                     k_descale=kd, v_descale=vd)
    o_tri, lse_tri = flash_attn_varlen_triton(
        q, k, v, cu_q, cu_k, max_seqlen_q=maxL,
        return_softmax_lse=True, **kw_common)
    o_emu, lse_emu = emu_varlen(
        q, k, v, cu_q, cu_k, max_seqlen_q=maxL,
        return_softmax_lse=True, **kw_common)
    tol = 1.5e-1 if kw.get("fp8_kv") else 1e-2
    assert o_tri.shape == o_emu.shape and o_tri.dtype == o_emu.dtype
    _, _, maxabs = _metrics(o_tri, o_emu)
    cos, _, _ = _metrics(o_tri, o_emu)
    assert cos >= 0.9999, f"{name}: cos(tri,emu)={cos:.6f}"
    assert maxabs <= 0.05, f"{name}: maxabs(tri,emu)={maxabs:.4f}"


# ---------------------------------------------------------------------------
# pits — one pit, one test
# ---------------------------------------------------------------------------


@requires_gpu
def test_fa_varlen_zero_len_q_seq():
    """Pit: a zero-length q window (equal cu_q entries) must be skipped
    silently — upstream chunked prefill emits empty extends; the kernel
    early-exits those programs, output token count stays M."""
    flash_attn_varlen_triton = _kernel()
    torch.manual_seed(3)
    dev = "cuda"
    cu_q = torch.tensor([0, 0, 24, 24, 58], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, 40, 180, 200, 400], dtype=torch.int32, device=dev)
    M, N, H, Hk, D = 58, 400, 8, 8, 128  # M = cu_q[-1] (0+24+0+34)
    q = (torch.randn(M, H, D) * 0.5).bfloat16().to(dev)
    k = (torch.randn(N, Hk, D) * 0.4).bfloat16().to(dev)
    v = (torch.randn(N, Hk, D) * 0.4).bfloat16().to(dev)
    o_tri = flash_attn_varlen_triton(q, k, v, cu_q, cu_k, max_seqlen_q=34)
    ref = _hand_ref(q, k, v, cu_q, cu_k, D ** -0.5, True, (-1, -1))
    _assert_face("zero-len-q", o_tri, ref, 5e-2)


@requires_gpu
def test_fa_varlen_zero_len_k_seq():
    """Pit: a zero-Lk sequence — the emu raises (torch chunk loop has no
    chunks); the kernel cannot raise under graph capture, it writes
    o=0 / lse=-inf for those rows instead. Pins the documented divergence
    so a serving flip between the two arms is at least diagnosable."""
    flash_attn_varlen_triton = _kernel()
    torch.manual_seed(4)
    dev = "cuda"
    cu_q = torch.tensor([0, 10, 30], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, 0, 128], dtype=torch.int32, device=dev)
    M, N, H, Hk, D = 30, 128, 8, 8, 128
    q = (torch.randn(M, H, D) * 0.5).bfloat16().to(dev)
    k = (torch.randn(N, Hk, D) * 0.4).bfloat16().to(dev)
    v = (torch.randn(N, Hk, D) * 0.4).bfloat16().to(dev)
    o_tri, lse_tri = flash_attn_varlen_triton(
        q, k, v, cu_q, cu_k, max_seqlen_q=20, return_softmax_lse=True)
    assert torch.all(o_tri[:10] == 0), "zero-Lk rows must be exactly 0"
    assert torch.all(torch.isinf(lse_tri[:10]) & (lse_tri[:10] < 0))
    # valid second sequence: local cu over the same full k pool
    ref = _hand_ref(q[10:30], k, v,
                    torch.tensor([0, 20], dtype=torch.int32, device=dev),
                    torch.tensor([0, 128], dtype=torch.int32, device=dev),
                    D ** -0.5, True, (-1, -1))
    _assert_face("zero-Lk-valid-part", o_tri[10:30], ref, 5e-2)
    # emu cross-check on the same input: the emu's per-sequence python
    # loop clamps total_tokens to >=1, so a zero-Lk sequence reads one
    # OUT-OF-SLICE token (its k slice is empty) — garbage rows cross-
    # contaminate its valid-part numerics vs the hand ref. That is an
    # emu-side edge bug on a face sglang never emits (windows always have
    # >=1 key); the kernel's o=0/lse=-inf is the honest form and is the
    # value already gated against the hand ref above. Log, don't gate.
    o_emu = emu_varlen(q, k, v, cu_q, cu_k, max_seqlen_q=20)
    zero_max = o_emu[:10].abs().max().item()
    print(f"zero-Lk pit: emu zero-row maxabs={zero_max:.4f} (garbage-tolerated "
          "emu edge; kernel rows exactly 0)", flush=True)


# ---------------------------------------------------------------------------
# graph-safety (dimension 2): capture -> replay x2, fresh inputs follow
# ---------------------------------------------------------------------------


@requires_gpu
def test_fa_varlen_capture_replay_follows_inputs():
    """Prefill graph buckets capture at a fixed (bs, maxL, M) and replay
    with new token content AND new per-seq lengths (within maxL). The
    kernel grid must be static (max_seqlen_q) and the wrapper must not
    sync on device tensors at capture time."""
    flash_attn_varlen_triton = _kernel()
    torch.manual_seed(9)
    dev = "cuda"
    B, maxL, H, Hk, D = 3, 64, 8, 8, 128
    M = B * maxL  # capture shape: every window exactly maxL wide
    q0 = (torch.randn(M, H, D) * 0.5).bfloat16().to(dev)
    N = B * 512
    k0 = (torch.randn(N, Hk, D) * 0.4).bfloat16().to(dev)
    v0 = (torch.randn(N, Hk, D) * 0.4).bfloat16().to(dev)

    def pack(lq_list, N_per):
        cu_q = torch.tensor([0] + list(torch.tensor(lq_list).cumsum(0)),
                            dtype=torch.int32, device=dev)
        cu_k = torch.tensor([0] + list(torch.tensor(N_per).cumsum(0)),
                            dtype=torch.int32, device=dev)
        return cu_q, cu_k

    cu_q0, cu_k0 = pack([64, 64, 64], [512, 512, 512])
    q = q0.clone(); k = k0.clone(); v = v0.clone()
    args = (q, k, v, cu_q0, cu_k0)
    out = flash_attn_varlen_triton(*args, max_seqlen_q=maxL)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        flash_attn_varlen_triton(*args, max_seqlen_q=maxL)  # warmup/jit
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        out_c = flash_attn_varlen_triton(*args, max_seqlen_q=maxL)

    # replay 1: same shapes, new content, ragged lengths inside maxL
    lq1 = [10, 33, 64]
    n1 = [300, 700, 512]
    cu_q1, cu_k1 = pack(lq1, n1)
    q.copy_(q0); q[:int(cu_q1[-1])] = (
        torch.randn(int(cu_q1[-1]), H, D) * 0.5).bfloat16().to(dev)
    k.copy_(k0); k[:int(cu_k1[-1])] = (
        torch.randn(int(cu_k1[-1]), Hk, D) * 0.4).bfloat16().to(dev)
    v.copy_(v0); v[:int(cu_k1[-1])] = (
        torch.randn(int(cu_k1[-1]), Hk, D) * 0.4).bfloat16().to(dev)
    n_tok1 = int(cu_q1[-1])
    cu_q0.copy_(cu_q1); cu_k0.copy_(cu_k1)
    g.replay()
    ref1 = _hand_ref(q, k, v, cu_q0, cu_k0, D ** -0.5, True, (-1, -1))
    _assert_face("replay-1", out_c[:n_tok1], ref1, 5e-2)

    # replay 2: different lengths again — a baked mask would fail here
    # (per-seq kv lens vary; total stays under the N=B*512 pool)
    cu_q2, cu_k2 = pack([64, 64, 64], [64, 448, 384])
    q.copy_(q0); q[:M] = (torch.randn(M, H, D) * 0.5).bfloat16().to(dev)
    k.copy_(k0); k[:int(cu_k2[-1])] = (
        torch.randn(int(cu_k2[-1]), Hk, D) * 0.4).bfloat16().to(dev)
    v.copy_(v0); v[:int(cu_k2[-1])] = (
        torch.randn(int(cu_k2[-1]), Hk, D) * 0.4).bfloat16().to(dev)
    n_tok2 = int(cu_q2[-1])
    cu_q0.copy_(cu_q2); cu_k0.copy_(cu_k2)
    g.replay()
    ref2 = _hand_ref(q, k, v, cu_q0, cu_k0, D ** -0.5, True, (-1, -1))
    _assert_face("replay-2", out_c[:n_tok2], ref2, 5e-2)


# ---------------------------------------------------------------------------
# bench (not collected): python tests/kernels/test_fa_varlen_triton.py
# ---------------------------------------------------------------------------


def _bench():
    flash_attn_varlen_triton = _kernel()

    def timeit(fn, iters=20):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
        t0.record()
        for _ in range(iters):
            fn()
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / iters  # ms

    print(f"{'case':<16}{'emu ms':>10}{'triton ms':>11}{'speedup':>9}")
    for name, kw in CASES:
        torch.manual_seed(7)
        seqs, H, Hk, D = kw["seqs"], kw["H"], kw["Hk"], kw["D"]
        q, k, v, cu_q, cu_k, kd, vd = _make_case(
            seqs, H, Hk, D, kw.get("fp8_kv", False), seed=1)
        maxL = max(lq for lq, _ in seqs)
        kw_common = dict(softmax_scale=None, causal=kw["causal"],
                         window_size=kw["window"])
        t_emu = timeit(lambda: emu_varlen(q, k, v, cu_q, cu_k,
                                          max_seqlen_q=maxL, **kw_common))
        t_tri = timeit(lambda: flash_attn_varlen_triton(
            q, k, v, cu_q, cu_k, max_seqlen_q=maxL, **kw_common))
        print(f"{name:<16}{t_emu:>10.3f}{t_tri:>11.3f}"
              f"{t_emu / max(t_tri, 1e-9):>8.1f}x  "
              "[BW1101 — 待 BW1102 复核]")

    # serving-realistic prefill bucket: bs=8 x 4K ctx GQA
    torch.manual_seed(1)
    dev = "cuda"
    B, Lk, Lq, H, Hk, D = 8, 4096, 512, 32, 8, 128
    cu_k = (torch.arange(B + 1, device=dev, dtype=torch.int32) * Lk)
    cu_q = (torch.arange(B + 1, device=dev, dtype=torch.int32) * Lq)
    q = (torch.randn(B * Lq, H, D) * 0.5).bfloat16().to(dev)
    k = (torch.randn(B * Lk, Hk, D) * 0.4).bfloat16().to(dev)
    v = (torch.randn(B * Lk, Hk, D) * 0.4).bfloat16().to(dev)
    t_emu = timeit(lambda: emu_varlen(q, k, v, cu_q, cu_k,
                                      max_seqlen_q=Lq, softmax_scale=None,
                                      causal=True, window_size=(-1, -1)))
    t_tri = timeit(lambda: flash_attn_varlen_triton(
        q, k, v, cu_q, cu_k, max_seqlen_q=Lq, softmax_scale=None,
        causal=True, window_size=(-1, -1)))
    print(f"{'bs8x4kx512':<16}{t_emu:>10.3f}{t_tri:>11.3f}"
          f"{t_emu / max(t_tri, 1e-9):>8.1f}x  [BW1101 — 待 BW1102 复核]")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("requires HCU/CUDA device")
    pytest.main([__file__, "-v", "-x"])
    print("\n=== bench vs emu (BW1101, 不迁移 gfx938 结论) ===")
    _bench()
