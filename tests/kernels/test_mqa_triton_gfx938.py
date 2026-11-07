# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""R-02 parity gate: homegrown triton paged-MQA logits + the emu layout fix.

Pool layout census (2026-10-18, validation-host-26 GPU4): GLM-5.3 (IndexerKPool,
kpool=4) writes its index-K pool through the kpool writer family and
SetKAndS under the forced preshuffle gate — every page = k in the 16-tile
INTERLEAVE order + PS fp32 scales at the tail. The prefill gather reads
the same order (consistent); only the decode paged-MQA reader was wrong
(the emu's old row-level 132-byte split). Arms:

  1. PRODUCTION layout (SetKAndS interleave writer, gate on): triton AND
     the interleave-fixed torch emu must both match direct math on the
     source values (rel < 0.02, full top-k). Pins the decode fix.
  2. legacy row-major world (HCU_DSA_PRESHUFFLE=0): SetKAndS tile=0 and
     fused_store writers; triton row-major arm + emu row-major unpack.
  3. dispatch smoke + decode-shape timing evidence.

Run (container, PYTHONPATH=/root/hcu_t2:/sglang-v0521/python):
HIP_VISIBLE_DEVICES=4 python tests/kernels/test_mqa_triton_gfx938.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sglang.kernels.ops.attention.dsa import index_buf_accessor as iba  # noqa: E402
from sglang.kernels.ops.attention.fused_store_index_cache import (  # noqa: E402
    fused_store_index_k_cache,
)

from hcu_adapter.dsa_mqa_logits_emu import fp8_paged_mqa_logits as emu  # noqa: E402
from hcu_adapter.mqa_logits_triton import triton_paged_mqa_logits as tri  # noqa: E402

FAILS = []


def ok(tag, cond, detail=""):
    print(f"{tag}: {'OK' if cond else 'FAIL ' + detail}", flush=True)
    if not cond:
        FAILS.append(tag)


def _write_setk(dev, k_fp8, k_scale, pages, PS, tile):
    T = pages * PS
    buf = torch.zeros(pages, PS * 132, dtype=torch.uint8, device=dev)
    iba._set_k_and_s_triton_kernel[(T,)](
        buf.view(torch.float8_e4m3fn),
        buf.view(torch.float32),
        torch.arange(T, device=dev, dtype=torch.int32),
        k_fp8,
        k_scale,
        k_fp8.stride(0),
        PAGE_SIZE=PS,
        BUF_NUMEL_PER_PAGE=PS * 132,
        NUM_K_ELEMS_PER_TOKEN=128,
        S_OFFSET_NBYTES_IN_PAGE=PS * 128,
        PRESHUFFLE_TILE=tile,
    )
    torch.cuda.synchronize()
    return buf


def _decode_pool_rowmajor(buf, PS):
    """Golden view for row-major pools: k rows + fp32 tail scales."""
    k = buf[:, : PS * 128].contiguous().view(torch.float8_e4m3fn).float()
    s = buf[:, PS * 128 :].contiguous().view(torch.float32).reshape(-1)
    return k.reshape(-1, 128), s


def _make_case(dev, n_rows=4, pages=8, PS=64, seed=7):
    g = torch.Generator(device="cpu").manual_seed(seed)
    H, D = 32, 128
    q = (torch.randn(n_rows, H, D, generator=g) * 0.3).to(dev)
    q_fp8 = q.to(torch.float8_e4m3fn)
    w = (torch.rand(n_rows, H, generator=g) + 0.5).to(dev)
    lens = torch.tensor(
        [[100], [180], [256], [507]], dtype=torch.int32, device=dev
    )[:n_rows]
    perm = torch.randperm(pages, generator=g)
    bt = torch.stack([perm.roll(-i)[:pages] for i in range(n_rows)]).to(dev)
    return q_fp8, w, lens, bt


def _golden(q_fp8, k_dec, s_dec, w, lens, bt, pages, PS, width, dev):
    n = q_fp8.shape[0]
    qdec = q_fp8.float()
    out = torch.zeros(n, width)
    for i in range(n):
        L = int(lens[i, 0])
        for p in range(pages):
            if p * PS >= L:
                break
            for s in range(PS):
                col = p * PS + s
                if col >= L:
                    break
                t = int(bt[i, p]) * PS + s
                dots = (qdec[i] * k_dec[t].unsqueeze(0)).sum(-1)
                out[i, col] = float(dots @ w[i].float()) * float(s_dec[t])
    return out.to(dev)


def _rel_topk(ref, got, lens):
    r = ref[:, :512].float()
    v = got[:, :512].float()
    d = (v - r).abs() / r.abs().clamp_min(1e-3)
    meds = [float(d[i, : int(lens[i, 0])].median()) for i in range(r.shape[0])]
    match = sum(
        set(r[i, : int(lens[i, 0])].topk(8).indices.tolist())
        == set(v[i, : int(lens[i, 0])].topk(8).indices.tolist())
        for i in range(r.shape[0])
    )
    return sum(meds) / len(meds), match, r.shape[0]


def main():
    import hcu_adapter.dsa_mqa_logits_emu as _dp

    _dp._fp8_view_dtype = lambda: torch.float8_e4m3fn  # emu reads fn bytes

    dev = "cuda"
    torch.cuda.init()
    os.environ.pop("HCU_DSA_PRESHUFFLE", None)  # default arm: gate ON
    common = dict(max_context_len=512, clean_logits=True)
    g = torch.Generator(device="cpu").manual_seed(11)
    pages, PS = 8, 64
    T = pages * PS
    k_fp8 = (torch.randn(T, 128, generator=g) * 0.4).to(dev).to(torch.float8_e4m3fn)
    k_scale = (torch.rand(T, generator=g) * 0.5 + 0.5).to(dev)
    q_fp8, w, lens, bt = _make_case(dev)
    gold_src = _golden(q_fp8, k_fp8.float(), k_scale, w, lens, bt,
                       pages, PS, 512, dev)

    # arm 1: PRODUCTION layout — SetKAndS interleave writer, gate on
    buf_il = _write_setk(dev, k_fp8, k_scale, pages, PS, tile=16)
    kv_il = buf_il.view(pages, PS, 1, 132)
    got = tri(q_fp8.view(torch.uint8), kv_il, w, lens, bt, None, **common)
    rel, match, n = _rel_topk(gold_src, got, lens)
    ok("triton-interleave(rel<0.02)", rel < 0.02, f"median rel={rel:.5f}")
    ok("triton-interleave-topk(full)", match == n, f"{match}/{n}")

    got_emu = emu(q_fp8.view(torch.uint8), kv_il, w, lens, bt, None, **common)
    rel_e, match_e, _ = _rel_topk(gold_src, got_emu, lens)
    ok("emu-interleave-fix(rel<0.02)", rel_e < 0.02, f"median rel={rel_e:.5f}")
    ok("emu-interleave-topk(full)", match_e == n, f"{match_e}/{n}")

    # arm 2: legacy row-major world — gate off, SetKAndS tile=0 + emu
    os.environ["HCU_DSA_PRESHUFFLE"] = "0"
    try:
        buf_rm = _write_setk(dev, k_fp8, k_scale, pages, PS, tile=0)
        kv_rm = buf_rm.view(pages, PS, 1, 132)
        got_rm = tri(q_fp8.view(torch.uint8), kv_rm, w, lens, bt, None,
                     **common)
        rel_r, match_r, _ = _rel_topk(gold_src, got_rm, lens)
        ok("triton-rowmajor(rel<0.02)", rel_r < 0.02, f"median rel={rel_r:.5f}")
        ok("triton-rowmajor-topk(full)", match_r == n, f"{match_r}/{n}")
        got_emu_rm = emu(q_fp8.view(torch.uint8), kv_rm, w, lens, bt, None,
                         **common)
        rel_re, match_re, _ = _rel_topk(gold_src, got_emu_rm, lens)
        ok("emu-rowmajor-arm(rel<0.02)", rel_re < 0.02,
           f"median rel={rel_re:.5f}")

        # arm 2b: fused_store writer (kpool=1 world), still row-major
        key = (torch.randn(T, 128, generator=g) * 0.4).to(dev).to(torch.bfloat16)
        buf_fs = torch.zeros(pages, PS * 132, dtype=torch.uint8, device=dev)
        fused_store_index_k_cache(
            key, buf_fs, torch.arange(T, device=dev, dtype=torch.int64), PS
        )
        torch.cuda.synchronize()
        k_dec, s_dec = _decode_pool_rowmajor(buf_fs, PS)
        gold_fs = _golden(q_fp8, k_dec, s_dec, w, lens, bt, pages, PS, 512, dev)
        got_fs = tri(q_fp8.view(torch.uint8), buf_fs.view(pages, PS, 1, 132),
                     w, lens, bt, None, **common)
        rel_f, match_f, _ = _rel_topk(gold_fs, got_fs, lens)
        ok("triton-fusedstore-pool(rel<0.02)", rel_f < 0.02,
           f"median rel={rel_f:.5f}")
        ok("triton-fusedstore-topk(full)", match_f == n, f"{match_f}/{n}")
    finally:
        os.environ.pop("HCU_DSA_PRESHUFFLE", None)

    # arm 3: dispatch smoke — HCU_MQA_LOGITS_IMPL=triton routes here
    os.environ["HCU_MQA_LOGITS_IMPL"] = "triton"
    try:
        got_d = emu(q_fp8.view(torch.uint8), kv_il, w, lens, bt, None, **common)
    finally:
        os.environ.pop("HCU_MQA_LOGITS_IMPL", None)
    ok("dispatch-triton(routes)", torch.allclose(got, got_d),
       "entry did not route to triton tier")

    # arm 4: timing evidence (production layout, full 64K window, ctx ~2K)
    pagesB, N = 1024, 8
    kB = (torch.randn(pagesB * PS, 128, generator=g) * 0.4).to(dev).to(torch.float8_e4m3fn)
    sB = (torch.rand(pagesB * PS, generator=g) * 0.5 + 0.5).to(dev)
    bufB = _write_setk(dev, kB, sB, pagesB, PS, tile=16)
    kvB = bufB.view(pagesB, PS, 1, 132)
    qB = (torch.randn(N, 32, 128, generator=g) * 0.3).to(dev).to(torch.float8_e4m3fn)
    wB = (torch.rand(N, 32, generator=g) + 0.5).to(dev)
    lensB = torch.full((N, 1), 2048, dtype=torch.int32, device=dev)
    btB = torch.arange(pagesB, device=dev).unsqueeze(0).repeat(N, 1)
    big = dict(max_context_len=pagesB * PS, clean_logits=True)

    def _time(fn, iters=5):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / iters

    t_tri = _time(lambda: tri(qB.view(torch.uint8), kvB, wB, lensB, btB, None, **big))
    t_emu = _time(lambda: emu(qB.view(torch.uint8), kvB, wB, lensB, btB, None, **big))
    print(f"timing decode-shape N={N} window={pagesB*PS} ctx=2048: "
          f"triton={t_tri:.2f}ms emu(vec)={t_emu:.2f}ms "
          f"({t_emu / t_tri:.1f}x)", flush=True)

    print(f"RESULT: {'PASS' if not FAILS else 'FAIL ' + str(FAILS)} "
          f"(triton rel={rel:.5f}, fixed-emu rel={rel_e:.5f})")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
