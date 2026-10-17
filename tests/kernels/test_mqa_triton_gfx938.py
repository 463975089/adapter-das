"""R-02 parity gate: homegrown triton paged-MQA logits + the emu layout fix.

The pool is written by the REAL production writer in arm 1 —
fused_store_index_k_cache (the _is_cuda store arm; can_use_dsa_fused_store()
is True on HCU) — which quantizes a bf16 key and lays each page out as
k row-major [0, PS*128) + PS fp32 scales at the tail. Golden = direct
math on the pool decoded per that layout. Arms:

  1. production-writer pool: triton AND the layout-fixed torch emu must
     both match golden (rel < 0.02, full top-k). The emu's pre-2026-10-18
     row-level 132-byte split misread this pool (decode DSA top-k was
     deterministically wrong in production) — this arm pins the fix.
  2. SetKAndS non-preshuffle pool: same layout via the other writer.
  3. SetKAndS aiter-preshuffle pool (16-tile interleave): triton's
     opt-in arm (HCU_MQA_TRITON_PRESHUFFLE=1) must match direct math on
     the source values.
  4. dispatch smoke + decode-shape timing evidence.

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

from sgl_kernel._adapters.dsa_paged_mqa import fp8_paged_mqa_logits as emu  # noqa: E402
from sgl_kernel._adapters.mqa_triton import triton_paged_mqa_logits as tri  # noqa: E402

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


def _decode_pool(buf, PS):
    """Golden view: k row-major + fp32 tail scales, decoded to float."""
    k = buf[:, : PS * 128].contiguous().view(torch.float8_e4m3fn).float()
    s = buf[:, PS * 128 :].contiguous().view(torch.float32).reshape(-1)
    return k.reshape(-1, 128), s


def _make_case(dev, n_rows=4, pages=8, PS=64, seed=7):
    g = torch.Generator(device="cpu").manual_seed(seed)
    H, D = 32, 128
    T = pages * PS
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
    import sgl_kernel._adapters.dsa_paged_mqa as _dp

    _dp._fp8_view_dtype = lambda: torch.float8_e4m3fn  # emu reads fn bytes

    dev = "cuda"
    torch.cuda.init()
    for k in ("HCU_MQA_TRITON_PRESHUFFLE", "HCU_MQA_LOGITS_IMPL"):
        os.environ.pop(k, None)
    common = dict(max_context_len=512, clean_logits=True)
    g = torch.Generator(device="cpu").manual_seed(11)
    pages, PS = 8, 64
    T = pages * PS
    key = (torch.randn(T, 128, generator=g) * 0.4).to(dev).to(torch.bfloat16)

    q_fp8, w, lens, bt = _make_case(dev)

    # arm 1: PRODUCTION writer (fused quantize+store)
    buf = torch.zeros(pages, PS * 132, dtype=torch.uint8, device=dev)
    fused_store_index_k_cache(
        key, buf, torch.arange(T, device=dev, dtype=torch.int64), PS
    )
    torch.cuda.synchronize()
    kv4 = buf.view(pages, PS, 1, 132)
    k_dec, s_dec = _decode_pool(buf, PS)
    gold = _golden(q_fp8, k_dec, s_dec, w, lens, bt, pages, PS, 512, dev)

    got = tri(q_fp8.view(torch.uint8), kv4, w, lens, bt, None, **common)
    rel, match, n = _rel_topk(gold, got, lens)
    ok("triton-vs-golden(rel<0.02)", rel < 0.02, f"median rel={rel:.5f}")
    ok("triton-topk(full)", match == n, f"{match}/{n}")

    got_emu = emu(q_fp8.view(torch.uint8), kv4, w, lens, bt, None, **common)
    rel_e, match_e, _ = _rel_topk(gold, got_emu, lens)
    ok("emu-layout-fix(rel<0.02)", rel_e < 0.02, f"median rel={rel_e:.5f}")
    ok("emu-topk(full)", match_e == n, f"{match_e}/{n}")

    # arm 2: SetKAndS non-preshuffle writer (same layout, other writer)
    k_fp8 = key.to(torch.float8_e4m3fn)
    k_scale = (torch.rand(T, generator=g) * 0.5 + 0.5).to(dev)
    buf_rm = _write_setk(dev, k_fp8, k_scale, pages, PS, tile=0)
    gold_rm = _golden(q_fp8, k_fp8.float(), k_scale, w, lens, bt,
                      pages, PS, 512, dev)
    got_rm = tri(q_fp8.view(torch.uint8), buf_rm.view(pages, PS, 1, 132),
                 w, lens, bt, None, **common)
    rel_r, match_r, _ = _rel_topk(gold_rm, got_rm, lens)
    ok("setk-rowmajor(rel<0.02)", rel_r < 0.02, f"median rel={rel_r:.5f}")
    ok("setk-rowmajor-topk(full)", match_r == n, f"{match_r}/{n}")

    # arm 3: SetKAndS aiter-preshuffle (16-tile interleave), opt-in arm
    buf_il = _write_setk(dev, k_fp8, k_scale, pages, PS, tile=16)
    os.environ["HCU_MQA_TRITON_PRESHUFFLE"] = "1"
    try:
        got_il = tri(q_fp8.view(torch.uint8), buf_il.view(pages, PS, 1, 132),
                     w, lens, bt, None, **common)
    finally:
        os.environ.pop("HCU_MQA_TRITON_PRESHUFFLE", None)
    rel_i, match_i, _ = _rel_topk(gold_rm, got_il, lens)
    ok("setk-interleave-arm(rel<0.02)", rel_i < 0.02, f"median rel={rel_i:.5f}")
    ok("setk-interleave-topk(full)", match_i == n, f"{match_i}/{n}")

    # arm 4: dispatch smoke — HCU_MQA_LOGITS_IMPL=triton routes here
    os.environ["HCU_MQA_LOGITS_IMPL"] = "triton"
    try:
        got_d = emu(q_fp8.view(torch.uint8), kv4, w, lens, bt, None, **common)
    finally:
        os.environ.pop("HCU_MQA_LOGITS_IMPL", None)
    ok("dispatch-triton(routes)", torch.allclose(got, got_d),
       "entry did not route to triton tier")

    # arm 5: timing evidence (decode shape, full 64K window, ctx ~2K)
    pagesB, N = 1024, 8
    keyB = (torch.randn(pagesB * PS, 128, generator=g) * 0.4).to(dev).to(torch.bfloat16)
    bufB = torch.zeros(pagesB, PS * 132, dtype=torch.uint8, device=dev)
    fused_store_index_k_cache(
        keyB, bufB, torch.arange(pagesB * PS, device=dev, dtype=torch.int64), PS
    )
    torch.cuda.synchronize()
    qB = (torch.randn(N, 32, 128, generator=g) * 0.3).to(dev).to(torch.float8_e4m3fn)
    wB = (torch.rand(N, 32, generator=g) + 0.5).to(dev)
    lensB = torch.full((N, 1), 2048, dtype=torch.int32, device=dev)
    btB = torch.arange(pagesB, device=dev).unsqueeze(0).repeat(N, 1)
    kvB = bufB.view(pagesB, PS, 1, 132)
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
