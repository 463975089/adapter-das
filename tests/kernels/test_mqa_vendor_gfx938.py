"""R-02 watchtower: DTK deepgemm paged_mqa_logits vs the emu gold standard.

Probe campaign 2026-10-18 (nmz26 GPU4, deterministic across repeats) pinned
two facts about the wheel op (deepgemm 2.1.0+dtk2604):

1. LAYOUT: fused_kv_cache (P,64,1,132) uint8 pages of 8448B = row-major k
   (slot s -> bytes [s*128, s*128+128)) + 64 fp32 scales at the page tail
   [8192,8448). Upstream DeepGEMM SM90 layout — and the layout the
   production HCU writer (fused_store_index_k_cache) emits, so the emu
   (page-level split since 2026-10-18) reads the same pool natively.
   Metadata gates: q float8_e4m3fn 4D [B,R,H,128], H in {8,16,32,64}
   (GLM-5.3 =32), weights fp32 (N,H), lens/bt int32.

2. WHEEL BUG: the kernel clamps NEGATIVE fp8 q AND k bytes to exactly 0.0
   (logits == sum_h w_h max(q_h,c,0) max(k_t,c,0) s_t, fitted to 4 decimals
   over 9 probe geometries). All-positive inputs are computed EXACTLY;
   real signed data comes out garbage (rel~3.2 vs gold).

So this script builds the pool in the RECOVERED layout and asserts:
  - positive-matches-emu: all-positive case matches the emu exactly —
    pins bridge + layout correctness (if this FAILS the wheel contract
    changed, e.g. a fixed or reworked kernel — re-evaluate the bridge);
  - signed-mismatch-persists: signed case still disagrees with the emu —
    the sign-clamp bug signature (if this FAILS the wheel got fixed:
    re-run the full parity gate and promote the tier, see mqa_vendor.py).

Run (container, PYTHONPATH=/root/hcu_t2:/sglang-v0521/python):
HIP_VISIBLE_DEVICES=4 python tests/kernels/test_mqa_vendor_gfx938.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hcu_adapter.dsa_paged_mqa import fp8_paged_mqa_logits as emu  # noqa: E402
from hcu_adapter.mqa_vendor import (  # noqa: E402
    deepgemm_paged_mqa_logits as vendor,
)

FAILS = []


def ok(tag, cond, detail=""):
    print(f"{tag}: {'OK' if cond else 'FAIL ' + detail}", flush=True)
    if not cond:
        FAILS.append(tag)


def _make_case(dev, signed, n_rows=4, pages=8, page_slots=64, seed=7):
    """Pool in the row-major k + fp32 tail-scale layout — the layout the
    production writer (fused_store_index_k_cache) emits AND the layout-
    fixed emu (2026-10-18 page-level split) reads, so ONE pool feeds both
    implementations under test.

    signed=False: |randn| values only (every fp8 byte positive) — the
    wheel bug is invisible and the op must match the emu exactly.
    signed=True: ordinary randn — the sign clamp must garble the output.
    """
    import hcu_adapter.dsa_paged_mqa as _dp

    _dp._fp8_view_dtype = lambda: torch.float8_e4m3fn  # emu reads fn bytes
    g = torch.Generator(device="cpu").manual_seed(seed)
    H, D = 32, 128
    src = torch.randn(n_rows, H, D, generator=g) * 0.3
    if not signed:
        src = src.abs()
    q = src.to(dev)
    q_fp8 = q.to(torch.float8_e4m3fn).view(torch.uint8)
    ksrc = torch.randn(pages * page_slots, D, generator=g) * 0.4
    if not signed:
        ksrc = ksrc.abs()
    k_fp8 = ksrc.to(dev).to(torch.float8_e4m3fn)
    scale = (torch.rand(pages * page_slots, generator=g) * 0.5 + 0.5).to(dev)

    # pool: row-major k + fp32 scale tail per page (vendor contract =
    # production fused_store layout)
    pool = torch.zeros(pages, page_slots * 132, dtype=torch.uint8, device=dev)
    pool[:, : page_slots * 128] = k_fp8.reshape(pages, -1)
    pool[:, page_slots * 128 :] = scale.view(torch.uint8).reshape(pages, page_slots, 4).reshape(pages, -1)
    kv4 = pool.view(pages, page_slots, 1, 132)

    w = (torch.rand(n_rows, H, generator=g) + 0.5).to(dev)
    lens = torch.tensor([[100], [180], [256], [200]], dtype=torch.int32, device=dev)[:n_rows]
    perm = torch.randperm(pages, generator=g)
    bt = torch.stack([perm.roll(-i)[:pages] for i in range(n_rows)]).to(dev)
    return q_fp8, kv4, w, lens, bt


def _rel_topk(ref, got, lens):
    r = ref[:, :256].float()
    v = got[:, :256].float()
    rel = ((v - r).abs() / r.abs().clamp_min(1e-3))
    meds = [float(rel[i, : int(lens[i, 0])].median()) for i in range(r.shape[0])]
    match = sum(
        set(r[i, : int(lens[i, 0])].topk(8).indices.tolist())
        == set(v[i, : int(lens[i, 0])].topk(8).indices.tolist())
        for i in range(r.shape[0])
    )
    return sum(meds) / len(meds), match, r.shape[0]


def main():
    dev = "cuda"
    torch.cuda.init()
    # the emu unpacks interleave under the default gate; this pool is the
    # vendor kernel's ROW-MAJOR+tail contract -> row-major arm
    os.environ["HCU_DSA_PRESHUFFLE"] = "0"
    common = dict(max_context_len=512, clean_logits=True)

    # arm 1: all-positive — vendor must match the emu EXACTLY (bug invisible)
    q, kv4, w, lens, bt = _make_case(dev, signed=False)
    ref = emu(q, kv4, w, lens, bt, None, **common)
    got = vendor(q, kv4, w, lens, bt, None, **common)
    rel, match, n = _rel_topk(ref, got, lens)
    ok("positive-matches-emu(rel<0.02)", rel < 0.02, f"median rel={rel:.4f}")
    ok("positive-topk(full)", match == n, f"{match}/{n}")

    # arm 2: signed — the sign-clamp bug must keep the output wrong
    q, kv4, w, lens, bt = _make_case(dev, signed=True)
    ref = emu(q, kv4, w, lens, bt, None, **common)
    got = vendor(q, kv4, w, lens, bt, None, **common)
    rel, match, n = _rel_topk(ref, got, lens)
    # nan counts as mismatch: the wheel's garbage output can contain NaNs
    ok("signed-mismatch-persists(rel>0.1)", rel > 0.1 or rel != rel,
       f"median rel={rel:.4f}")
    ok("signed-topk(low)", match < n, f"{match}/{n} rows agree — wheel fixed?!")

    print(f"RESULT: {'PASS(dual-fact watchtower)' if not FAILS else 'FAIL ' + str(FAILS)} "
          f"(pos rel see above, signed rel={rel:.4f})")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
