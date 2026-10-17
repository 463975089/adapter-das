"""R-02 watchtower: DTK deepgemm paged_mqa_logits vs the emu gold standard.

KNOWN-RED sentinel (probed 2026-10-18, nmz26 GPU4): with a flat
[T, 132] kv (128 fp8 bytes + fp32 scale tail per token) the vendor op
RUNS but returns garbage (median rel 1.0, inf totals) — its fused_kv_cache
contract is the ORIGINAL preshuffle kpool page layout (page = 64 slots,
8192B of 16-tile-interleaved fp8 data + 256B of fp32 scales — the same
layout our kpool WRITE path produces and our kpool READ emu unpacks away
before the mqa emu sees it). Metadata gate facts pinned by the same probe:
q must be float8_e4m3fn (fnuz rejected), H in {8,16,32,64} (GLM-5.3
index_n_heads=32 fits), weights fp32 (N,H), kv uint8 (P,S,1,132)-shaped,
lens/bt int32.

So wiring the vendor tier properly means feeding it the RAW pool at the
dsa layer (bypassing the kpool read-emu unpack), not the converted tensor
this test fabricates. Until then this script asserts the KNOWN mismatch:
it FAILS loudly if the vendor suddenly matches the flat layout (meaning
the contract changed — re-evaluate the bridge), and documents the gap
otherwise.

Run (container, PYTHONPATH=/root/hcu_t2:/sglang-v0521/python):
HIP_VISIBLE_DEVICES=4 python tests/kernels/test_mqa_vendor_gfx938.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sgl_kernel._adapters.dsa_paged_mqa import fp8_paged_mqa_logits as emu  # noqa: E402
from sgl_kernel._adapters.mqa_vendor import (  # noqa: E402
    deepgemm_paged_mqa_logits as vendor,
)

FAILS = []


def ok(tag, cond, detail=""):
    print(f"{tag}: {'OK' if cond else 'FAIL ' + detail}", flush=True)
    if not cond:
        FAILS.append(tag)


def _make_case(dev, n_rows=4, ctx=200, pages=8, page_slots=64, seed=7):
    """fn-semantics fp8 (the is_cuda platform route quantizes with
    float8_e4m3fn — is_fp8_fnuz() is False there; the vendor op requires
    fn too, probed 2026-10-18)."""
    import sgl_kernel._adapters.dsa_paged_mqa as _dp

    _dp._fp8_view_dtype = lambda: torch.float8_e4m3fn  # emu reads fn bytes
    g = torch.Generator(device="cpu").manual_seed(seed)
    H, D = 32, 128  # GLM-5.3 index_n_heads=32, index_head_dim=128
    q = (torch.randn(n_rows, H, D, generator=g) * 0.3).to(dev)
    q_fp8 = q.to(torch.float8_e4m3fn).view(torch.uint8)
    total_toks = pages * page_slots
    k = (torch.randn(total_toks, D, generator=g) * 0.4).to(dev)
    k_fp8 = k.to(torch.float8_e4m3fn)
    scale = (torch.rand(total_toks, generator=g) * 0.5 + 0.5).to(dev)
    kv = torch.cat(
        [k_fp8.view(torch.uint8), scale.view(torch.uint8).reshape(-1, 4)],
        dim=1,
    )  # (T, 132): 128 fp8 bytes + fp32 scale tail (emu layout)
    kv4 = kv.reshape(pages, page_slots, 1, 132)
    w = torch.rand(n_rows, H, generator=g).to(dev) + 0.5
    lens = torch.full((n_rows, 1), ctx, dtype=torch.int32, device=dev)
    # page table: row i reads pages perm-i..; token t = page*64 + slot
    perm = torch.randperm(pages, generator=g)
    bt = torch.stack([perm.roll(-i)[: pages] for i in range(n_rows)]).to(dev)
    return q_fp8, kv4, w, lens, bt


def main():
    dev = "cuda"
    torch.cuda.init()
    q_fp8, kv4, w, lens, bt = _make_case(dev)
    common = dict(max_context_len=512, clean_logits=True)

    ref = emu(q_fp8, kv4, w, lens, bt, None, **common)
    got = vendor(q_fp8, kv4, w, lens, bt, None, **common)

    ctx = int(lens.reshape(-1)[0])
    r = ref[:, :ctx].float()
    v = got[:, :ctx].float()
    rel = ((v - r).abs() / r.abs().clamp_min(1e-3)).median().item()
    # KNOWN-RED sentinel: flat-layout kv feeds garbage through the vendor
    # (contract = preshuffle pool, see module docstring). rel MUST stay
    # high; a sudden match means the contract changed — revisit the bridge.
    ok("known-mismatch-persists(rel>0.1)", rel > 0.1, f"median rel={rel:.4f}")
    top_match = sum(
        set(r[i].topk(8).indices.tolist())
        == set(v[i].topk(8).indices.tolist())
        for i in range(r.shape[0])
    )
    ok("known-mismatch-topk(low)", top_match < r.shape[0],
       f"{top_match}/{r.shape[0]} rows agree — unexpected!")

    print(f"RESULT: {'PASS(known-red documented)' if not FAILS else 'FAIL ' + str(FAILS)} "
          f"(rel={rel:.4f}, topk {top_match}/{r.shape[0]})")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
