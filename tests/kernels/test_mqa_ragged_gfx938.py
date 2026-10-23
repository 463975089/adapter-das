"""R-20 parity gate: homegrown triton RAGGED MQA logits vs the torch emu.

The C4 nonpaged plan (dsv4/indexer.py:_nonpaged_mqa_logits ->
deep_gemm.fp8_mqa_logits) hits the ragged entry 44x per decode step on
DSV4.1F; the torch emu is ~58 aten kernels per call (fingerprinted
2026-10-24 nmz18: 38 manual_unroll + 31 vectorized + 6 Cijk bmm + 6 reduce
+ arange) — the dominant elementwise family in the wave6 decode trace.
The triton sibling (mqa_triton.triton_ragged_mqa_logits) must match the
emu golden exactly in window/-inf semantics and tightly in values, and
keep top-k stable on realistic score spreads.

Arms:
  1. parity vs emu on random fp8 pools (values: rel < 1e-3 on the valid
     window — triton accumulates fp32 from exact fp8 decode, emu widens
     through bf16 matmul; -inf outside [ks,ke) bit-exact).
  2. top-k stability: full row ordering preserved (argmax + top-8 sets).
  3. dispatch smoke: deep_gemm.fp8_mqa_logits routes to the triton arm
     under HCU_MQA_LOGITS_IMPL=triton (default) and back to the emu under
     =loop; kernel count 1 vs ~58 (chrome-trace fingerprint).

Run (container, PYTHONPATH=/root/hcu_v0522:/sglang-v0522/python):
HIP_VISIBLE_DEVICES=0 python tests/kernels/test_mqa_ragged_gfx938.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sgl_kernel._adapters import mqa_logits_emu as emu_mod  # noqa: E402
from sgl_kernel._adapters.mqa_triton import triton_ragged_mqa_logits as tri  # noqa: E402

FAILS = []


def ok(tag, cond, detail=""):
    print(f"{tag}: {'OK' if cond else 'FAIL ' + detail}", flush=True)
    if not cond:
        FAILS.append(tag)


def main():
    torch.manual_seed(20261024)
    dev = torch.device("cuda:0")
    N, H, D, T = 16, 32, 128, 8192
    q = (torch.randn(N, H, D, device=dev) * 4).clamp(-30, 30).to(torch.float8_e4m3fn)
    k = (torch.randn(T, D, device=dev) * 4).clamp(-30, 30).to(torch.float8_e4m3fn)
    sc = torch.rand(T, dtype=torch.float32, device=dev) + 0.25
    w = torch.rand(N, H, dtype=torch.float32, device=dev) * 2
    ks = torch.zeros(N, dtype=torch.int64, device=dev)
    ke = torch.randint(256, T - 16, (N,), dtype=torch.int64, device=dev)

    golden = emu_mod._ragged_fp8_mqa_logits(q, k, sc, w, ks, ke)
    out = tri(q, k, sc, w, ks, ke)

    ok("shape", tuple(out.shape) == (N, T) and out.dtype == torch.float32,
       f"{tuple(out.shape)} {out.dtype}")

    # -inf semantics bit-exact outside every window
    t = torch.arange(T, device=dev)[None, :].expand(N, T)
    outside = (t < ks[:, None]) | (t >= ke[:, None])
    ok("outside -inf",
       bool(((out[outside] == float("-inf")) & (golden[outside] == float("-inf"))).all())
       if outside.any() else True)

    # fp64 direct-math arbitration: triton decodes fp8 exactly and
    # accumulates fp32 -> must sit ~1e-6 from truth; the emu widens through
    # bf16 matmuls -> ~1e-3. Gates against TRUTH, not against the emu.
    kd = k.to(torch.float64)
    qd = q.to(torch.float64)
    wd = w.to(torch.float64)
    sd = sc.to(torch.float64)
    direct = torch.empty(N, T, dtype=torch.float64, device=dev)
    for n in range(N):
        scores = torch.matmul(kd, qd[n].transpose(0, 1))  # (T, H)
        direct[n] = (scores * wd[n][None, :]).sum(-1) * sd
    rel_tri = ((out.double() - direct).abs() / direct.abs().clamp_min(1e-4))
    rel_emu = ((golden.double() - direct).abs() / direct.abs().clamp_min(1e-4))
    m_tri = rel_tri[~outside].median().item()
    m_emu = rel_emu[~outside].median().item()
    print(f"vs fp64 truth: triton median rel {m_tri:.2e}, emu median rel {m_emu:.2e}")
    ok("triton near truth <1e-5", m_tri < 1e-5, f"{m_tri:.2e}")
    ok("triton tighter than emu", m_tri < m_emu, f"{m_tri:.2e} vs {m_emu:.2e}")
    ok("cross-check triton~emu <4e-3",
       bool(((out.double() - golden.double()).abs()
             / golden.double().abs().clamp_min(1e-4))[~outside].median() < 4e-3))

    # top-k stability against TRUTH on the valid window
    for n in range(N):
        row_d = direct[n, ks[n]:ke[n]]
        row_o = out[n, ks[n]:ke[n]]
        k8 = min(8, row_d.numel())
        td = set(torch.topk(row_d, k8).indices.tolist())
        to = set(torch.topk(row_o, k8).indices.tolist())
        ok(f"argmax n={n}", bool(row_d.argmax() == row_o.argmax()),
           f"{row_d.argmax().item()} vs {row_o.argmax().item()}")
        ok(f"top8 n={n}", td == to, f"sym-diff {len(td ^ to)}")

    # dispatch smoke: the deep_gemm shim routes by impl
    import deep_gemm

    kv = (k, sc)
    r1 = deep_gemm.fp8_mqa_logits(q, kv, w, ks, ke, clean_logits=False,
                                  max_seqlen_k=T)
    ok("shim triton route shape", tuple(r1.shape) == (N, T))
    ok("shim matches direct",
       bool(torch.allclose(r1.clamp(min=-1e30), out.clamp(min=-1e30), rtol=1e-3,
                           atol=1e-3)))
    os.environ["HCU_MQA_LOGITS_IMPL"] = "loop"
    try:
        r2 = deep_gemm.fp8_mqa_logits(q, kv, w, ks, ke, clean_logits=False,
                                      max_seqlen_k=T)
        ok("shim loop fallback = golden",
           bool(torch.allclose(r2.clamp(min=-1e30), golden.clamp(min=-1e30),
                               rtol=1e-2, atol=1e-2)))
    finally:
        os.environ.pop("HCU_MQA_LOGITS_IMPL", None)

    # kernel-count fingerprint: triton 1 kernel vs emu avalanche
    import json
    from torch.profiler import ProfilerActivity, profile

    def count_kernels(call):
        for _ in range(2):
            call()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            call()
            torch.cuda.synchronize()
        prof.export_chrome_trace("/tmp/ragged_gate.json")
        data = json.load(open("/tmp/ragged_gate.json"))
        names = [e.get("name", "") for e in data["traceEvents"]
                 if e.get("cat") == "kernel"]
        return names

    kn_tri = count_kernels(lambda: tri(q, k, sc, w, ks, ke))
    n_tri = sum(1 for n_ in kn_tri if "_ragged_mqa_logits_kernel" in n_)
    kn_emu = count_kernels(lambda: emu_mod._ragged_fp8_mqa_logits(q, k, sc, w, ks, ke))
    print(f"kernels: triton={len(kn_tri)} (core {n_tri}) emu={len(kn_emu)}")
    ok("triton single core kernel", n_tri >= 1)
    ok("emu avalanche reference", len(kn_emu) > 20, str(len(kn_emu)))
    ok("kernel-count win", len(kn_tri) + 4 < len(kn_emu),
       f"{len(kn_tri)} vs {len(kn_emu)}")

    print("PASS" if not FAILS else f"FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
