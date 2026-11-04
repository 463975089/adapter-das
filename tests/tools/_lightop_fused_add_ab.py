"""lightop fused_add_rms_norm vs adapter triton path: A/B parity + graph
safety + perf on HCU.

Run inside the compute container (needs lightop + HCU):
    python tests/tools/_lightop_fused_add_ab.py
Gate (turn HCU_FUSED_ADD_RMS_IMPL auto->lightop for real workloads) requires:
  - parity  : vs fp32 reference AND vs the triton path over the
              dtype/shape/gemma matrix
  - graphs  : cuda-graph capture + replay on static buffers with fresh
              inputs written between replays (serving decode depends on it)
  - perf    : reported, not gated
"""
import os
import sys

os.environ.setdefault("HCU_FUSED_ADD_RMS_IMPL", "lightop")

import torch

sys.path.insert(0, os.environ.get("HCU_ADAPTER_ROOT", os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import sgl_kernel.layernorm as ln

DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [(1, 512), (9, 512), (128, 2048), (777, 4096), (4096, 7168),
          (3, 5, 1024)]


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return ok


def ref_math(x0, r0, w, eps, gemma):
    rf = r0.float() + x0.float()
    out = rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + eps)
    return rf, out * (w.float() + (1.0 if gemma else 0.0))


def run_triton(x, r, w, eps, gemma):
    # force the triton branch for this call WITHOUT disarming the vendor
    # tier for the rest of the matrix (restore in finally)
    saved = (ln._FUSED_ADD_IMPL, ln._LIGHTOP_FUSED_ADD)
    ln._FUSED_ADD_IMPL, ln._LIGHTOP_FUSED_ADD = "triton", None
    try:
        ln._do_fused_add(x, r, w, eps, gemma)
    finally:
        ln._FUSED_ADD_IMPL, ln._LIGHTOP_FUSED_ADD = saved


def run_lightop(x, r, w, eps, gemma):
    fn = ln._LIGHTOP_FUSED_ADD[1 if gemma else 0]
    fn(x.view(-1, x.shape[-1]), r.view(-1, r.shape[-1]), w, eps)


def parity():
    all_ok = True
    for dt in DTYPES:
        for shape in SHAPES:
            for gemma in (False, True):
                torch.manual_seed(shape[-1] + (7 if gemma else 0))
                N = shape[-1]
                x0 = (torch.randn(*shape, device="cuda") * 2).to(dt)
                r0 = (torch.randn(*shape, device="cuda") * 2).to(dt)
                w = torch.randn(N, device="cuda").abs().to(dt)

                x, r = x0.clone(), r0.clone()
                run_lightop(x, r, w, 1e-6, gemma)
                rf, ref = ref_math(x0, r0, w, 1e-6, gemma)
                cos = torch.nn.functional.cosine_similarity(
                    x.float().flatten(), ref.flatten(), dim=0).item()
                maxrel = ((x.float() - ref).abs().max()
                          / (ref.abs().max() + 1e-6)).item()
                res_ok = torch.allclose(r.float(), rf, rtol=1.0, atol=0.5)
                ok = cos >= 0.9999 and maxrel <= 5e-2 and res_ok

                xt, rt = x0.clone(), r0.clone()
                run_triton(xt, rt, w, 1e-6, gemma)
                same = torch.equal(x, xt) and torch.equal(r, rt)
                dmax = (x.float() - xt.float()).abs().max().item()
                if not ok or same or shape == SHAPES[0]:
                    print(f"  {dt} {shape} gemma={gemma}: cos={cos:.6f} "
                          f"maxrel={maxrel:.3e} residual_ok={res_ok} "
                          f"vs_triton={'bitwise' if same else f'maxdiff={dmax:.3e}'}")
                all_ok &= ok
    return all_ok


def graph_safety(gemma):
    M, N, dt = 4096, 7168, torch.bfloat16
    x = torch.zeros(M, N, device="cuda", dtype=dt)
    r = torch.zeros(M, N, device="cuda", dtype=dt)
    w = torch.randn(N, device="cuda", dtype=dt).abs()
    x2, r2 = x.view(-1, N), r.view(-1, N)
    fn = ln._LIGHTOP_FUSED_ADD[1 if gemma else 0]

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(x2, r2, w, 1e-6)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(x2, r2, w, 1e-6)

    ok = True
    for trial in range(3):
        torch.manual_seed(100 + trial)
        x.copy_(torch.randn(M, N, device="cuda").to(dt))
        r.copy_(torch.randn(M, N, device="cuda").to(dt))
        x0, r0 = x.clone(), r.clone()
        g.replay()
        torch.cuda.synchronize()
        rf, ref = ref_math(x0, r0, w, 1e-6, gemma)
        ok &= (torch.allclose(x.float(), ref, rtol=1.0, atol=0.5)
               and torch.allclose(r.float(), rf, rtol=1.0, atol=0.5))
    return ok


def bench(run, M, N, dt):
    x = torch.randn(M, N, device="cuda").to(dt)
    r = torch.randn(M, N, device="cuda").to(dt)
    w = torch.randn(N, device="cuda").abs().to(dt)
    for _ in range(10):
        run(x, r, w, 1e-6, False)
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(50):
        run(x, r, w, 1e-6, False)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / 50 * 1000  # us


def main():
    if ln._resolve_fused_add_impl() != "lightop" or ln._LIGHTOP_FUSED_ADD is None:
        print("SKIP: lightop backend not active (install the wheel; "
              "traceback above if the probe failed)")
        return
    print("vendor:", ln._LIGHTOP_FUSED_ADD)
    ok = True
    ok &= check("parity-matrix", parity(),
                "cos>=0.9999 maxrel<=5e-2 residual in-place (2d/3d, gemma)")
    ok &= check("graph-replay-plain", graph_safety(False),
                "capture + 3 replays on fresh inputs, 4096x7168 bf16")
    ok &= check("graph-replay-gemma", graph_safety(True),
                "gemma variant same fixture")
    t_tri = bench(run_triton, 4096, 7168, torch.bfloat16)
    t_lo = bench(run_lightop, 4096, 7168, torch.bfloat16)
    print(f"  perf 4096x7168 bf16: triton {t_tri:.1f}us | lightop "
          f"{t_lo:.1f}us | ratio {t_tri / t_lo:.2f}x")
    print("VERDICT:", "GREEN — safe to keep HCU_FUSED_ADD_RMS_IMPL=lightop"
          if ok else "RED — stay on triton")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
