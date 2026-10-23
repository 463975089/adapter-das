"""Standalone bench: _silu_clamp_ptgq_kernel at the production padded shape
M=960, N2=4608 (D=2304, group 128, GPB=2) vs bandwidth floor, plus config
sweep (num_warps, GPB, multi-row CTAs) and the stock pair for reference."""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/root/hcu_v0522")
sys.path.insert(0, "/sglang-v0522/python")

import torch
import triton
import triton.language as tl

from sgl_kernel._adapters.dsv4_silu_fused_quant import silu_and_mul_clamp_quant

dev = "cuda"
torch.manual_seed(0)

M, D = 960, 2304
x = (torch.randn(M, 2 * D, device=dev, dtype=torch.bfloat16) * 3)
limit = 7.0

q0, s0 = silu_and_mul_clamp_quant(x, limit, 128)
print(f"shape M={M} D={D} q={tuple(q0.shape)} s={tuple(s0.shape)}", flush=True)


def bench_graph(fn, rep=200):
    for _ in range(5):
        fn()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep * 1e6


us = bench_graph(lambda: silu_and_mul_clamp_quant(x, limit, 128, q0, s0))
traffic = M * (2 * D * 2 + D * 1 + D // 128 * 4)
print(f"prod config (GPB=2, warps=1): {us:.1f}us  "
      f"({traffic/1e9/(us*1e-6):.1f} GB/s eff; floor ~{traffic/3300e9*1e6:.1f}us)",
      flush=True)

# stock pair reference
from sglang.kernels.ops.moe.dsv4 import silu_and_mul_clamp  # noqa: E402
y = torch.empty(M, D, device=dev, dtype=torch.bfloat16)
us_pair = bench_graph(lambda: silu_and_mul_clamp(x, y, limit))
print(f"pair act only: {us_pair:.1f}us", flush=True)


# sweep variants: warps x GPB via monkey grid
@triton.jit
def _k(X, Q, S, D, x_stride, q_stride, s_stride, limit, eps,
       bit8_min, bit8_max, GROUP: tl.constexpr, GPB: tl.constexpr,
       ROWS: tl.constexpr):
    row0 = tl.program_id(0).to(tl.int64) * ROWS
    gb = tl.program_id(1)
    W: tl.constexpr = GROUP * GPB
    cols = gb * W + tl.arange(0, W)
    mask = cols < D
    for r in tl.static_range(ROWS):
        row = row0 + r
        xb = X + row * x_stride
        gate = tl.load(xb + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(xb + D + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
        y = gate * tl.sigmoid(gate) * up
        y2 = tl.reshape(y, (GPB, GROUP))
        _absmax = tl.maximum(tl.max(tl.abs(y2), axis=1), eps)
        y_s = _absmax / bit8_max
        y_q = tl.clamp(y2 * (1.0 / y_s)[:, None], bit8_min, bit8_max)
        tl.store(Q + row * q_stride + cols,
                 tl.reshape(y_q, (W,)).to(Q.dtype.element_ty), mask=mask)
        tl.store(S + row * s_stride + gb * GPB + tl.arange(0, GPB), y_s)


from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, fp8_max  # noqa: E402

q = torch.empty(M, D, device=dev, dtype=fp8_dtype)
s = torch.empty(M, D // 128, device=dev, dtype=torch.float32)
n_groups = D // 128
print(f"{'warps':>6} {'GPB':>4} {'ROWS':>5} {'us':>8}  {'GB/s':>7}", flush=True)
for warps in (1, 2, 4):
    for gpb in (1, 2, 4):
        if n_groups % gpb:
            continue
        for rows in (1, 2, 4):
            if M % rows:
                continue

            def run(warps=warps, gpb=gpb, rows=rows):
                _k[(M // rows, n_groups // gpb)](
                    x, q, s, D, x.stride(0), q.stride(0), s.stride(0),
                    limit, 1e-10, -float(fp8_max), float(fp8_max),
                    GROUP=128, GPB=gpb, ROWS=rows,
                    num_warps=warps, num_stages=1,
                )
            try:
                run()
                torch.cuda.synchronize()
                mad = (q.float() - q0.float()).abs().max().item()
                us2 = bench_graph(run)
                print(f"{warps:6d} {gpb:4d} {rows:5d} {us2:8.1f}  "
                      f"{traffic/1e9/(us2*1e-6):7.1f}  maxabs={mad:.4f}", flush=True)
            except Exception as e:
                print(f"{warps} {gpb} {rows} FAIL {type(e).__name__} {str(e)[:60]}",
                      flush=True)
print("BENCH_DONE", flush=True)
