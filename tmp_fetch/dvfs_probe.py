"""DVFS probe: does sparse kernel launching (serving-like bursty duty cycle)
inflate per-kernel time vs back-to-back replay?  If yes, the production
trace's inflated p50s (silu 598us vs 37us standalone) are clock-governor
punishment for bursty tiny-kernel decode, not kernel regression."""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/root/hcu_v0522")
sys.path.insert(0, "/sglang-v0522/python")

import torch

from sgl_kernel._adapters.dsv4_silu_fused_quant import silu_and_mul_clamp_quant

dev = "cuda"
M, D = 960, 2304
x = torch.randn(M, 2 * D, device=dev, dtype=torch.bfloat16) * 3
q = torch.empty(M, D, device=dev, dtype=torch.bfloat16)  # placeholder dtype
q0, s0 = silu_and_mul_clamp_quant(x, 7.0, 128)

# build graph of one call
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        silu_and_mul_clamp_quant(x, 7.0, 128, q0, s0)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    silu_and_mul_clamp_quant(x, 7.0, 128, q0, s0)
torch.cuda.synchronize()

ev0 = torch.cuda.Event(enable_timing=True)
ev1 = torch.cuda.Event(enable_timing=True)


def measure_duty(sleep_s, iters=300, tag=""):
    # kernel-time via events inside a replay-only span; wall includes sleep
    ks = []
    for _ in range(30):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        if sleep_s > 0:
            g.replay()
            torch.cuda.synchronize()
            ev0.record()
            g.replay()
            ev1.record()
            torch.cuda.synchronize()
            ks.append(ev0.elapsed_time(ev1) * 1000)
            time.sleep(sleep_s)
        else:
            g.replay()
    if sleep_s > 0:
        torch.cuda.synchronize()
        ks.sort()
        print(f"duty[{tag}] sleep={sleep_s*1000:.0f}ms: kernel p50={ks[len(ks)//2]:.1f}us "
              f"p90={ks[len(ks)*9//10]:.1f}us", flush=True)
    else:
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / iters * 1e6
        print(f"duty[{tag}] back-to-back: wall/iter={wall:.1f}us", flush=True)


measure_duty(0, tag="100%")
measure_duty(0.001, tag="~4%")
measure_duty(0.005, tag="~1%")
measure_duty(0.010, tag="~0.4%")
measure_duty(0.050, tag="~0.1%")
print("DVFS_PROBE_DONE", flush=True)
