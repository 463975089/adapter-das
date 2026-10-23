"""Unit test for the new flashinfer.sampling.softmax triton port (HCU adapter).

Run inside the nmz18 container:
    python /root/fi_softmax_test.py

Checks the exact face the dspark verify path uses:
    _flashinfer_softmax(logits=logits_2d, temperature=temp_per_row)
"""
import sys

import torch

sys.path.insert(0, "/root/hcu_v0522")
import sgl_kernel  # noqa: F401  (registers the flashinfer adapter)

from flashinfer.sampling import softmax as fi_softmax  # the face that crashed

dev = "cuda"
torch.manual_seed(0)


def ref(x, t):
    if not torch.is_tensor(t):
        t = torch.full((x.shape[0],), float(t), device=x.device)
    tt = torch.where(t > 0, t, torch.ones_like(t))
    return torch.softmax(x.float() / tt.view(-1, 1), dim=-1)


def check(tag, x, t):
    if not torch.is_tensor(t):
        t = torch.full((x.shape[0],), float(t), device=x.device)
    out = fi_softmax(logits=x.clone(), temperature=t.clone())
    r = ref(x, t)
    assert out.dtype == torch.float32, f"{tag}: dtype {out.dtype}"
    mad = (out - r).abs().max().item()
    rel = ((out - r).abs() / r.clamp_min(1e-6)).max().item()
    smin, smax = out.sum(-1).min().item(), out.sum(-1).max().item()
    ok = mad < 2e-6 and abs(smin - 1) < 1e-4 and abs(smax - 1) < 1e-4
    print(f"{tag:28s} maxabs={mad:.3e} maxrel={rel:.3e} "
          f"rowsum=[{smin:.6f},{smax:.6f}] {'OK' if ok else 'FAIL'}", flush=True)
    return ok


N, V = 112, 151936  # bs16 x gamma7 rows, ~150k vocab
allok = True

x = torch.randn(N, V, device=dev) * 4.0
t = torch.rand(N, device=dev) * 1.2 + 0.3
allok &= check("f32 per-row temp", x, t)

allok &= check("scalar-broadcast temp", x, torch.tensor([0.7], device=dev))
allok &= check("float temp", x[:16], 0.9)

t0 = t.clone()
t0[3] = 0.0
t0[5] = -1.0
allok &= check("temp<=0 rows (->1.0)", x, t0)

xi = x.clone()
xi[:, ::7919] = float("-inf")
allok &= check("-inf columns", xi, t)

allok &= check("odd vocab 130001", torch.randn(7, 130001, device=dev),
               torch.rand(7, device=dev) + 0.2)
allok &= check("small N=1", torch.randn(1, 8191, device=dev),
               torch.tensor([1.1], device=dev))

xb = (torch.randn(33, 4099, device=dev) * 3).to(torch.bfloat16)
allok &= check("bf16 input", xb, torch.rand(33, device=dev) + 0.2)

# non-contiguous rows (stride face)
xt = torch.randn(N, 2 * V, device=dev)[:, ::2]
assert not xt.is_contiguous() and xt.stride(1) == 2
out = fi_softmax(logits=xt, temperature=t)
r = ref(xt.float(), t)
mad = (out - r).abs().max().item()
print(f"{'non-contig stride-2 rows':28s} maxabs={mad:.3e} "
      f"{'OK' if mad < 2e-6 else 'FAIL'}", flush=True)
allok &= mad < 2e-6

# CUDA-graph capture safety + timing at serving shape
g_x = torch.randn(N, V, device=dev) * 4.0
g_t = torch.rand(N, device=dev) * 1.2 + 0.3
g_out = fi_softmax(logits=g_x, temperature=g_t)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        fi_softmax(logits=g_x, temperature=g_t)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    g_out = fi_softmax(logits=g_x, temperature=g_t)
torch.cuda.synchronize()
g.replay()
torch.cuda.synchronize()
mad = (g_out - ref(g_x, g_t)).abs().max().item()
print(f"{'graph replay parity':28s} maxabs={mad:.3e} {'OK' if mad < 2e-6 else 'FAIL'}",
      flush=True)
allok &= mad < 2e-6

import time
for _ in range(5):
    g.replay()
torch.cuda.synchronize()
t0 = time.perf_counter()
REP = 200
for _ in range(REP):
    g.replay()
torch.cuda.synchronize()
us = (time.perf_counter() - t0) / REP * 1e6
print(f"graph timing N={N} V={V}: {us:.1f} us/replay "
      f"({N * V * 4 * 2 / 1e9 / (us * 1e-6):.2f} GB/s eff r+w)", flush=True)

# torch reference timing for context
def torch_arm():
    return torch.softmax(g_x / g_t.view(-1, 1), dim=-1)
for _ in range(5):
    torch_arm()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(REP):
    torch_arm()
torch.cuda.synchronize()
print(f"torch ref same shape: {(time.perf_counter() - t0) / REP * 1e6:.1f} us", flush=True)

print("ALL_OK" if allok else "SOME_FAILED", flush=True)
