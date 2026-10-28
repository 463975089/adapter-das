# -*- coding: utf-8 -*-
"""CUDA-graph capture smoke for the ragged triton arm (decode captures it)."""
import torch

torch.manual_seed(7)
dev = "cuda:0"
N, H, D, T = 16, 32, 128, 8192

from hcu_adapter.mqa_logits_triton import triton_ragged_mqa_logits as tri
from hcu_adapter import mqa_logits_emu as emu_mod

q = (torch.randn(N, H, D, device=dev) * 4).to(torch.float8_e4m3fn)
k = (torch.randn(T, D, device=dev) * 4).to(torch.float8_e4m3fn)
sc = torch.rand(T, dtype=torch.float32, device=dev) + 0.25
w = torch.rand(N, H, dtype=torch.float32, device=dev)
ks = torch.zeros(N, dtype=torch.int64, device=dev)
ke = torch.full((N,), T - 10, dtype=torch.int64, device=dev)

# warmup on a side stream (allocator capture mode requirement)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        out = tri(q, k, sc, w, ks, ke)
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = tri(q, k, sc, w, ks, ke)

# mutate inputs in place, replay, verify the kernel re-reads them
q.copy_(torch.randn_like(q.float()).clamp(-4, 4).to(torch.float8_e4m3fn))
sc.mul_(1.5)
g.replay()
torch.cuda.synchronize()

ref = tri(q, k, sc, w, ks, ke)
rel = ((static_out.clamp(min=-1e30) - ref.clamp(min=-1e30)).abs()
       / ref.abs().clamp_min(1e-4))
mask = ref > float("-inf")
print("capture OK; replay vs fresh median rel:",
      float(rel[mask].median()))
assert float(rel[mask].median()) < 1e-5, "graph replay mismatch"
# window edges stay -inf after replay
t = torch.arange(T, device=dev)[None, :].expand(N, T)
outside = t >= ke[:, None]
assert bool((static_out[outside] == float("-inf")).all()), "-inf broken"
print("replay -inf semantics OK — GRAPH SAFE")
