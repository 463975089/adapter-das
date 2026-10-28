"""vec 形态 graph-replay 病理复现：eager vs 图内 replay vs 预分配 out= 版。

serving 现象：vec 在图内 replay 时最大 aten 算子 ~124-130ms，eager 仅
1-5ms；loop（小 tile 分配）图内无此现象。怀疑图私有池大块分配 -> replay
爬行。本探针在真实 decode 形状上三路对照：
  A eager vec（现状代码）
  B graph 捕获+replay vec（复现病理）
  C graph 捕获+replay，全部大中间体用预分配工作区 + out= 算子

跑法：容器 GPU 上 python3 /root/mqa_graph.py
"""
import os
import time

import torch

torch.manual_seed(3)
DEV = "cuda"
N, H, D, T, PS = 16, 16, 128, 65536, 64
POOL_TOKENS = 729856  # real per-rank pool

q = (torch.randn(N, H, D, device=DEV) * 0.3).to(torch.float8_e4m3fn)
kv = torch.zeros(POOL_TOKENS, 132, device=DEV, dtype=torch.uint8)
kv[:, :D] = (torch.randn(POOL_TOKENS, D, device=DEV) * 0.2).to(torch.float8_e4m3fn).view(torch.uint8)
kv[:, D:] = (torch.rand(POOL_TOKENS, 1, device=DEV) * 0.02 + 0.99).float().view(torch.uint8)
w = torch.rand(N, H, device=DEV).float()
ctx = torch.randint(1, T, (N, 1), device=DEV, dtype=torch.int32)
bt = torch.stack([torch.randperm(POOL_TOKENS // PS)[: T // PS] for _ in range(N)]).to(torch.int32).to(DEV)

from hcu_adapter.dsa_mqa_logits_emu import fp8_paged_mqa_logits  # noqa: E402


def vec_body():
    # mirrors the impl=="vec" arm (N<=32) with max_context_len=T
    return fp8_paged_mqa_logits(q, kv, w, ctx, bt, None, T, clean_logits=True)


def vec_body_prealloc(ws):
    """same math, all big intermediates in caller-provided workspaces."""
    kg_ws, s_ws, row_ws = ws
    fp8 = q.dtype
    q_b = q.to(torch.bfloat16).contiguous()
    k_all = kv[:, :128].view(fp8).to(torch.bfloat16)  # (P, D) -- still per-call; small enough? measure
    k_sf = kv[:, 128:].view(torch.float32).reshape(-1)
    pages = bt
    toks = (pages.unsqueeze(2) * PS + torch.arange(PS, device=DEV)).reshape(N, -1)[:, :T]
    torch.index_select(k_all, 0, toks.reshape(-1), out=kg_ws)
    kg2 = kg_ws.view(N, T, D)
    sc = k_sf[toks]
    torch.matmul(q_b, kg2.transpose(1, 2), out=s_ws)  # (N,H,T) bf16
    torch.matmul(w.unsqueeze(1).to(torch.bfloat16), s_ws, out=row_ws)  # (N,1,T)
    row = row_ws.squeeze(1).float() * sc
    valid = torch.arange(T, device=DEV).unsqueeze(0) < ctx
    return torch.where(valid, row, torch.zeros((), device=DEV))


def timeit(fn, iters=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


os.environ["HCU_MQA_LOGITS_IMPL"] = "vec"

tA = timeit(vec_body)
print(f"A eager vec:                 {tA:8.2f} ms/call")

g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2):
        vec_body()
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    outB = vec_body()
tB = timeit(lambda: g.replay())
print(f"B graph replay vec:          {tB:8.2f} ms/call")

# C: prealloc workspaces OUTSIDE capture, out= ops inside
kg_ws = torch.empty(N * T, D, device=DEV, dtype=torch.bfloat16)
s_ws = torch.empty(N, H, T, device=DEV, dtype=torch.bfloat16)
row_ws = torch.empty(N, 1, T, device=DEV, dtype=torch.bfloat16)
g2 = torch.cuda.CUDAGraph()
s2 = torch.cuda.Stream()
s2.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s2):
    for _ in range(2):
        vec_body_prealloc((kg_ws, s_ws, row_ws))
torch.cuda.current_stream().wait_stream(s2)
with torch.cuda.graph(g2):
    outC = vec_body_prealloc((kg_ws, s_ws, row_ws))
tC = timeit(lambda: g2.replay())
print(f"C graph replay prealloc out=: {tC:8.2f} ms/call")
print("DONE")
