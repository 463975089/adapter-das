"""Which kernels does dsa_mqa_logits_emu.fp8_paged_mqa_logits fire at decode shapes?"""
import os, sys, json
import torch
sys.path.insert(0, "/root/hcu_v0522")
sys.path.insert(0, "/sglang-v0522/python")
from hcu_adapter import dsa_mqa_logits_emu
from hcu_adapter.hcu_env import resolve
v, src = resolve("HCU_MQA_LOGITS_IMPL")
print(f"resolve HCU_MQA_LOGITS_IMPL = {v} ({src})")

N, H, D, PS = 96, 32, 128, 64
P = 1024
q = torch.randn(N, H, D, device="cuda").clamp(-2, 2).to(torch.float8_e4m3fn)
pool = torch.randint(0, 255, (P, PS, 1, 132), device="cuda", dtype=torch.uint8)
w = torch.rand(N, H, device="cuda")
lens = torch.randint(64, 1024, (N,), device="cuda", dtype=torch.int32)
bt = torch.randint(0, P, (N, 16), device="cuda", dtype=torch.int32)
out = dsa_mqa_logits_emu.fp8_paged_mqa_logits(
    q, pool, w, lens, bt, None, 1024, False)
torch.cuda.synchronize()
from torch.profiler import ProfilerActivity, profile
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    dsa_mqa_logits_emu.fp8_paged_mqa_logits(q, pool, w, lens, bt, None, 1024, False)
    torch.cuda.synchronize()
prof.export_chrome_trace("/tmp/mqa_route.json")
d = json.load(open("/tmp/mqa_route.json"))
names = [e.get("name","") for e in d["traceEvents"] if e.get("cat") == "kernel"]
print(f"kernels={len(names)}")
for n in names[:10]:
    print("  ", n[:80])
