"""R-21 micro-bench: fused small moe_align vs quartet at decode shapes."""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import sgl_kernel

E, B = 48, 64
ids = torch.randint(0, E, (16, 6), device="cuda", dtype=torch.int32)
numel = ids.numel()
max_padded = numel + (E + 1) * (B - 1)
srt = torch.empty(max_padded, device="cuda", dtype=torch.int32)
eid = torch.empty((max_padded + B - 1)//B, device="cuda", dtype=torch.int32)
pst = torch.empty(1, device="cuda", dtype=torch.int32)
cs = torch.empty(E + 2, device="cuda", dtype=torch.int32)

def bench(tag):
    for _ in range(50):
        sgl_kernel.moe_align_block_size(ids, E, B, srt, eid, pst, cs)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(200):
        sgl_kernel.moe_align_block_size(ids, E, B, srt, eid, pst, cs)
    e.record(); torch.cuda.synchronize()
    print(f"{tag}: {s.elapsed_time(e)/200*1000:.1f} us/call (44 layers/step = {s.elapsed_time(e)/200*44/1000:.2f} ms/step)")

os.environ["HCU_MOE_ALIGN_FUSED"] = "2048"; bench("fused ")
os.environ["HCU_MOE_ALIGN_FUSED"] = "0";    bench("quartet")
os.environ["HCU_MOE_ALIGN_FUSED"] = "2048"
