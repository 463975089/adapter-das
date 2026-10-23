import torch, aiter
from aiter.fused_moe_c import per_token_group_quant_fp8

torch.manual_seed(0)
E, K, N, M, topk, BM = 384, 5120, 2304, 64, 6, 16
qw = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device="cuda")
ws = torch.randint(1, 250, (E, N, K // 32), dtype=torch.uint8, device="cuda")
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.05
qinput, a_scale = per_token_group_quant_fp8(x, 32)
qinput_cm, a_scale_cm = per_token_group_quant_fp8(x, 32, column_major_scales=True)

# hand-built moe_align outputs (deterministic layout, independent of aiter align kernel)
assign = torch.stack([torch.randperm(E, device="cpu")[:topk] for _ in range(M)]).int()
t_idx = torch.arange(M).repeat_interleave(topk)
e_flat = assign.flatten()
order = torch.argsort(e_flat.long(), stable=True)
st = t_idx[order]
se = e_flat[order]
counts = torch.bincount(e_flat.long(), minlength=E)
nblocks = (counts + BM - 1) // BM
total = int(nblocks.sum()) * BM
sorted_ids = torch.full((total,), M, dtype=torch.int32)
expert_ids = torch.zeros(total // BM, dtype=torch.int32)
pos = 0
for e in range(E):
    if int(nblocks[e]) == 0:
        continue
    ids = st[se == e]
    sorted_ids[pos:pos + len(ids)] = ids.to(torch.int32)
    expert_ids[pos // BM] = e
    pos += int(nblocks[e]) * BM
ntp = torch.tensor([total], dtype=torch.int32)
sorted_ids, expert_ids, ntp = sorted_ids.cuda(), expert_ids.cuda(), ntp.cuda()
print("align handmade: total", total, "blocks", total // BM, flush=True)

out = torch.zeros(M * topk, N, device="cuda", dtype=torch.bfloat16)
kq = aiter.moe_c_moe_gemm_marlin_wfp4a8_groupwise_qgroup
kg = aiter.moe_c_moe_gemm_marlin_wfp4a8_groupwise
for mode in (9525, 9523):
    for nm, k, q, a in (("qgroup_rm", kq, qinput, a_scale), ("qgroup_cm", kq, qinput_cm, a_scale_cm), ("plain_rm", kg, qinput, a_scale)):
        for delta, size_m in ((topk, M), (BM, M), (topk, total // BM)):
            try:
                k(q, qw, out, a, ws, None, sorted_ids, expert_ids, ntp, topk, mode, delta, size_m)
                torch.cuda.synchronize()
                nz = (out != 0).float().mean().item()
                print(f"RUNS mode={mode} {nm} d={delta} s={size_m}: nonzero={nz:.3f} row0[:4]={out[0,:4].tolist()}", flush=True)
                break
            except RuntimeError as e:
                print(f"mode={mode} {nm} d={delta} s={size_m}: {str(e)[:110]}", flush=True)
print("done", flush=True)
