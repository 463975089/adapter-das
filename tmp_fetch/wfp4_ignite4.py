import torch, aiter
from aiter.fused_moe_c import per_token_group_quant_fp8

torch.manual_seed(0)
E, K, N, M, topk, BM = 384, 5120, 2304, 64, 6, 16
qw = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device="cuda")
ws = torch.randint(1, 250, (E, N, K // 32), dtype=torch.uint8, device="cuda")
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.05
qinput, a_scale = per_token_group_quant_fp8(x, 32)
qinput_cm, a_scale_cm = per_token_group_quant_fp8(x, 32, column_major_scales=True)

topk_ids = torch.stack([torch.randperm(E, device="cuda")[:topk] for _ in range(M)]).to(torch.int32)
max_pad = M * topk + E * (BM - 1)
sorted_ids = torch.zeros(max_pad, dtype=torch.int32, device="cuda")
exp_ids = torch.zeros(max_pad // BM, dtype=torch.int32, device="cuda")
ntp = torch.zeros(1, dtype=torch.int32, device="cuda")
torch.ops.aiter.moe_align_block_size(topk_ids, E, BM, sorted_ids, exp_ids, ntp)
torch.cuda.synchronize()
print("align OK: ntp =", ntp.item(), "blocks_used", ntp.item() // BM, flush=True)

out = torch.zeros(M * topk, N, device="cuda", dtype=torch.bfloat16)
kq = aiter.moe_c_moe_gemm_marlin_wfp4a8_groupwise_qgroup
modes = [11001101, 11101101, 11201102, 11002101, 11102101, 9525, 9523, 9531]
for mode in modes:
    for nm, q, a, delta, size_m in (
        ("qgroup_rm", qinput, a_scale, topk, M),
        ("qgroup_cm", qinput_cm, a_scale_cm, topk, M),
    ):
        try:
            kq(q, qw, out, a, ws, None, sorted_ids, exp_ids, ntp, topk, mode, delta, size_m)
            torch.cuda.synchronize()
            nz = (out != 0).float().mean().item()
            print(f"RUNS mode={mode} {nm}: nonzero={nz:.3f} row0[:4]={out[0,:4].tolist()}", flush=True)
        except RuntimeError as e:
            print(f"mode={mode} {nm}: {str(e)[:100]}", flush=True)
print("done", flush=True)
