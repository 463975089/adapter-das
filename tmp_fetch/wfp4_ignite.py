import torch, aiter
from aiter.fused_moe_c import per_token_group_quant_fp8, moe_align_block_size

torch.manual_seed(0)
E, K, N, M, topk = 384, 5120, 2304, 64, 6
qw = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device="cuda")
ws = torch.randint(1, 250, (E, N, K // 32), dtype=torch.uint8, device="cuda")
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.05
qinput, a_scale = per_token_group_quant_fp8(x, 32)
qinput_cm, a_scale_cm = per_token_group_quant_fp8(x, 32, column_major_scales=True)
topk_ids = torch.stack([torch.randperm(E, device="cuda")[:topk] for _ in range(M)]).to(torch.int32)
sorted_ids, expert_ids, ntp = moe_align_block_size(topk_ids, E, 16, None)
out = torch.zeros(M * topk, N, device="cuda", dtype=torch.bfloat16)
kq = aiter.moe_c_moe_gemm_marlin_wfp4a8_groupwise_qgroup
kg = aiter.moe_c_moe_gemm_marlin_wfp4a8_groupwise
for mode in (9525, 9523):
    for nm, k, q, a, delta, size_m in (
        ("qgroup_rm", kq, qinput, a_scale, topk, M),
        ("qgroup_cm", kq, qinput_cm, a_scale_cm, topk, M),
        ("plain_rm", kg, qinput, a_scale, topk, M),
    ):
        try:
            k(q, qw, out, a, ws, None, sorted_ids, expert_ids, ntp, topk, mode, delta, size_m)
            torch.cuda.synchronize()
            nz = (out != 0).float().mean().item()
            print(f"RUNS mode={mode} {nm}: nonzero={nz:.3f} row0[:4]={out[0,:4].tolist()}", flush=True)
        except RuntimeError as e:
            print(f"mode={mode} {nm}: {type(e).__name__}: {str(e)[:130]}", flush=True)
print("done", flush=True)
