import torch, aiter
from aiter.fused_moe_c import per_token_group_quant_fp8, moe_align_block_size

torch.manual_seed(0)
E, K, N, M, topk = 384, 5120, 2304, 64, 6
qw = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device="cuda")
ws = torch.randint(1, 250, (E, N, K // 32), dtype=torch.uint8, device="cuda")
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.05
qinput, a_scale = per_token_group_quant_fp8(x, 32)
qinput_cm, a_scale_cm = per_token_group_quant_fp8(x, 32, column_major_scales=True)
print('scale rm', tuple(a_scale.shape), a_scale.stride(), 'cm', tuple(a_scale_cm.shape), a_scale_cm.stride(), flush=True)
topk_ids = torch.stack([torch.randperm(E, device="cuda")[:topk] for _ in range(M)]).to(torch.int32)
sorted_ids, expert_ids, ntp = moe_align_block_size(topk_ids, E, 16, None)
out = torch.zeros(M * topk, N, device="cuda", dtype=torch.bfloat16)
kq = aiter.moe_c_moe_gemm_marlin_wfp4a8_groupwise_qgroup
kg = aiter.moe_c_moe_gemm_marlin_wfp4a8_groupwise
found = 0
tried = 0
for nm, k, qin, asc in (("qgroup_cm", kq, qinput_cm, a_scale_cm), ("qgroup_rm", kq, qinput, a_scale), ("plain", kg, qinput, a_scale)):
    for stage in (1, 2):
        for family in (0, 1, 2, 3):
            for bm16 in (1, 2, 4):
                for profile in (101, 102, 104):
                    mode = stage*10**7 + 10**6 + family*10**5 + bm16*10**3 + profile
                    for delta, size_m in ((topk, M), (16, M * topk)):
                        tried += 1
                        print(f"TRY {nm} mode={mode} delta={delta} size_m={size_m}", flush=True)
                        try:
                            k(qin, qw, out, asc, ws, None,
                              sorted_ids, expert_ids, ntp, topk, mode, delta, size_m)
                            torch.cuda.synchronize()
                            print(f"RUNS {nm} mode={mode} delta={delta} size_m={size_m} nonzero={(out != 0).float().mean().item():.3f}", flush=True)
                            found += 1
                        except RuntimeError as e:
                            msg = str(e)
                            if "mode" not in msg[:60]:
                                print(f"NEW-ERR {nm} mode={mode} d{delta} s{size_m}: {msg[:140]}", flush=True)
                        if found >= 3:
                            break
                    if found >= 3:
                        break
                if found >= 3:
                    break
            if found >= 3:
                break
        if found >= 3:
            break
    if found >= 3:
        break
print(f"tried={tried} found={found}", flush=True)
