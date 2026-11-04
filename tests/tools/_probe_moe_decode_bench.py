# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""serving 形状 fused MoE decode 微基准（GLM-5.3-Flash TP4 分片，stock triton）。

E=288/N1=1024(shard 512)/N2=4096/K=4096/topk=8/block=128 —— 该形状不在 aiter
ASM 调优表（tuned_fmoe_asm_w8a8_group.csv 已加载但无条目）→ 桥自动落
stock triton + smem clamp（BN=BK=64/stages≤2）。2026-10-16 nmz26 实测：
36ms/步 @M=1（0.792ms/层×45）——decode 侧第二大头（第一是宿主图发射）。
后续拿到调优窗口跑 sweep 后用本探针复测对比。

跑法：容器内 `python tests/tools/_probe_moe_decode_bench.py`（standalone 跑
sglang 顶层函数需要先 ServerArgs+publish，脚本已内置）。
"""
import time

import torch

torch.manual_seed(7)
dev = "cuda"

from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.server_args import ServerArgs
import sglang.srt.runtime_context as _rtc
_sa = ServerArgs(model_path='/models/glm53f', trust_remote_code=True, tp_size=4)
_rtc.publish(_sa, role='scheduler')
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts

E, K, N_shard, TOPK = 288, 4096, 512, 8
w1 = torch.randn(E, 2 * N_shard, K, device=dev, dtype=torch.float32).div_(8).to(torch.float8_e4m3fn)
w2 = torch.randn(E, K, N_shard, device=dev, dtype=torch.float32).div_(8).to(torch.float8_e4m3fn)
w1s = torch.rand(E, (2 * N_shard) // 128, K // 128, device=dev, dtype=torch.float32) * 0.02
w2s = torch.rand(E, K // 128, N_shard // 128, device=dev, dtype=torch.float32) * 0.02
cfg = MoeRunnerConfig(num_experts=E, num_fused_shared_experts=0, top_k=TOPK,
                      hidden_size=K, intermediate_size_per_partition=N_shard,
                      swiglu_limit=10.0)

for M in (1, 8, 16):
    hs = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    ids = torch.randint(0, E, (M, TOPK), device=dev, dtype=torch.int32)
    wts = torch.rand(M, TOPK, device=dev, dtype=torch.float32)
    wts = wts / wts.sum(-1, keepdim=True)
    topk = (wts, ids, None)
    args = dict(hidden_states=hs, w1=w1, w2=w2, topk_output=topk, moe_runner_config=cfg,
                use_fp8_w8a8=True, w1_scale=w1s, w2_scale=w2s, block_shape=[128, 128])
    try:
        out = fused_experts(**args)
        torch.cuda.synchronize()
        def call():
            return fused_experts(**args)
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(30):
            call()
        torch.cuda.synchronize()
        per = (time.time() - t0) / 30 * 1000
        print(f"M={M}: {per:.3f} ms/layer -> x45 = {per*45:.1f} ms/step  out={tuple(out.shape)} {out.dtype}")
    except Exception as e:
        print(f"M={M}: FAILED {type(e).__name__}: {e}")
print("DONE")
