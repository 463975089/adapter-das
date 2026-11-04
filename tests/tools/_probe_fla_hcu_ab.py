# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""vendored sglang-das hcu 内核 vs adapter emu —— A/B 对拍 + 微基准（gfx938）。

参考侧（gfx938 已知正确）：
- KDA prefill: kda_prefill_emu（顺序 torch，绿门）
- conv1d fn/update: conv1d emus（绿门）
- KDA decode: 官方 v0.5.21 fused_sigmoid_gating_recurrent（健康）
挑战侧：hcu_adapter.fla_hcu.*（vendored fork 内核，纯 triton）。

2026-10-16 实测：chunk_kda out/final-state cos≥0.9999 快 473-1009×；
conv fn 16.4×、update 4.3×（状态位精确相等）；decode 与 stock 同速不换。
beta 约定：fork 期望已激活（sigmoid 过）的 beta（v0.5.21 后端给的就是激活值）。

**注意**：本探针是内核级证据，standalone 全绿 ≠ serving 门绿——fork 在服务
上下文 Berlin 锚点确定性翻转（详见 docs/models.md 已知问题表 + env-vars.md
的 HCU_KDA_PREFILL_IMPL 行）。serving dump 对拍是后续立项。

跑法：容器内 `python tests/tools/_probe_fla_hcu_ab.py`（需 GPU +
sitecustomize 所在 adapter 树为 PYTHONPATH 根）。
"""
import time

import torch

torch.manual_seed(7)
dev = "cuda"
H, K, V, SLOTS = 16, 128, 128, 32


def stats(a, b, tag):
    a = a.float(); b = b.float()
    va, vb = a.flatten(), b.flatten()
    cos = torch.nn.functional.cosine_similarity(va, vb, dim=0).item()
    rms = (va - vb).pow(2).mean().sqrt().item()
    rel = rms / (va.pow(2).mean().sqrt().item() + 1e-9)
    print(f"  {tag}: cos={cos:.6f} rms={rms:.4g} rel={rel:.4g}")
    return cos


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


print("=== 1) KDA prefill chunk: fork triton vs emu ===")
from hcu_adapter.fla_hcu.kda import chunk_kda
from hcu_adapter.kda_prefill_emu import kda_extend_emu

for (T1, T2) in [(1536, 512), (2048, 0), (8192, 0)]:
    B = 2 if T2 else 1
    T = T1 + T2
    q = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, T, H, V, device=dev, dtype=torch.bfloat16)
    g = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16) * 0.5
    beta = torch.randn(1, T, H, device=dev, dtype=torch.bfloat16)
    A_log = torch.randn(H, device=dev, dtype=torch.float32) * 0.1 - 1.0
    dt_bias = torch.randn(H * K, device=dev, dtype=torch.float32) * 0.05
    cu = torch.tensor([0, T1, T1 + T2][: B + 1], device=dev, dtype=torch.int32)
    idx = torch.arange(B, device=dev, dtype=torch.int32)
    st0 = torch.randn(SLOTS, H, V, K, device=dev, dtype=torch.float32) * 0.05
    lb, scale = -5.0, K**-0.5

    st_e = st0.clone()
    o_e, _ = kda_extend_emu(q, k, v, g, beta, ssm_states=st_e, cache_indices=idx,
                            query_start_loc=cu, A_log=A_log, dt_bias=dt_bias,
                            lower_bound=lb, scale=scale)
    st_f = st0.clone()
    beta_act = beta.float().sigmoid().to(beta.dtype)
    o_f = chunk_kda(q=q, k=k, v=v, g=g, beta=beta_act, initial_state=st_f,
                    initial_state_indices=idx, use_qk_l2norm_in_kernel=True,
                    cu_seqlens=cu, A_log=A_log, dt_bias=dt_bias, lower_bound=lb)
    print(f" T={T} B={B}:")
    stats(o_e, o_f.to(o_e.dtype), "out")
    stats(st_e[idx], st_f[idx].float(), "final-state")
    if T >= 2048:
        t_e = bench(lambda: kda_extend_emu(q, k, v, g, beta, ssm_states=st_e,
                                           cache_indices=idx, query_start_loc=cu,
                                           A_log=A_log, dt_bias=dt_bias,
                                           lower_bound=lb, scale=scale), iters=3, warmup=1)
        t_f = bench(lambda: chunk_kda(q=q, k=k, v=v, g=g, beta=beta_act, initial_state=st_f,
                                      initial_state_indices=idx,
                                      use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
                                      A_log=A_log, dt_bias=dt_bias, lower_bound=lb), iters=10)
        print(f"  time: emu={t_e:.1f}ms fork={t_f:.2f}ms  speedup={t_e/t_f:.0f}x")

print("=== 2) causal_conv1d_fn varlen: fork vs emu ===")
from hcu_adapter.conv1d_fn_emu import conv1d_fn_emu
from hcu_adapter.fla_hcu.conv1d_triton import causal_conv1d_fn

DIM, W = 2048, 4
T1, T2 = 1536, 512
T = T1 + T2
x = torch.randn(DIM, T, device=dev, dtype=torch.bfloat16).t().contiguous()  # token-major
wt = torch.randn(DIM, W, device=dev, dtype=torch.bfloat16) * 0.2
bias = torch.randn(DIM, device=dev, dtype=torch.bfloat16) * 0.1
cu = torch.tensor([0, T1, T1 + T2], device=dev, dtype=torch.int32)
idx = torch.arange(2, device=dev, dtype=torch.int32)
cis = torch.tensor([True, True], device=dev)
cs_e = torch.randn(SLOTS, DIM, W - 1, device=dev, dtype=torch.bfloat16) * 0.3
cs_f = cs_e.clone()
o_e = conv1d_fn_emu(x, wt, bias, conv_states=cs_e, query_start_loc=cu,
                    seq_lens_cpu=[T1, T2], cache_indices=idx, has_initial_state=cis)
o_f = causal_conv1d_fn(x, wt, bias, conv_states=cs_f, query_start_loc=cu,
                       cache_indices=idx, has_initial_state=cis, activation="silu")
stats(o_e, o_f.to(o_e.dtype), "out")
stats(cs_e[idx], cs_f[idx], "state")
t_e = bench(lambda: conv1d_fn_emu(x, wt, bias, conv_states=cs_e, query_start_loc=cu,
                                  seq_lens_cpu=[T1, T2], cache_indices=idx,
                                  has_initial_state=cis), iters=10)
t_f = bench(lambda: causal_conv1d_fn(x, wt, bias, conv_states=cs_f, query_start_loc=cu,
                                     cache_indices=idx, has_initial_state=cis,
                                     activation="silu"), iters=20)
print(f"  time: emu={t_e:.2f}ms fork={t_f:.2f}ms speedup={t_e/t_f:.1f}x")

print("=== 3) causal_conv1d_update: fork vs emu ===")
from hcu_adapter.conv1d_update_emu import causal_conv1d_update_emu
from hcu_adapter.fla_hcu.conv1d_triton import causal_conv1d_update

BS = 16
xu = torch.randn(BS, DIM, device=dev, dtype=torch.bfloat16)
su_e = torch.randn(SLOTS, DIM, W - 1, device=dev, dtype=torch.bfloat16) * 0.3
su_f = su_e.clone()
ui = torch.arange(BS, device=dev, dtype=torch.int32)
o_e = causal_conv1d_update_emu(xu, su_e, wt, bias, activation="silu", conv_state_indices=ui)
o_f = causal_conv1d_update(xu, su_f, wt, bias, activation="silu", conv_state_indices=ui)
stats(o_e, o_f.to(o_e.dtype), "out")
stats(su_e[ui], su_f[ui], "state")
t_e = bench(lambda: causal_conv1d_update_emu(xu, su_e, wt, bias, activation="silu",
                                             conv_state_indices=ui), iters=50)
t_f = bench(lambda: causal_conv1d_update(xu, su_f, wt, bias, activation="silu",
                                         conv_state_indices=ui), iters=100)
print(f"  time: emu={t_e:.3f}ms fork={t_f:.3f}ms speedup={t_e/t_f:.1f}x")

print("=== 4) KDA decode: fork hcu recurrent vs stock v0.5.21 ===")
try:
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update as stock_dec,
    )
    from hcu_adapter.fla_hcu.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update as fork_dec,
    )
    qd = torch.randn(1, BS, H, K, device=dev, dtype=torch.bfloat16)
    kd = torch.randn_like(qd)
    vd = torch.randn(1, BS, H, V, device=dev, dtype=torch.bfloat16)
    ad = torch.randn(1, BS, H, K, device=dev, dtype=torch.bfloat16) * 0.3
    bd = torch.randn(1, BS, H, device=dev, dtype=torch.bfloat16)
    sd_e = torch.randn(SLOTS, H, V, K, device=dev, dtype=torch.float32) * 0.05
    sd_f = sd_e.clone()
    cu_d = torch.arange(0, BS + 1, device=dev, dtype=torch.int32)
    common = dict(A_log=A_log, dt_bias=dt_bias, initial_state_source=sd_e,
                  initial_state_indices=ui, cu_seqlens=cu_d,
                  use_qk_l2norm_in_kernel=True, softplus_beta=1.0,
                  softplus_threshold=20.0, is_kda=True, lower_bound=lb)
    o_e = stock_dec(q=qd, k=kd, v=vd, a=ad, b=bd, **common)
    common["initial_state_source"] = sd_f
    o_f = fork_dec(q=qd, k=kd, v=vd, a=ad, b=bd, **common)
    stats(o_e, o_f.to(o_e.dtype), "out")
    t_e = bench(lambda: stock_dec(q=qd, k=kd, v=vd, a=ad, b=bd, **common), iters=50)
    t_f = bench(lambda: fork_dec(q=qd, k=kd, v=vd, a=ad, b=bd, **common), iters=100)
    print(f"  time: stock={t_e:.3f}ms fork={t_f:.3f}ms speedup={t_e/t_f:.1f}x")
except Exception as e:
    print(f"  decode compare skipped: {type(e).__name__}: {e}")

print("ALL DONE")
