# REDUCTION_PATCHES — 减核役批量账（emu→triton/融合，单测先行，serving 集中验证）

单批工作流（用户定向）：逐项 parity（含 fp8 ulp 语义门）+ 秒级微基准 → 批量完成后
**一次**集中 serving A/B（nmz18 dsv41f dp8ep8+DSPARK，四门 + bench_step p50）。
不做逐项 boot 对比实验（用户 2026-10-23 指令：prof 发现问题直接解）。

## 批次账（2026-10-22/23，B 臂 = 全融合）

| # | 改动 | 文件 | 单测判决 | serving 效果 |
|---|---|---|---|---|
| 1 | DSV4.1 sparse rope/norm torch fallback → triton 门（`_fused_dsv41_kernels` needle） | `sgl_kernel/_adapters/dsv4_sparse_triton_gates.py`（7508c7d） | bitwise 10/10 + probe 3/3 | B 臂组成 |
| 2 | silu_and_mul_clamp × per-token-group-quant 融合为一核（`silu_and_mul_clamp_quant`，GPB=2 重铺修复大 M 断崖）+ **针 v3 挂点修正**（routed arm 锚 `act_and_mul_triton` else 臂，非 `silu_and_mul_clamp`；MISS/ENGAGED 一次性诊断） | `sgl_kernel/_adapters/dsv4_silu_fused_quant.py`（54d45d0 → 6ef8b3e） | fp8 ulp 门（pair-bytes 逐值 + rtol 0.08）；大 M 0.42×→1×+ | B trace act_and_mul 104ms/步段收回（wall 21.65→19.62s）；B-v2 ENGAGED 四门绿，p50 975ms |
| 3 | MHC `_HC_MIX_DOT_PRECISION`/`_HC_MIX_NUM_STAGES` import-frozen 常量针（tf32x3 崩→ieee fp32 MFMA 路） | `sgl_kernel/_adapters/dsv4_mhc_stats.py`（f74faec） | B 臂 boot 绿（原 tf32x3 compile 崩） | B 臂可跑的前置 |
| 4 | **MoE combine emu 内置 triton 化**：`moe_sum(_reduce)` 原为 aten 三连 `sum→mul→copy`（=trace in-graph `reduce_kernel<128,4>` 467µs×40 层/prefill 步 + 8×116ms 巨怪 + 每层多两趟 elementwise）；换 `_moe_sum_scale_kernel`（fp32 acc ×rsf 单次舍入，一核零中转，BLOCK_D=512/w4） | `sgl_kernel/moe.py`（6ef8b3e） | rsf=0 bitwise；rsf≠0 ≤1ulp（=上游 stock `moe_sum_reduce_triton` 语义）；M=1024 422→74µs（5.7×,848GB/s）、M=8192 3257→563µs（5.8×,895GB/s）；decode 形状（16/80）持平 | p50 975→925ms（四门绿，silu ENGAGED 复验） |

## 归因方法论（可复用）

- in-graph kernel 的 Python 栈不可用（replay correlation 对不上）→ 邻域指纹
  （prev/next kernel 名）+ e2e probe（publish(server_args, role, ranks) 前置，
  真链 `inplace_fused_experts` eager 跑，cpu_op 带形状）双轨定位。
- step 分类：按 graph replay 的 kernel 数分段（255=draft 图 / 4121=verify 图），
  不要按 correlation 全局分组（非图 kernel 污染）。
- 每 40 步 Decode-batch 日志行取间隔 = bench_step p50，不受 profiler 影响。

## 未竟（按优先级）

1. **116ms GPU 停摆（「拍」）**——见 docs/models.md 2026-10-23 行：环境级
   （kernarg 池/管理面候选），`HSA_KERNARG_POOL_SIZE=8388608` 试验进行中。
   停摆不除，一切 kernel 优化只能兑现在剩余 ~36% 占空比内。
2. `fused_k_norm_rope_flashmla` 巨实例（116ms）——已被证实为停摆落点受害者，
   非自身病（同图同 grid 6µs/116ms 两态、期间全 GPU 零并发）。停摆解决后复测。
3. `apply_rotary_emb_triton`/`_rmsnorm` decode 步各 ~5-16ms/步——量级远小于停摆，
   排后。
4. lightop 融合位（norm+rope/silu+quant 已内部完成/moe gate 冗余）——
   docs/vendor-kernels.md §2 机会表，等停摆定性后再评。
