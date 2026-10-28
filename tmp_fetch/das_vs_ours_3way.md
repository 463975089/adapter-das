# das(bw1000) vs 我们(nmz18) 三路对照 — 2026-10-22

基准口径：
- **das**：DPSK-V4.1_I32768_O1024_BS32_PD_S5.zip（2026-10-20 trace，bw1000 8卡，PD 分离，decode 节点 DP8EP8，DSPARK S5，BS32）。实测 TPOT 17ms、节点 8 卡 3000 tok/s。
- **我们**：nmz18 统一节点 DP8EP8，DSPARK S5，BS16（launch_ab1023_B），step p50 925ms ≈ TPOT ~300ms（~3 接受/步）。

---

## 路一：停摆环境对照

| | das (bw1000, dtk26041 镜像) | 我们 (nmz18, 旧 DTK) |
|---|---|---|
| >10ms kernel | **0** | 全部 8 卡周期性受害 kernel |
| GPU duty | 86.9%（busy≈wall） | ~36%（stall 吃掉 64%） |
| 停摆 | 无 | ~116ms @ ~181ms 全卡同步拍，idle 也拍 |
| 驱动层规避 env | **一个都没有**（无 HSA_KERNARG_POOL_SIZE 等） | HSA_KERNARG_POOL_SIZE=8M 已排队待测 |

cookbook 全程 9-15 → 10-22 无任何驱动级时间片规避手段——他们从未遇到这个问题。
⇒ **拍是 nmz18 盒子/驱动栈特有，不是模型/框架/DSPARK 家族固有**。修复阶梯：HSA_KERNARG_POOL_SIZE 试验 → 驱动/DTK 对齐 dtk2604 → 盒级工单。

das 步长时间自洽：draft 14.8ms + verify 51.5ms = 66.3ms busy（零空泡），~3.9 接受/步 → TPOT 17ms。

## 路二：结构 diff（单步 kernel 账）

| | das | 我们 |
|---|---|---|
| draft kernels | 206 / 14.8ms | 255 |
| verify kernels | **2196** / 51.5ms | **4121**（≈2×） |
| GEMM 占 busy | 56%（健康：GEMM 主导） | 胶水 kernel 主导，decode GEMM 是 µs 级噪音 |
| rmsnorm | **5.6µs**（lightop vendor） | ~450µs（**80×**） |
| router gate | **28.4µs**（moe_fused_gate_sqrtsoftplus vendor） | ~350µs（12×） |
| rotary / silu_mul_clamp | 5.6µs / 9.5µs | 同类膨胀 |
| a2a | DeepEP internode_ll | aiter AG |

das 自己的优化账印证"地板即收益"：fp4_mqa_logits 2-3×→4% e2e；dequantize_k_cache+combine_topk→2%；**消 active_pages DtoH 空泡→5%**；sbo overlap→1.5%；select_candidate_blocks/low_ratio_pair_partners；静态 EPLB。
DSPARK 双方同款（S5）；他们 draft 极便宜（14.8ms），我们 draft/verify 吃同样的膨胀地板 → **投机净负不是算法问题，是地板问题**。

## 路三：配置 diff（decode 侧）

| 维度 | das 17ms 配方 | 我们 launch_ab1023_B |
|---|---|---|
| 权重 | **INT4-W4A8**-fix_mtp | FP8-Channel |
| MoE runner | **deep_gemm**（deepgemm 2.1.0 dtk2604, low_latency） | triton fused 链 |
| 投机 MoE | deep_gemm + **deepep a2a** | triton, a2a none |
| a2a | **deepep low_latency**（LL layered opt, HDP flush off, rocm-shmem heap 3G/288QP, NUM_MAX_DISPATCH=256） | aiter AG |
| 量化 env | W4A8_MASKED_HIPC(解)/CONTIGUOUS(填)+ENGRAM load opt+FP4 paged mqa_logits/indexer | 仅 SGLANG_DSV4_FP4_DEQUANT=1 |
| norm/router | **lightop vendor 全家桶**（RMSNORM=1, topk_transform_ragged_v2, fused gate） | 自研 JIT（in-flight） |
| EPLB | **静态 init-expert-location .pt** | 无 |
| GEMM 库 | **ROCBLAS_TENSILE_LIBPATH 调优库** + NCCL_SIMPLE_CHANNELS=32 | 无 |
| 拓扑 | **PD 分离**（decode 纯解码；prefill CP8EP8+layersplit+sbo+swa-bounded-replay，mooncake/mlx5） | 统一节点（chunked-prefill 8192 混跑） |
| graph | cuda-graph-max-bs 32/64, max-running 32/64 | 16/16 |
| 栈 | sglang-das 0.5.18 + **lightop 0.6.0 + deepgemm 2.1.0** | v0522 树 + adapter（**镜像同代：dtk26041/torch2.11/HIP6.3.26113，与 das 调试节点同驱动同 userspace**） |
| 注意力 | flashmla_sparse 家族 | flashmla_sparse（**同款**） |

共同点：DSPARK S5、flashmla_sparse、DP8EP8、dp-lm-head、disable-radix-cache。差距全在栈代 + 配方。

## 行动阶梯（300ms → 17ms 的台阶）

1. **P0 消拍**（×2.8）：HSA_KERNARG_POOL_SIZE 试验 → 驱动/DTK 对齐 dtk2604 → 盒级工单。拿到 ~110-145ms。
2. **P1 地板+kernel 数砍半**（→~40-70ms）：rmsnorm/router/topk/mqa_logits 对齐 vendor 地板（我们 JIT 融合战役同向；das 数字给了每 op 目标线）。
3. **P2 decode MoE runner**：triton → deepgemm low_latency（依赖 torch2.11/dtk2604 栈）。
4. **P3 INT4-W4A8 权重 + FP4 paged logits 路径**（模型同族现成；依赖 lightop）。
5. **P4 PD 分离/decode 节点隔离 + 静态 EPLB + rocBLAS 调优库 + NCCL_SIMPLE_CHANNELS**。

P2/P3/P4 都押在 lightop 0.6.0 + torch 2.11 + dtk2604 这一代栈上（wheel 公开：pypi.sourcefind.cn/das_nightly/dtk2604-rc4/）——**栈代差是独立决策点**。P0/P1 在现栈上就能拿。

材料：tmp_fetch/das_cookbook_raw.txt（737KB 178 屏原样）→ das_cookbook.txt（185KB 去重线性）；das_prof/ trace；本地仓 E:\GITS\sglang-das（=E:/GITS/sglang worktree，分支 hygon/feat/20260920_deepseek-v4.1）。

---

## 10-23 增补（修正 + 用户输入吸收）

- **prof 量化档实证 = FP8**（per_token_quant_fp8 / rocBLAS F8BS channelwise / deepgemm kGroupGemm fp32 累加，零 int4 标记；形状 5120×2304 与我方同款）⇒ das 在 bw1000 **FP8 就到 66ms/步（TPOT 17ms 级）**，INT4 是后续优化。量化布局仍有差：das dense 层 channelwise（Channel-FP8 checkpoint）vs 我方 fp4→fp8 [128,128] block-wise 全链（FP4_DEQUANT=1）——decode GEMM 微秒级下影响小，按用户定向暂缓对齐（roadmap P3），台阶分解与量化正交。
- **「我方旧 userspace」说被推翻**：dsv41f-dspark 镜像即 dtk26041/torch2.11/HIP6.3.26113，与盒上健康容器同代同驱动；das 团队也在 nmz 系调试。残余差分=**env + 框架代码构成（v0522+adapter vs 0.5.18-das）**。
- env 侧已排空：expandable_segments 平台不支持（HCU torch 硬拒，das 配方里那行同样是摆设）；镜像烤 env 差异仅 OMPI（惰性）+NCCL_TOPO_FILE（launch 已 unset）；ROCSHMEM_/NCCL_/MC_ 系只 engage deepep 路径（我们不跑）；das PD 配方本无抗拍旋钮。**可测残余=HSA 层两钮**：HSA_FORCE_FINE_GRAIN_PCIE=1（das 4 卡配方有）、HSA_KERNARG_POOL_SIZE=8M（自提）。
- 代码构成嫌疑（idle 也拍⇒驻留态触发）：engram 设备表 ~23.6G/卡、ROCM_HOME JIT shim、DSA topk v2/adapter 常开件。
- **已 staged 四臂**（nmz18 /root/launch_ab1023_{A,B,C,D}.sh）：A 基线 / B kernarg / C fine-grain-pcie / D 双开；每臂=boot+4s idle 探测+bench，~40 分钟。
