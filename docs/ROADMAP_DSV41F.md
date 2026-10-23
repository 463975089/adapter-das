# dsv41f 性能追赶 Roadmap —— 我们(v0522+adapter) vs sglang-das

> 2026-10-23 立。目标：说清与 das 的每一级差距、在我们「零修改上游、adapter/自有 kernel 层受限修改」约束下怎么拿回、按什么顺序。
> das 是侵入式 fork 且仍在持续优化（目标会动）；本图按 2026-10-20 prof + 10-22 cookbook 定格。
> 账本：REDUCTION_PATCHES.md；度量工具全部现成（bench_step 日志间隔 / stall_detect / trace 普查脚本族）。

## 0. 口径与基线

| | 我们（nmz18 统一节点） | das（bw1000 PD decode 节点） |
|---|---|---|
| 模型/量化 | dsv41f FP8-Channel（**同模型同量化**，形状 5120×2304 同款） | FP8-Channel（prof 实证；INT4 是其后续优化） |
| 投机 | DSPARK S5（同款） | DSPARK S5 |
| 注意力 | flashmla_sparse 家族（同款） | 同款（fp8 sparse mla） |
| 步时 | p50 925ms（~3 接受/步） | 66.3ms busy（draft 14.8 + verify 51.5，~3.9 接受/步） |
| TPOT | ~300ms @BS16 | ~17ms @BS32 |
| GPU duty | ~36%（拍吃 64%） | 86.9%（零停摆） |
| 单步 kernel | verify 4121 / draft 255 | verify 2196 / draft 206 |

总差距 ~17×，分解四级台阶：**拍 ×2.8 → 每-op 地板+kernel 数 ×2~3 → MoE runner/权重 ×1.3~1.5 → 拓扑(PD/EPLB/调优库) ×1.2~1.5**。逐级验收，不跳级（下一级的兑现率被上一级压着）。

修改面图例：`[env]` 环境变量/启动参数 ｜ `[kern]` sgl_kernel JIT/自写 kernel（我们自己的库，自由）｜ `[ada]` adapter needle/钩子（受限修改）｜ `[dep]` 引入依赖 wheel（决策点）｜ `[fork]` 需动上游框架（尽量避免）

---

## P0 消拍（×2.8 —— 一切的前提）

拍：GPU 全局同步停摆 ~116ms@181ms，idle 服务也拍（duty 48.6%），驻留态触发。已证：非盒级驱动（同盒 6 个外来栈驻留零停摆）、非 launch 风暴/单纯大驻留（单因素不复现）、我们容器与外来健康容器同代 userspace（dtk26041/torch2.11）。残余差分 = env + 代码构成。

| # | 动作 | 修改面 | 预期 |
|---|---|---|---|
| 0.1 | 四臂 A/B：`launch_ab1023_{A,B,C,D}.sh`（A 基线 / B HSA_KERNARG_POOL_SIZE=8M / C HSA_FORCE_FINE_GRAIN_PCIE=1 / D 双开），每臂 boot+4s idle 探测+bench | `[env]` 已 staged | 任一臂 idle 0 gap |
| 0.2 | 若四臂全拍 → 构成剥离梯队：去 engram 设备表（→host 表或关）→ 去 ROCM_HOME JIT shim → 去 topk_v2/DSA 常开件，逐臂定位触发件 | `[env]`/`[ada]` | 锁定触发件 |
| 0.3 | 若仍不除 → idle 拍窗内对 rank 做 strace(ioctl)/HSA 工具侧观测，定位驱动交互点 | 工具 | 根因报告 |
| 0.4 | 跨盒验证：nmz26（首发地）同臂复跑，确认非单盒 | `[env]` | 结论稳定 |

**验收**：idle 探测 0 个 >10ms gap；loaded duty ≥80%；step p50 ≤ 430ms（TPOT ≤ ~145ms）。
**不做的**：任何 kernel 优化在此之前不集中跑 serving（兑现率 36%）。

## P1 每-op 地板对齐 + kernel 数砍半（×2~3 —— 现栈可做，融合战役主战场）

das 地板（实测目标线）：rmsnorm 5.6µs｜rotary 5.6µs｜router gate 28.4µs（vendor fused sqrtsoftplus）｜silu_mul_clamp 9.5µs｜fp4_paged_logits（单算子优化后 2-3×）。
我方膨胀位（trace 归因账）：rmsnorm ~450µs（80×）｜router ~350µs（12×）｜mbtopk 57/步（topk_v2 已接通，paged 臂 parity 5/5）｜combine 已修（467µs×40层 → 74µs 级，5.7×）｜silu v3 已挂。

| # | 项 | 修改面 | 目标 | 状态/先例 |
|---|---|---|---|---|
| 1.1 | rmsnorm 家族对齐（fused norm+qkv 路径，fp32 acc 单舍入语义） | `[kern]` | ≤3× das 地板 | main_norm_rope.cuh JIT 家族先例 |
| 1.2 | router/gate 等价融合（topk_v2 全臂转正 + fused gate sqrtsoftplus 等价） | `[kern]`+`[ada]` | router ≤85µs | topk_v2 paged 臂已验；ragged 臂补 parity |
| 1.3 | mqa_logits/paged indexer 路径（das 单算子 2-3× = 端到端 4%） | `[kern]` | 对齐 das 同位 | das fp4_paged_logits 形态参考 |
| 1.4 | DtoH 消泡（active_pages 类同步点，das 账面 5%） | `[ada]` | 图外同步归零 | trace 同步点普查工具已有 |
| 1.5 | 杂项 elementwise/copy 收编（融合进邻居 kernel） | `[kern]` | 单步 launch 数 -30% | combine/silu 同打法 |
| 1.6 | （决策点）lightop 0.6.0 wheel 复用——公开 nightly 挂着，若许可直接 import 其算子，P1 大幅捷径 | `[dep]` | 地板一步到位 | 需问 das 团队/袁元 |

**工作流**（用户定向，不变）：单件 parity（含 ulp 语义门）+ 微基准 → 全部改完集中一次 serving A/B + 四门。
**验收**：verify 图 kernel 4121 → ≤2600；无任何 op 超 das 同位 3×；在 P0 达标基础上 step ≤ 550ms。

## P2 decode MoE runner：triton → deepgemm 形态（×1.3~1.5）

das：decode 主/投机 MoE 全 deep_gemm（deepgemm 2.1.0 low_latency，masked/contiguous grouped GEMM，LL 分层 a2a）。我方现况：triton fused 链，decode GEMM 本体是 µs 级（结构账：das GEMM 占 busy 56%，我们胶水主导——P1 做完后 GEMM 占比自然上升，此时 P2 才有肉）。

| # | 项 | 修改面 | 备注 |
|---|---|---|---|
| 2.1 | deepgemm 桥接 decode masked/contig 分支（W4A16 役 2.06-3.29× 先例，同桥面） | `[ada]`+`[dep]`(deepgemm wheel 公开) | 先单测 parity 矩阵 |
| 2.2 | deepep LL engage：deep-ep-shca 1.1.1 DTK 移植**已在库**（dsv4f bf16 专家路已通）；需解 spec hook 拒 deepep 的 32 并发边界 | `[ada]` | LL buffer 定长：SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 参照 |
| 2.3 | a2a 对照：aiter AG vs deepep LL（同批集中 A/B，不单独跑） | `[env]` | ROCSHMEM_ 调参数照 das 配方 |

**验收**：decode MoE 段耗时对齐 das 同位（±30%）；GEMM 占 busy ≥40%。

## P3 INT4-W4A8 权重路径（×1.1~1.3，低优先）

das 的 INT4 是带宽/容量优化（decode GEMM 微秒级时 TPOT 收益有限，主要兑现在吞吐/容量/prefill）。我方无 W4A8_MASKED_HIPC（das 私有），自修路径 = W4A16 deepgemm 桥改 W4A8 或 aiter 表。模型现成（INT4-w4a8-fix_mtp）。**排在 P2 后**，且先过精度成本评估（R-10 勘误：上游 scheme 层有坑，channel 策略算术要自证）。

## P4 拓扑与调度（×1.2~1.5，工程量大，按需决策）

| # | 项 | 修改面 | 备注 |
|---|---|---|---|
| 4.1 | PD 分离（decode 节点隔离，mooncake/IB 传输） | `[env]`+`[ada]` | das 的 17ms 是 decode 节点口径；需第二节点+IB，用户决策 |
| 4.2 | 静态 EPLB（init-expert-location .pt，随机数据统计分布） | `[env]`+`[ada]` | 我方 topk 分布统计脚本可自制 |
| 4.3 | rocBLAS 调优库（ROCBLAS_TENSILE_LIBPATH 按 box 调优包）+ NCCL_SIMPLE_CHANNELS=32 | `[env]` | 调优包 das 侧有成品（BW80_v10），格式可自制 |
| 4.4 | sbo（共享专家与 dispatch overlap，das 账面 1.5%）| `[ada]` | 侵入式特性，adapter 层用双流 overlap 近似 |

## P5 跟进 das 持续优化（收割机制）

cookbook 每 1-2 周收割一次（178 屏 harvest 流程已固化），新优化点按「收益/修改面」折算进本图对应阶段。已知其管线中：mqa_logits（4%）、DtoH 消泡（5%）、sbo（1.5%）、EPLB、GEMM tuning、decoder-swa-bounded-replay、layersplit。

---

## 台阶总账（TPOT 预期线）

```
现状 300ms ──P0 消拍──▶ 110~145ms ──P1 地板+砍核──▶ 50~70ms
      ──P2 deepgemm LL──▶ 35~50ms ──P3 INT4──▶ 30~45ms ──P4 拓扑──▶ 20~30ms（das 17ms 级）
```
统一节点（无 PD）的现实终点 ~30-45ms；要进 20ms 档必须 P4.1。P0/P1 完成前不承诺任何绝对数。

## 决策点汇总（需用户/外部拍板）

1. **lightop/deepgemm wheel 复用许可**（P1.6/P2.1 加速器）——同公司资产，问一句的事，收益最大。
2. **PD 分离是否在范围**（P4.1，需要第二节点+IB 面）。
3. **8 卡机时窗口**（P0 四臂 ~3h；P1 集中 A/B 每轮 ~1h）。
4. dspark 拍死期间投机是否暂关（裸 decode 273ms 基线跑日常验证，省机时）——建议 P0 落地前 A/B 都带无 dspark 臂。

## 纪律（沿用现役）

- 单件 parity（ulp 语义门）+ 微基准先行；serving 数据只集中跑（用户定向）。
- 四门（boot/生成/精度/长文）绿才算数；REDUCTION_PATCHES.md 逐项落账。
- 不 kill 非我方进程；临时脚本用后清理；密码/坐标不落盘。
