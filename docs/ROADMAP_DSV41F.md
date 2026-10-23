# dsv41f 性能追赶 Roadmap（勾选式）—— 我们(v0522+adapter) vs sglang-das

> 用法：完成一项把 `- [ ]` 改 `- [x]`；每项自带验收线；集中 serving A/B 与账本纪律见文末。
> 定格于 das 2026-10-20 prof + 10-22 cookbook（das 仍在动，P5 收割机制跟进）。
> 修改面图例：`[env]` 环境变量/启动参数 ｜ `[kern]` sgl_kernel JIT/自写 kernel（自有库，自由）｜ `[ada]` adapter needle/钩子（受限）｜ `[dep]` 引入依赖 wheel（决策点）。

## 口径

| | 我们（统一节点） | das（bw1000 PD decode） |
|---|---|---|
| 模型 | dsv41f 同家族 | 同家族 |
| 量化 | **fp4→fp8 [128,128] block-wise 全链**（FP4_DEQUANT=1） | Channel-FP8 checkpoint：dense 层 channelwise（rocBLAS 实证）+ 专家 deepgemm 块形 |
| 投机 / 注意力 | DSPARK S5 / flashmla_sparse（**双方同款**） | 同款 |
| 步时 / TPOT | p50 925ms（~3 接受/步）/ ~300ms @BS16 | 66.3ms busy（~3.9 接受/步）/ ~17ms @BS32 |
| GPU duty | ~36%（拍吃 64%） | 86.9% 零停摆 |
| 单步 kernel | verify 4121 / draft 255 | verify 2196 / draft 206 |

台阶：**300ms ─P0消拍×2.8→ 110~145 ─P1地板+砍核×2~3→ 50~70 ─P2 deepgemm LL→ 35~50 ─P4轻项→ 30~45ms**。逐级验收不跳级。量化对齐与拓扑重项（PD/EPLB/sbo）按用户定向**暂缓**；台阶数字与量化正交。

---

## P0 消拍（×2.8，一切前提）

- [ ] **0.1 四臂 env A/B**（nmz18 `/root/launch_ab1023_{A,B,C,D}.sh` 已 staged：A 基线 / B HSA_KERNARG_POOL_SIZE=8M / C HSA_FORCE_FINE_GRAIN_PCIE=1 / D 双开）`[env]`
  验收：任一臂 idle 4s 探测 0 个 >10ms gap；记下是哪一钮
- [ ] **0.2 构成剥离梯队**（若 0.1 四臂全拍）：去 engram 设备表（23.6G/卡）→ 去 ROCM_HOME JIT shim → 去 topk_v2/DSA 常开件，逐臂 boot+idle 探测 `[env][ada]`
  验收：锁定触发件（哪件摘掉拍即消）
- [~] **0.2′ engram 剂量路线（10-23 首证，头号嫌疑）**：剪层 cut8 复现器（nmz26 容器 /root/dsv41f_cut8 + launch_cut8_A.sh 现成）已证 idle 拍 onset 剂量 ~23.6G/卡（1.5G 净 / 23.6G 间歇 3.1% / 47G 突发 10-25ms 与生产量子同族）；10-23 夜 **host 路修复全通**（seam is_hip 假→指针翻译短路=真凶；容量配平 vocab=250000；修复 4 处见 REDUCTION_PATCHES 10-23 夜行），cut8 TPOT host=device 持平；剩：①per_rank 23.6G 分块注册（>12G 单次必败）②8 卡全幅 matched-dose 消拍确认 `[env][ada]`
  验收：拍幅随 engram 表尺寸的单调曲线 + 生产级稳态复现（host 表替换设备表后拍消失）
- [ ] **0.3 idle 拍窗驱动交互观测**（若 0.2 全不消）：拍窗内 strace(ioctl)/HSA 工具侧看 rank 与驱动交互
  验收：根因定位报告（拍=驱动对何种驻留态的响应）
- [ ] **0.4 跨盒复验**：nmz26（拍首发地）同臂复跑
  验收：结论跨盒稳定
- [ ] **0.5 P0 结账**：loaded duty ≥80%、step p50 ≤430ms（TPOT ≤~145ms）、REDUCTION_PATCHES 落账

## P1 每-op 地板对齐 + kernel 数砍半（×2~3，现栈主战场）

das 地板目标线：rmsnorm 5.6µs｜rotary 5.6µs｜router gate 28.4µs｜silu_mul_clamp 9.5µs。我方膨胀：rmsnorm ~450µs（80×）、router ~350µs（12×）。

- [x] 1.0a MoE combine emu→triton（467µs×40 层→74µs 级，5.7×，parity ≤1ulp）`[kern]`
- [x] 1.0b silu v3 挂点修正（ENGAGED）`[kern]`
- [x] 1.0c topk_v2 融合接通（paged 臂 parity 5/5，B 臂四门绿）`[kern][ada]`
- [ ] **1.1 topk_v2 ragged 臂补 parity + 转正**（mbtopk 57/步残账）`[kern]`
  验收：ragged 臂逐值 parity；单步 topk 族耗时 ≤100µs
- [ ] **1.2 rmsnorm 家族对齐**（fused norm+qkv，fp32 acc 单舍入）`[kern]`
  验收：≤3× das 地板（≤17µs）
- [ ] **1.3 router/gate 融合**（fused gate sqrtsoftplus 等价）`[kern]`
  验收：router ≤85µs（3× das）
- [ ] **1.4 mqa_logits/paged indexer 路径**（das 单算子 2-3×=端到端 4%）`[kern]`
  验收：对齐 das 同位 ±30%
- [ ] **1.5 DtoH 消泡**（active_pages 类图外同步点；das 账面 5%）`[ada]`
  验收：trace 图外同步归零
- [ ] **1.6 杂项 elementwise/copy 收编**（融合进邻居 kernel）`[kern]`
  验收：verify 图 kernel 4121→≤2600
- [ ] **1.7 集中 serving A/B + 四门**（P1 全项改完一次跑，带无 dspark 臂）
  验收：P0 基础上 step ≤550ms；无 op 超 das 同位 3×
- [ ] 1.8 （决策点）lightop 0.6.0 wheel 复用 `[dep]` —— 许可若给，1.2/1.3/1.4 大幅捷径

## P2 decode MoE runner → deepgemm LL 形态（×1.3~1.5）

- [ ] 2.1 deepgemm 桥 decode masked/contig 分支（W4A16 役 2.06-3.29× 先例同桥面）`[ada][dep]`
  验收：单测 parity 矩阵全绿
- [ ] 2.2 deepep LL engage（deep-ep-shca 1.1.1 DTK 移植已在库；解 spec hook 拒 deepep 与 32 并发边界）`[ada]`
  验收：LL 稳定跑满 max-running
- [ ] 2.3 a2a 对照集中 A/B：aiter AG vs deepep LL（ROCSHMEM 参数照 das 配方）`[env]`
  验收：decode MoE 段对齐 das 同位 ±30%；GEMM 占 busy ≥40%

## 暂缓桶（用户 2026-10-23 定向：先对齐其他差异）

量化对齐：
- [ ] 3.1 dense 层 blockwise→channelwise 布局（对齐 das rocBLAS channelwise 快核路）`[kern]`
- [ ] 3.2 INT4-W4A8 权重路径（模型现成 fix_mtp；W4A16 桥改 W4A8 或 aiter 表；先过 R-10 勘误的 scheme 坑）`[ada][kern]`
- [ ] 3.3 精度成本评估后决定是否启动（收益主要在吞吐/容量/prefill，不在 TPOT）

拓扑重项：
- [ ] X.1 PD 分离（decode 节点隔离 + mooncake/IB；需第二节点）
- [ ] X.2 静态 EPLB（自做专家分布统计 .pt → init-expert-location）
- [ ] X.3 sbo 近似（共享专家与 dispatch 双流 overlap，das 账面 1.5%）

## P4 轻项（拓扑类重项已按用户定向暂缓）

- [ ] 4.3 rocBLAS 调优库（ROCBLAS_TENSILE_LIBPATH）+ NCCL_SIMPLE_CHANNELS=32 `[env]`
  验收：dense GEMM 段对齐 das 同位 ±20%

## P5 das 持续优化收割（滚动）

- [ ] 5.1 cookbook 每 1-2 周收割（178 屏 harvest 流程已固化），新点按「收益/修改面」折入本图
- [ ] 5.2 已知其管线：mqa_logits(4%)、DtoH 消泡(5%)、sbo(1.5%)、EPLB、GEMM tuning、decoder-swa-bounded-replay、layersplit

---

## 决策点（拍板后勾掉）

- [ ] D1 lightop/deepgemm wheel 复用许可（同公司资产，问 das 团队/袁元；P1/P2 加速器）
- [x] D2 PD 分离是否在范围 —— 已拍板（2026-10-23）：先不做（连同 EPLB/sbo 入暂缓桶）
- [ ] D3 8 卡机时窗口（P0 四臂 ~3h；P1 集中 A/B 每轮 ~1h）
- [ ] D4 P0 落地前 A/B 均带无 dspark 臂（裸 decode 273ms 基线，省机时）

## 纪律（沿用现役）

- 单件 parity（ulp 语义门）+ 微基准先行；serving 数据只集中跑（用户定向）。
- 四门（boot/生成/精度/长文）绿才算数；REDUCTION_PATCHES.md 逐项落账。
- 不 kill 非我方进程；临时脚本用后清理；密码/坐标不落盘。

## 台阶总账

```
现状 300ms ─P0→ 110~145 ─P1→ 50~70 ─P2→ 35~50 ─P4 轻项→ 30~45ms
（量化/拓扑重项暂缓，本役终点 ~30~45ms；das 17ms 级含 PD 分离红利，不在本役口径）
P0/P1 完成前不承诺绝对数。
```
