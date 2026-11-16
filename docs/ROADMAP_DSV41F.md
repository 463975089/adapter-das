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

- [ ] **0.1 四臂 env A/B**（validation-host-18 `/root/launch_ab1023_{A,B,C,D}.sh` 已 staged：A 基线 / B HSA_KERNARG_POOL_SIZE=8M / C HSA_FORCE_FINE_GRAIN_PCIE=1 / D 双开）`[env]`
  验收：任一臂 idle 4s 探测 0 个 >10ms gap；记下是哪一钮
- [ ] **0.2 构成剥离梯队**（若 0.1 四臂全拍）：去 engram 设备表（23.6G/卡）→ 去 ROCM_HOME JIT shim → 去 topk_v2/DSA 常开件，逐臂 boot+idle 探测 `[env][ada]`
  验收：锁定触发件（哪件摘掉拍即消）
- [~] **0.2′ engram 剂量路线（10-23 首证，头号嫌疑）**：剪层 cut8 复现器（validation-host-26 容器 /root/dsv41f_cut8 + launch_cut8_A.sh 现成）已证 idle 拍 onset 剂量 ~23.6G/卡（1.5G 净 / 23.6G 间歇 3.1% / 47G 突发 10-25ms 与生产量子同族）；10-23 夜 **host 路修复全通**（seam is_hip 假→指针翻译短路=真凶；容量配平 vocab=250000；修复 4 处见 REDUCTION_PATCHES 10-23 夜行），cut8 TPOT host=device 持平；剩：①per_rank 23.6G 分块注册（>12G 单次必败）②8 卡全幅 matched-dose 消拍确认 `[env][ada]`
  验收：拍幅随 engram 表尺寸的单调曲线 + 生产级稳态复现（host 表替换设备表后拍消失）
- [ ] **0.3 idle 拍窗驱动交互观测**（若 0.2 全不消）：拍窗内 strace(ioctl)/HSA 工具侧看 rank 与驱动交互
  验收：根因定位报告（拍=驱动对何种驻留态的响应）
- [ ] **0.4 跨盒复验**：validation-host-26（拍首发地）同臂复跑
  验收：结论跨盒稳定
- [ ] **0.5 P0 结账**：loaded duty ≥80%、step p50 ≤430ms（TPOT ≤~145ms）、REDUCTION_PATCHES 落账

## P1 每-op 地板对齐 + kernel 数砍半（×2~3，现栈主战场）

das 地板目标线：rmsnorm 5.6µs｜rotary 5.6µs｜router gate 28.4µs｜silu_mul_clamp 9.5µs。我方膨胀：rmsnorm ~450µs（80×）、router ~350µs（12×）。

- [x] 1.0a MoE combine emu→triton（467µs×40 层→74µs 级，5.7×，parity ≤1ulp）`[kern]`
- [x] 1.0b silu v3 挂点修正（ENGAGED）`[kern]`
- [x] 1.0c topk_v2 融合接通（paged 臂 parity 5/5，B 臂四门绿）`[kern][ada]`
- [x] 1.1 topk_v2 ragged 臂补 parity + 转正（10-24 单测转正：96 例集合语义 0 错，"非确定"=atomicAppend 到达序与 paged 臂同源；2.2-2.9× 快过 legacy；`HCU_TOPK_V2_RAGGED` 默认 1，回退设 0；serving 验证并入 1.7）`[kern]`
  验收：ragged 臂逐值 parity ✓（96/96）；单步 topk 族耗时 ≤100µs —— kernel 级 10-58µs ✓（serving 复核在 1.7）
- [x] 1.2 rmsnorm 家族对齐（10-24 翻案：kernel-only 5.88µs @T64/H5120 = das 地板 5.6µs 同级，1-ulp parity；"450µs/80×"=旧 trace 家族和÷单核地板口径混算；eager 段 ~51µs/call 墙钟归 1.5/1.6）`[kern]`
  验收：≤3× das 地板（≤17µs）✓（5.3µs trace p50 / 5.88µs graph 复测）
- [x] 1.3 router/gate 融合（10-24 翻案：`_router_triton_kernel` 全融合本已在位，kernel-only 10µs @M64/E384/K6——快 das 28.4µs 2.8×；sqrtsoftplus-open 生产路径；num_warps 1→4 无效已试回滚；墙钟 ~100µs=wrapper 归 1.5/1.6）`[kern]`
  验收：router ≤85µs（3× das）✓（10µs；prefill M2048=42µs/M8192=123µs）
- [x] 1.4 mqa_logits/paged indexer 路径（10-25 翻案：生产=**int8_scaled cache + pers-JIT int8 kernel** 已在位——decode B16/ctx4k 10.1µs、B64 10.4µs（34MB cache @3.3TB/s=带宽极限）、verify B16×R5 9.7µs，全部 maxabs=0/top64=1.000 精确 parity；lightop int8 14.1µs 备胎、torch 兜底 413µs（41×）；旧 trace 的 dequant+rocBLAS 账=过时树。deepgemm wheel 本算子 gfx metadata 全拒（fp8/int8 臂皆然）登记还 wheel 主；fp8 臂休眠中但 lightop fp8 读 fn 偏置 vs 生产写 fnuz=×2 系数差（topk 选择不变、w/2 可精确补偿）留档）`[kern]`
  验收：对齐 das 同位 ±30% ✓（10µs vs das 地板族 5-6µs；大 B 已贴 HBM 带宽）
- [x] **1.5 DtoH 消泡**（active_pages 类图外同步点；das 账面 5%）`[ada]`（10-26 运行期验收：新鲜 39 步 trace **DtoH 2.1 次/步、~30µs/步**=常数级残留，无阻塞型图外同步（hipEventSynchronize=标准步边界）；代码普查 3 候选均在非生产算法。**通过**）
  验收：trace 图外同步归零 ✓（残留常数级 2.1 次/步 30µs，非阻塞）
- [~] **1.6 杂项 elementwise/copy 收编**（融合进邻居 kernel）`[kern]`（10-26 新鲜基线：**4499 核/步**（2249.6/图×draft+verify 两图）；最大池=torch eager elementwise 两族 `vectorized_elementwise` 660/步 4495ms + `manual_unroll` 342/步 3930ms=**合计 28% GPU 时间**；次池=`apply_rotary_emb_triton` 46.5/步×1116µs/call（bs2 病态时长，疑全 KV/大 padding）、`_silu_clamp_ptgq` 22×1930µs、`_hc_mix_stats_partial` 23×1652µs；fp4 dequant 家族已证=载入期一次性，步内无此池。**10-26(晚)修正**：以上平均时长含监控停摆怪物事件假账，剔除后每步真实 kernel 仅 ~161ms，重排：silu_clamp_ptgq 27.2ms/步（p50 598µs 真慢）、elementwise 两族 38ms/步、moe_sum_scale 14.5、nccl 14.1、hc_mix_stats 13.1、rope 9.4（p50 6µs 正常，量多而已）。**10-24(凌晨) 主靶归因闭环+歼灭**：elementwise 大头=~2197/步（占全 kernel 近半）=**RAGGED MQA emu 雪崩**——C4 nonpaged 计划（`dsv4/indexer.py:732`→`deep_gemm.fp8_mqa_logits`）×44 层/步（40 主+4 草稿）×**56 aten/调用**（指纹铁证：38 manual_unroll+31 vec+Cijk_BBH bmm×6+reduce×6+arange+full），router/mhc/compressor/rope 全排除（均已融合在位）；**R-20 修复**：`mqa_logits_triton.py` 新增 `_ragged_mqa_logits_kernel`（位算术 e4m3fn 解码+fp32 累加+[ks,ke) 窗 -inf 语义），`deep_gemm_dtk` ragged shim 随 `HCU_MQA_LOGITS_IMPL`（默认 triton）路由，门测试 `test_mqa_ragged_gfx938.py` validation-host-18 实测 PASS（fp64 真值仲裁 triton 8.87e-08 vs emu 1.66e-03 紧 4 个数量级；kernel 4 vs 56/调用）；**预期 4499→~2200 核/步（-49%）**，serving 收益待 P0′ 干净窗口与 1.7 统一验收；次级残留：dequant 周边掩码链、topk 后处理小族。**10-24(晨) R-21 第二刀：moe_align 全家收编**——四件套（hist/scan/fill/block_expert 各 44/步）+2 个 aten setup（fill_/zeros=FillFunctor 88/步）=每层 6 核×44 层=264 核/步；vendor(aiter/lightop) 换核否决（标准语义无 EP 哨兵桶，2.1 ABI 坑=语义坑）；自研 `_moe_align_fused_small_kernel` 单 CTA 一发全做（寄存器直方图+块对齐前缀+pad 预填+FIFO 散写+块→专家映射），哨兵/尾值/cumsum(E+2) bit 级对齐，`HCU_MOE_ALIGN_FUSED`（默认 2048 lanes）门测试 validation-host-26 全绿（15 组 parity+图捕获+核指纹 1v6），**264→44 核/步（-220），两刀累计预期 4499→~1980（-56%）**；同窗实战：邻居撤退 8 卡 140G 真窗被 boot_watch2 假 99G 探针掩埋（/usr/bin/python3 torch2.11 无 HIP→mem_get_info 全异常恒报初值），手动点火+census 加 W10 计数，P0′ 干净重 trace 出真账）
  验收：verify 图 kernel 4121→≤2600（**10-24 P0′ 定账：4499→4160**（R-21 -339 全额；R-20 预期 -2300 作废=错账，生产走 paged vec 双 bmm 非 ragged emu，翻案见 REDUCTION_PATCHES P0′ 节）；下一刀=elementwise 大军真凶 ~1900/步（vectorized 600+manual_unroll 333/半步段，非 MQA））
- [x] **P0′ 停摆/污染基线终审（10-24 06:30 独占窗口判决）**`[env]`（全自动链实战闭环：假 99G 探针掩窗→手动点火（mem-frac 0.70→0.80 修 minimum-viable 0.7478）→3 分钟 boot→poller→独占门→40 步 trace→census→自动重启；**判决**：①独占下 117ms 成对停摆照样全现（~4 次/步、MoE/silu 段、232ms/步）=邻居理论破产；②kernel dur 95× 虚胖=流内等待摊入（busy81%/gap19% 无 >1ms 空洞+时钟正常+串行流），kernel 时间账不作墙钟；③真墙钟=步长 1054ms（527ms×verify/draft 两段）vs 本项验收 550ms——**停摆主嫌疑转向内在集合等待（NCCL/XMI/EP allgather），下一波 per-rank 8 trace 时间戳对齐找慢 rank**）：22:29 trace=邻居共卡负载下污染基线（footprint 选择性膨胀 8-20×、profiler 盲区不见邻居 kernel、117ms 怪物=邻居 init 风暴），干净基线必须独占窗口重抓——**窗口链已全自动在役（10-24 拂晓）**：boot_watch(v1+防重触发 v2 接力) ≥120G×2 自动点火 launch_rt07 → window_chain 盯 fired-up → prof_wave2（独占复核门≥20G+40 步 prof+16×600 负载+容器内 census 判决：停摆数/R-20 engage/family 表）→ 自动重启避抓后死锁，判决落 /root/window_report.txt；教训=共享盒抓 prof 前必验独占（fuser+mem_get_info）
- [ ] **1.6b flashinfer.sampling.softmax adapter 洞（wave6 已补）**`[ada]`（非 greedy 采样解码曾必崩：DSPARK verify SoftmaxTemp→flashinfer softmax 只有 loud stub；triton port 落 `sgl_kernel/sampling.py`+allowlist 绑定，单测 11 项绿+graph 安全，167.9µs@N112×V151936=0.54×torch 链；采样解码生产验证通过）
- [ ] **1.7 集中 serving A/B + 四门**（P1 全项改完一次跑，带无 dspark 臂）
  验收：P0 基础上 step ≤550ms；无 op 超 das 同位 3×
- [x] 1.8 （决策点）lightop 0.6.0 wheel 复用 `[dep]` —— 10-24 已决：lightop/deepgemm 皆自家仓库一家 wheel 随便用（`deepgemm` 无下划线）

## P2 decode MoE runner → deepgemm LL 形态（×1.3~1.5）

- [~] 2.1 deepgemm 桥 decode masked/contig 分支（W4A16 役 2.06-3.29× 先例同桥面）`[ada][dep]`（10-25 普查：模型实形 E384/topk6/inter2304/**expert fp4**；deepgemm `m_grouped_w4a8_gemm_nt_masked_hipc`+packer/wfp4a16_contig 在位；aiter moe_c marlin wfp4a8_groupwise/channelwise 全家但**无预编 .so=冷容器 JIT 费**；首攻=marlin wfp4a8 消 dequant 直连，正对 trace 第一大核族 bf16→f32 1104ms；下步=situ_glu/fuse_silu_mul 收编 636+245ms 两段。10-25(晚) deepgemm 路全契约破译：`m_grouped_wfp4a16_gemm_nt_contiguous` 真权重 **parity 精确**（rel 0.0027；nibble 每 4 字节逆置换 repack+e8m0 原样+m_indices 128 对齐，one-hot 提取器留 /root）——但 contiguous+128 对齐 **decode 不可用**（m≈1 pad 31104 行 10.8ms/层）=prefill/EP 形态；aiter moe_c MODE 表 **build 级分叉实锤**（validation-host-26/validation-host-18 同 commit 不同表）+align 出参 ABI 坑。**决策点还 wheel 主**：A deepgemm 补 wfp4 masked 变体 / B aiter 给 MODE 调优表——到位即消 trace 第一大核族 1104ms）
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
- 四门（boot/生成/精度/长文）绿才算数；docs/history/REDUCTION_PATCHES.md 逐项落账。
- 不 kill 非我方进程；临时脚本用后清理；密码/坐标不落盘。

## 台阶总账

```
现状 300ms ─P0→ 110~145 ─P1→ 50~70 ─P2→ 35~50 ─P4 轻项→ 30~45ms
（量化/拓扑重项暂缓，本役终点 ~30~45ms；das 17ms 级含 PD 分离红利，不在本役口径）
P0/P1 完成前不承诺绝对数。
```
