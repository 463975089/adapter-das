# 任务看板（requirements board）

> 协作入口：**按下方 R-xx 编号领任务**，认领后在总表状态里署名并回填进展；
> 每条需求的明细自带验收标准，照标准交付。完成且无遗留的任务及时移出本表，
> 知识沉淀进 docs/ 现有文档（models.md / ci.md / env-vars.md /
> BACKEND_CATALOG 等）。R-xx 编号全局连续不复用。
> 本文进仓；初级敏感信息已在发布时脱敏（内部节点以代号指称）。

## 在办任务总表

（按优先级排序）

| ID | 需求 | 优先级 | 状态 | 登记日 | 最近更新 |
|---|---|---|---|---|---|
| R-10 | 模型×并行方式验证矩阵（服务器模型资产盘点先行） | P2 | **进行中（2026-10-18 首批，自跟踪）**：格 1/1b（Qwen3-INT8）红定性=上游 v0.5.21 W8A8Int8 动态 MoE scheme 仅 NPU（compressed_tensors.py:896 raise，非 HIP 特有 CUDA 同拒；dense 线性层路径绿=Qwen3.6-27B 前科，红面仅 MoE experts）→归因不修；格 3（GLM-5.3-INT8 717G）同族同因引用格 1；格 2（GLM-5.3 fp8）**硬 blocked=池内资产残缺**（2026-10-18 opandas 恢复核验：容器挂载直接生效、无空目录残留，但实跑三连取证——①GLM-5.3 目录无 tokenizer（借 glm5.3-channel-int8 完整 tokenizer 组装 overlay 配方已备 /root/g53/GLM-5.3-ovl）②**GLM-5.3 fp8 仅 22/141 片无 index**（命名 of-00141 单片 5G，110G/预期 700G）③替代源 **GLM-5-FP8 亦残缺 139/142 片**；唯一完整的 glm5.3-channel-int8 141/141=CT-int8 即格 3 同因红；**勘误二更：fp8 全量≈700G 级 TP4 装不下须 TP8**——运维补片进遗留跟踪）；**盘点勘误：GLM-5.3 实为原生 fp8 block 非 bf16**（与 glm53f 同格式族）；容器按评审方指示重建（**新镜像 sf_b075-20261018-0048 轮子 1013→1018+公共盘只读挂载+清 222G**）——**回归四门全绿收官**（锚点/并发 6/6/长文 5180tok 双事实对；单流 7 tok/s=10-16 基线 2×）；；**乙档首格判负（2026-10-19）**：hygon/GLM-5.3-Channel-INT4-w4a16 391G 完整可读，auto boot 崩=上游判 W4AFP8 scheme 而该 scheme 仅支持 GROUP 策略、channel（g=-1）负维崩——实际面 int4×fp8-dynamic 仅路由专家量化；命名勘误三更+「moe_wna16 现成可跑」作废（models.md 2026-10-19 行）**镜像升级新雷=tilelang 0.1.15 ThreadSync 编译回归**（mHC 融合内核硬错，0.1.12 同条件仅 Warning）→评审定规跟版本保融合→钉 0.1.12（vendor tgz 留档+bootstrap §5+upstream-watch §5 登记，待报轮子团队）；tvm-ffi 递归链再爆已解（离线验证节点 拷 .so）；战报 models.md 两行+cookbook §1.5+bootstrap §0 runbook 已落 | 2026-10-17 | 2026-10-18 |
| R-11 | 自研 triton 档产线（emu 尾部三件→融合面→marlin 桥）——独立 agent 轨 | P2 | 进行中（**前置状态：首件 fa3 prefill/varlen 已合并 master（2bbdea0，2026-10-19 协调人 执行），L1 全量门 GREEN（三跑故事收口，fa_varlen 33 用例 生产节点 A/BW1102 首实跑全 PASSED）**；**fa3 varlen serving A/B 已执行 2026-10-19：判决=triton 臂守 candidate，晋级阻塞**——≤8k ctx 质量+性能平价（解码 8.27 tok/s 两臂逐字同速 30.95s×3、prefill ±10%、锚点等价、LONGCTX 事实对）；**≥11k ctx 臂 B 2/2 boot 毙服务**（emu `_attend_chunked` 单笔 28.88GiB OOM 杀树/watchdog 挂死）vs 臂 A 3/3 过；glm53f 的 q_v paged 面超 triton 合同=内核级 13× bench 本模型不兑现属预期；A/B 顺带揪出**ops 重绑丢 q_v 包参 boot 杀手**（dsa _forward_fa3 decode 捕获崩，2759711 修）；战报 models.md 2026-10-19 行。**墙修复后终验（同日 c59135b）：晋级障碍已摘除——triton 臂 11k×3 绿（11.3s=auto）+全电池绿+LONGCTX-6k 28.4→3.1s（~9×）+锚点与 auto 臂逐字一致（修复前文本微差同源 tile 敏感性）；档位守 candidate（glm53f q_v 面恒 emu，翻默认无感；待行使 triton 核的模型过 serving A/B 再晋级）**。余项：大窗/H=32 调优；BW1102 perf 复核；kpool write/flash_mla 面承接） | 2026-10-18 | 2026-10-19 |
| R-13 | 验收后遗留收编（conv1d-flake / kpool=1 风险 / aiter reduce_scatter 契约缺口 / R-09④ wheel 复评 / R-09①②③ 晋级） | P2 | 进行中（2026-10-19 协调人 登记；**#6 R-12 守卫已完成：ee836e2 已推**；**#4 paged_mqa 晋级已完成：27c5847+3848d96 已推**；**#2 10k 容量墙已完成：c59135b+战报已推**——q 行子分块击穿（峰值 22→6.8GiB、fp64 金标 0.0046、47 门绿+新钉、serving 终验 11k×3 绿+LONGCTX ~9× 彩蛋，详见明细）；conv1d-flake 有 xfail-strict 桩自动跟踪；格 2 资产残缺补片挂运维遗留；**#1 ar_ext.cu 已改判收口（2026-10-19）：不修复，废弃——aiter GraphSafe 新镜像终验 4/8-rank 15/15 全绿（图重放净）后无修复意义，坏输出亦不配当对比参照，custom AR 七删四改除役（ce7e402），serving 验金绿） | 2026-10-19 | 2026-10-19 |
| R-05 | W4A16 Marlin 内核族接入（生产模型主力量化路线的高性能档） | P3（2026-10-17 评审定维持 P3） | **实施段已落地（2026-10-19 dev（serving 轨） 自验收：桥+w4a16_moe.py+三段式 9/9+bench 2.1-3.3×，51acb72 已推；摸底段此前 协调人 核证通过）**；**e2e 试验田变更（2026-10-19 同日 10-19 探役）**：池内 392G INT4-w4a16 实为 W4AFP8 面（int4-channel×fp8-dynamic）不适用本桥与 moe_wna16——e2e 挂「等真 group-wise w4a16 checkpoint 入池」，R-05 维持候选 opt-in 不晋级——本段后状态：**候选档 opt-in 落位**（HCU_W4A16_MOE_IMPL=deepgemm 显式；auto 现答 triton），剩 serving A/B+e2e 后评审默认翻转。细节：生产节点 A GPU4 单测级三轮探针——①deepgemm w4a16 masked 族=**数值健康**：C++ low_latency_w4a16_gemm 经 w4a16_marlin_weight repack 后对独立 torch 金标准 rel=0.0022（bf16 噪声级），轮子自带 triton golden（moe_w4a16_groupwise_triton）+repack 自洽，布局全文档化（b [E,N,K//2] u8 lo_even、scale [E,N,K//32] 与激活同 dtype、约定 (q−8)·s、actual_tokens 掩码）→ **桥接首选**；②aiter 纯 triton AWQ 反量化=精确 rel=0（小端 nibble+整零点 (q−z)·s，GEMM 同解码逻辑，配 config json 待接）→ 兜底/参考；③lightop marlin w4a16 C++ 族=**不可用**——scales_zeros 格式无文档且不可复现（20+组合+受控 s=1/z=0 微探针均失配，输出幅度与注入 scale 无关）、小形状 dequant 硬 VM-fault（设备非法访存杀进程）、config 表仅 M1-16 单一 MoE 形状族、moe_gemm_marlin_w4a16 包装源码 NameError；④deepgemm moe_marlin_prefill=w8a8-fp8 族=R-04 已判死。探针资产 tmp_fetch/probe_r05_{marlin,micro,aiter,deepgemm}.py + 容器 /root 同名 | 2026-10-17 | 2026-10-19 |
| R-07 | MTP 红配置（快配置）heisenbug 攻坚——下阶段 | P3（依赖 R-06 方法学前置） | 待启动 | 2026-10-17 | 2026-10-17 |
| R-14 | 上游单测收集面扩容——模型族 + 功能面全查证收编（DSv4/GLM-Next/Qwen3Next/KimiK3/Hunyuan4/MiniMax-M3 + PD 分离/dspark/mtp/dflash/EP/hicache/mock canary） | P2 | 进行中（2026-10-19 评审两轮点名缺口，协调人 全树查证定批；**批 A 已完成（同日）**：l15_mech 段落地（gdn/hybrid_linear/mamba/swa/lightning/dsv4 六目录+**known-fail 预算绊线** passed≥90 且 failed≤42 才绿、越线/超时即红）；首跑 38F/99P/15S/173 子绿 826.58s，六族归因零静默——K1 gdn cuteDSL 编译器 `ptr<1>`≠`ptr` lowering 失配×2（平台）、K2 dsv4 JIT 元数据编不过×3（平台）、K3 l1c_evict_priority triton-ROCm 无此拷贝 op×1（平台）、K4 flash_mla `block_table=None` 桥缺×10（候补 R-11 flash_mla 面）、K5 dsv4 数值 ~20（待深查）、K6 mamba2 triton 页界×3（与 conv1d-flake 同族嫌疑）——账本落 docs/ci.md L1.5b；15 skip 全合理（14×flashinfer 平台+1 SM90 门）；第 2 轮**全门 GREEN**（gpu=0）计数复读 38/99，绊线判 PASS） | 2026-10-19 | 2026-10-19 |
| R-15 | 性能回归门（相对阈值 + 历史锚定 + 三级防线；微基准进 L1、e2e 进夜间） | P2 | 待启动（2026-10-19 评审拍板"需要"；协调人 已完成上游 perf 面查证与设计定案，见明细） | 2026-10-19 | 2026-10-19 |
| R-16 | skill/CONTRIBUTING 与门禁体系同步维护（CLI 上手守则多人化适配） | P2 | 待启动（2026-10-19 协调人 完成 skill 补丁首轮=kill-switches AR 除役定位/shadow-matrix 四档状态机/cookbook-matrix 零 env 原则/SKILL 法则 10-11+治理闭环节+仓库结构刷新——本轮即交付大半；本行跟踪剩余：CONTRIBUTING MR 门禁段与 ci.md 同步、flash_attn wheel 段落复核刷新、新同事上手路径实效检验） | 2026-10-19 | 2026-10-19 |

## 需求明细

### R-07 MTP 红配置（快配置）heisenbug 攻坚——下阶段

**背景（2026-10-17 阶段收官挂起）**
- MTP 安全配置已全绿但 6× 慢（0.58 tok/s）；快配置（开图）输出乱。
- 定位战排除图腐蚀后，真凶定性为**探针敏感 heisenbug**（加侧缓冲拷贝即
  转绿）——现有观察通道会扰动它，需换不扰动时序的取证手段。

**工作内容（前置：R-06 的干净对拍方法学）**
1. 离线重放：用已捕获的输入快照离线喂 fa3 emu / verify 路径，绕开服务态时序。
2. 时序扰动二分：延迟注入定位敏感窗口（哪一段的时序变化会掩盖/触发腐蚀）。
3. 修复后走 R-01 同款收口：Berlin 连续 PASS → e2e 四门 → 性能回收留档。

**验收标准**
1. 红配置（快配置）Berlin/四门全绿，MTP 吞吐恢复数据留档（目标 >3 tok/s）。
2. 根因有书面结论（含为何探针敏感），坑沉淀 one-pit-one-test 单测。

### R-05 W4A16 Marlin 内核族接入（生产模型主力量化路线的高性能档）

**背景**
- 生产模型大部分为 channel-wise + Marlin（W4A16）量化——硬件依据见
  长期方向区「硬件特性备忘」；das 栈有自研 marlin 实现先例。
- 现状：W4A16 路径 adapter 侧靠 triton 兜底（AWQ 需
  `--quantization moe_wna16`）可跑，但缺 marlin 级高性能内核；上游
  marlin 内核为 CUDA 专有，HCU 按四级路由找归宿
  （lightop → aiter → 自研 → triton 兜底）。

**治理选位（2026-10-17 评审对齐：adapter 轨）**
- 判据套用：上游 sglang 在 CUDA 上原生支持 marlin 量化路径（gptq_marlin /
  awq_marlin / W4A16），HCU 缺 = **行为还原 → adapter**（同族先例：channel-wise
  int8、AWQ/moe_wna16）。"动机是性能 + 用自研内核"不改变选仓
  （CONTRIBUTING §3b）；自研 marlin 内核库本体放 vendor 侧（同 lightop/aiter
  轮子），adapter 只放桥（seam + 四级路由 + 登记）。
- **分支条件（摸底必答）**：若生产 marlin checkpoint 实为 das 私有 repack
  布局（非标准 marlin 格式，仅 das 栈可加载），则"加载该格式"为上游没有的
  行为，该部分走 plugin 轨（同 slimquant 先例）。摸底时看 das 树 marlin
  权重格式即可裁定。

**摸底结论（2026-10-17 协调人 完成，das 树实读）**
- das 的 marlin 内核形态：`kernels/ops/quantization/gptq_marlin.py` /
  `moe/moe_wna16_marlin.py` 等 = **自带 `gemm/marlin/marlin.cuh` 源码经
  tvm_ffi JIT 装载**（cache_once，与 bootstrap.md §3 的 tvm-ffi 预热同机制）。
- **格式标准性（必答项）已答**：das 走标准 `gptq_marlin_repack` /
  `awq_marlin_repack`（GPTQ/AWQ → 标准 marlin 布局）——**无私有格式，
  adapter 轨定案**。`slimquant_w4a8_marlin` 为私有格式部分，维持 plugin 轨。
- **候选二（vendor 档一步到位）**：DTK deepgemm 轮子自带
  `grouped_gemm_w4a16_nt_masked_entry`（附 vendor triton ref）+
  `moe_marlin_prefill`——R-04 摸底时容器实测确认，接入评估与 das
  marlin.cuh 路线二选一（bench 定）。

**wheel 侧运行时摸底（2026-10-19 dev（serving 轨），生产节点 A GPU4 单测级）**
- **deepgemm w4a16 masked 族 = 唯一数值健康的 vendor C++ 路径（首选）**：
  `w4a16_marlin_weight`（[E,N,K//2] u8 → 低延迟 W4A16 布局）+
  `grouped_gemm_w4a16_nt_masked_entry`（→ low_latency_w4a16_gemm C++）对
  独立 torch 金标准 rel=0.0022；轮内 triton golden
  `moe_w4a16_groupwise_triton` 同分自洽。约定（探针钉死）：nibble 小端
  沿 K 打包、(q−8)·s（无显式零点）、scale [E,N,K//32] 与激活同 dtype
  （bf16/fp16 均可，须一致）、actual_tokens 行掩码。N%16==0、K%128==0。
- **aiter 纯 triton AWQ w4a16 = 精确（rel=0）**：qweight [N,K//2] 小端、
  qzeros [K/G,N//2] 沿 N 打包整零点、scales [K/G,N]、(q−z)·s；GEMM 内核
  同解码逻辑但需 tuned config json（形状覆盖待查）——兜底/参考档。
- **lightop marlin w4a16 C++ 族 = 不可用（当前轮子）**：scales_zeros
  格式无文档、20+ 候选组合与受控微探针（s=1/z=0）均无法复现（输出幅度
  与注入 scale 无关）；小形状 dequant 直接 VM-fault 杀进程；config 表仅
  覆盖 M1-16×单一 MoE 形状族；`moe_gemm_marlin_w4a16` 包装源码引用未
  定义变量（NameError）。除非厂商补文档，此路封存。
- deepgemm `moe_marlin_prefill`（w8a8-fp8 族）随 R-04 终判死，不再计。

**工作内容（摸底可随 R-04 同批做，同在 das 树）**
1. 摸底：自研 marlin 库形态/接口/dtype 覆盖（int4/int8 权重？MoE 与
   dense Linear 都覆盖吗）、das 当时的接法、上游 marlin 调用面清单
   （repack + gemm 两段）。
2. 桥接：调用面映射 + env 档位枚举登记（命名对齐
   `HCU_KDA_PREFILL_IMPL` 先例）；布局/repack 语义差异在桥内消化。
3. 单测先行（三段式）→ e2e（AWQ/GLM int4 线）→ A/B bench vs
   moe_wna16 triton 兜底 → catalog 晋级。

**验收标准**
1. 数值对拍过（vs triton 兜底与离线 dequant 参考），容差按 test-suite.md。
2. bench 证明性能收益（vs moe_wna16 triton）。
3. 三处登记齐（L0 绿）、L1 绿、e2e 四门通过。

### R-10 模型×并行方式验证矩阵（服务器模型资产盘点先行）

**背景（2026-10-17 评审拍板登记）**
- 单测绿 ≠ serving 通（校准结论）：模型×并行的真实覆盖需要实跑记录在案，
  把"官方支持的模型/并行方式理论上都能顺利验证"变成绿格矩阵。
- 服务器模型资产比预想新——生产节点 A `hygon/` 10-16 刚到一批量化矩阵（GLM
  全家 + DSV3 + MiniMax + Qwen），先盘点后排批。

**盘点结论（2026-10-17 协调人 完成，生产节点 A 主池实测 config.json/体积；16 卡池 未扫——2026-10-19 试连 <远程池>连接受限（网络层拒），补扫等隧道/窗口）**

*甲档·验证邻域内（量化格式已验证 × 架构已跑过，预期顺）：*
| 模型 | 格式/架构 | 体积/资源 | 备注 |
|---|---|---|---|
| Qwen3-30B-A3B-Channel-INT8-w8a8 | CT int-quantized × Qwen3Moe | 30G，单卡 144G 可跑 | **首选**：已验证格式（Qwen3.8-27B 同族）的更大兄弟，最快出绿格 |
| ZhipuAI/GLM-5.3（fp8 block 勘误，**池内残缺 22/141 片**） | GlmMoeDsa | 141 片发布为 ~700G 级（现存 110G），TP8×144G（**原判 135G/TP4 作废**：10-18 实测单片 5G×141） | 大 GLM-5.3 的原生 fp8 全量版替代源 GLM-5-FP8 亦残缺（139/142）；运维补片后方可排期 |
| hygon/GLM-5.3-Channel-INT8-w8a8 | GlmMoeDsa（CT int-quantized 待核） | 717G，TP8×144G 紧（90G/卡+KV），16 卡池 更稳 | 已验证格式 × 新架构 GlmMoeDsa |
| GLM-5.3-Flash（bf16，在跑） | glm5_next | 306G，TP8 | 主线基线，矩阵原点 |

*乙档·新格式/新族（一役级，各挂对应需求）：*
| 模型 | 格式/架构 | 体积 | 去向 |
|---|---|---|---|
| GLM-5.3-Channel-INT4-w4a16 | GlmMoeDsa × CT pack-quantized | 392G | **10-19 实跑判负+勘误三更**（models.md）：实为 int4-channel×fp8-dynamic（W4AFP8），上游 scheme 有但 channel 算术崩；R-05 试验田假设作废 |
| GLM-5.2-Channel-INT4-w4a8 | GlmMoeDsa × **slimquant_w4a8 私有格式** | 363G | plugin 轨（das slimquant_w4a8_marlin）；新族+私有格式双重新，立需求再排 |
| MiniMax-M2.5-bf16 | MiniMaxM2ForCausalLM | 427G | 上游有 M2 族支持基线，新版本模型验证 |
| DeepSeek-V3-w4a8-MOE-Quant | w4a8 私有格式 | 334G | 同 slimquant 轨 |
| DeepSeek-V4.1-Flash-Channel-INT4-w4a8 | w4a8 | 721G | 16 卡级资源，远期 |

*丙档·暂不可用（盘点即排除，不占排期）：*
- **空目录占位**（4KB 未下载）：Hy4-preview-Channel-FP8-w8a8、Kimi-K3-DSpark-INT4。
- **权限不可读**（他人/训练组产物）：WeLM-v4.5-80B-A3B-0813-MTP（213G，MTP 能力
  与 R-07 相关，若能协调权限再启用）、GLM-5.3-Flash-CHANNEL-INT8-w8a8-v3
  （=iter_0033200 训练 checkpoint，非发布版）。
- **Channel-FP8 族**（GLM-5.3-Channel-FP8-w8a8 等）：CT float-quantized fp8 路径
  adapter 未覆盖（已知坑，fp8 serving 层病着——单列不混入本矩阵）。

**工作内容**
1. 首批（甲档，走现有模型套件守卡机制）：Qwen3-30B-A3B-INT8（单卡最快）→
   GLM-5.3 bf16 TP4 → GLM-5.3-INT8 TP8/16 卡池；每模型 launch→生成→锚点一条龙，
   models.md 记行。
2. 并行维度：TP 主线已通；首批顺手补一档 **EP**（Qwen3-30B-A3B-INT8 双卡 EP）；
   dp-attention（GLM-5.3-Flash）= 已知结构缺口，单列不进首批。
3. 乙档随对应需求排期（w4a16→R-05；w4a8→plugin 轨）；16 卡池 模型池补扫。
4. 失败项归因记录（格式未覆盖/架构缺口/资源不足），不静默丢弃。

**验收标准**
1. 首批每模型 models.md 绿行 + 启动配方入 cookbook（按 R-08 回仓原则）。
2. 模型×并行（TP/EP）矩阵表落 docs；乙档去向逐项标注。
3. 不可用项的理由留档（本明细即底账，后续变动在此更新）。

### R-11 自研 triton 档产线——独立 agent 轨（emu 尾部三件 → R-09 融合面 → R-05 桥）

**背景与定位（2026-10-18 评审拍板"开"）**
- 与dev（serving 轨）（serving 轨）并行的第二条工程线，资源错开：dev（serving 轨） =
  生产节点 A GPU4-7 serving 轨；dev（triton 离线轨） = **离线验证节点（BW1101）卡
  4-7 离线单测主场**，bench 与平台判别待 BW1102 窗口复核。
- 活源：emu 清退尾部三件无人认领（fa3 prefill/varlen、kpool write、
  flash_mla/dsa_paged_mqa 面），后续按 R-09 fusion-ledger 缺失项排序承接，
  R-05 marlin 桥候补——即长期方向第 3 条"自研 triton 档"的产线化。
- **卡型警示**：离线验证节点=**BW1101 ≠ 生产 BW1102（gfx938）**——正确性/对拍在
  离线验证节点 有效；性能数字不迁移；gfx938 特有 bug（如 kpool int64 div-4）在
  BW1101 不可复现，相关判别必须标注"待 BW1102 复核"。

**隔离三规**（见使用约定）：独立 clone+分支不推 master；止步 candidate 档
（production 晋级 = dev（serving 轨） serving A/B）；本条目 协调人 维护。

**工作内容**
1. 首任务：fa3 prefill/varlen triton 内核（现役 emu；decode 侧已 triton 化
   可参考——3D packed q/归一化 capture 合规等坑已沉淀在战报与测试）。
2. kpool write triton（前置 = gfx938 int64 div-4 平台 bug 判别，需 BW1102）。
3. flash_mla/dsa_paged_mqa 面 triton 化。
4. 完成后按 R-09 fusion-ledger 头部缺失四项承接（AR+norm 融合/silu+quant
   融合/quant vendor 档/weight_packed_linear）；R-05 marlin 桥候补。

**验收标准（每内核一验）**
1. 三段式单测绿（参照/形状矩阵/容差按 test-suite.md 快查表）。
2. bench vs emu 留档 + BW1102 复跑记录（或显式标注未复核）。
3. 三处登记齐（registry/env-vars/catalog）+ UI 快照同 commit；L0/L1 绿；
   catalog 状态=candidate。

### R-13 验收后遗留收编（R-01/R-02/R-12 出板后开放项跟踪）

**背景（2026-10-19 评审定：已完成项出板，遗留集中收编）**
三项验收收官（R-01 主线闭环 / R-02 emu 替换达成 / R-12 总装入口防护落钉
=commit e22bfb7，协调人 独立复跑本地 3P+2S、容器 L1 同款 env 4P+1S），各自
遗留收敛到本条统一跟踪；知识主体在 docs（models.md 战报 / env-vars.md /
fusion-ledger / catalog）。

**遗留清单**
1. ~~ar_ext.cu 真修复~~ **改判废弃（2026-10-19，ce7e402 已推）**：评审方
   判断实证——aiter.dist GraphSafe 在 sf_b075-20261018-0048 压测
   4-rank+8-rank 各 15/15 全绿（图捕获+新鲜输入重放×4，1eb038d 类不复
   现），serving 前科全绿，latency 与 NCCL 同档（88vs69us 小消息、
   1-16MB 0.93-0.95×）；custom AR 无生产位、坏输出不配当对比参照 →
   不修复，代码除役（triton_ar.py/ar_ext.cu/build+三手测工具删，
   allreduce.py 墓碑，四 env 键撤，catalog 除名）。汇合协议考古走 git 史。
2. **10k 容量墙**：fa3 emu 稠密化 OOM 32GiB（f1 长文 4972 过、10k 探针
   崩；与 fork 无关）。→ **已完成（2026-10-19，c59135b 已推）**：根因=
   emu (maxL×C) 板 q 行零分块（无 tile 时 8192 面实测 peak 22.02GiB，
   叠 serving 99.6GiB 驻留=28.88GiB 单笔 OOM；前役 triton 臂 2/2 毙 vs
   auto 3/3 过的"臂分叉"实为驻留差异不是臂差异）；修=QTILE_TOKENS=1024
   行子分块（内层 KV 流式 LSE 合并原样，逐 bit 同序、图安全）——峰值
   6.80GiB（-69%）、fp64 金标 0.0046、47 门绿+新钉
   test_extend_face_peak_bounded（跨卡缩放不变量）；**serving 终验
   triton 臂 11k×3 绿（11.3s=auto 同速）+全电池绿+LONGCTX-6k
   28.4→3.1s（~9× 彩蛋）+锚点与 auto 臂逐字一致+解码 8.27 tok/s 不变**。
   R-11 晋级障碍随之摘除（catalog 守 candidate 待行使 triton 核的模型过
   A/B）
3. **conv1d-flake**：fork `_causal_conv1d_fwd_kernel` 地址布局敏感误计算
   （L1 布局 cos 0.67 六进程稳定复现，换分配排列即 0.9999）——DTK triton
   codegen 嫌疑。**xfail-strict 观测桩常驻 L1（DTK 修好即 XPASS 翻红）**；
   主动排查候选：fla 上游对照 / DTK 编译器定位 / emu 回退评估。serving
   恒 channel-last 不受影响。
4. ~~**paged_mqa triton 档 production 晋级**~~ **已完成（2026-10-18 深夜，27c5847+3848d96 已推）**：validated（8849dc6 同窗 A/B：解码 +9%/+4%、四门绿+马拉松 0/20）→ production + 默认 vec→triton（hcu_env default/catalog note/env-vars 对账三面）；zero-env 翻转后四门验证绿（boot 即打 triton(default)、锚点 4/5 同基线形态、并发 6/6、确定性 x3、长文 5.9k 双事实对）；短 prompt 256 tok=33.1s/7.73 tok/s 三跑稳定（非同窗观测，仍优于 emu 参考值 34.18s）。emu-vec=显式 opt-out 保留（=vec/loop 双词）。
5. **kpool=1 DSA 模型接入风险（登记性）**：强制门下写行主序 vs 读交错
   将错位——届时 `HCU_DSA_PRESHUFFLE=0` 全链行主序或 writer 归一。
6. ~~**R-12 小打磨（minor）**~~ **已完成（2026-10-18 深夜，ee836e2 已推 Gitee）**：HIP 冒烟臂 bare-shell 守卫——`_sglang_on_child_path()` 收集期沿 _clean_env 组装的 PYTHONPATH 查 sglang 包目录（不真实 import：pytest 进程已被 sitecustomize 污染、探针子进程贵），两 HIP 臂 SKIPif 组合条件；容器双场景对照：bare 3P+3S（修复前 FAIL）、门禁 5P+1S 零回归；守卫契约测试 test_sglang_reachability_probe_contract 常驻；顺修 _clean_env `os.path != ROOT` 笔误。
7. **aiter reduce_scatter 重叠张量契约缺口（2026-10-19 R-04 EP 复验挖出，
   上游/DTK finding）**：EP 拓扑下上游 EP FFN 边界 attn-tp reduce_scatter
   是现树无条件路径，踩 aiter reduce_scatter 重叠张量契约 ValueError——
   现役解法=HCU_DISABLE_OPS=aiter_ar 强 NCCL 等价（af433c6 同役：ar_norm
   臂限纯 TP）。根因在 aiter 轮/上游边界语义，候选去向 operator-requests
   （对供应商提需求标准）跟踪。
8. **R-09④ weight_packed_linear**：随下轮 DTK wheel 复评（本轮 weight-pack
   机制面三重不可用已判死，R-04 终判同源）。
9. **R-09 ①②③ production 晋级**：quant_aiter 桶（15.1×/7.5×）与 ar_norm
   桶（1.38-2.11×、tp8 216 次融合实证）均 candidate——各需 serving A/B
   bench+马拉松后翻默认（quant 须 fp8 模型在池）。

**验收标准（按项）**
2/4/6 已完成（c59135b / 27c5847 / ee836e2，证据见验收记录）；1/3/7/8/9
各项：修复/晋级/复评落地时四门或 L1 证据 + 三处登记同步；5 接入时处理。

### R-14 上游单测收集面扩容——模型族全查证收编

**背景（2026-10-19 评审提缺口，协调人 上游树全查证）**
- 现状：l15_upstream 段只收 `attention/unittests` 四目录（dsa/kda/dense/mla，
  硬编码）；上游 kernels/ops/ 与 unit/models/ 两级**完全未收**；机制目录
  gdn/hybrid_linear/mamba/swa/lightning/dsv4 六个也未收。适配层对多个模型
  族有活桥/活路由，但上游契约面零单测覆盖。

**六族查证底数（容器 /sglang-v0521 实查，2026-10-19）**

| 族 | 上游 arch | 可收测试（未收） | 备注 |
|---|---|---|---|
| DSv4 / v4.1 | deepseek_v4.py(+nextn/dspark/vit) | unittests/dsv4/test_deepseek_v4.py（25 测）；kernels/ops/attention：dsv4 三件（indexer_quant 3 / q_rope_store 4 / dsv41_prefill_sparse_indexer 4）+ compress_plan_bounds / compress_state_runtime_shapes + **dsv4/ 子目录**；unit/models 六件（dsv4_shared_expert_fusion / unified_fp8_q_pair / rope_policy / mxfp4_shared_expert_requant / amd_fused_mhc / amd_wo_a_bf16——后两件 AMD 专属预期 SKIP） | 原始 R-14（dsv4 目录+三件）并入本条 |
| **GLM-5.3-Next**（生产主线！） | glm5_next.py(+nextn) | unit/models：test_glm5_next_bfg_fusion（3 测）/ test_glm5_next_modelopt / GLM 族件 test_glmmoedsa_correction_bias_fp32 · test_glm_moe_gate_fp32 · test_glm_nextn_moe_ptpc；kernels/ops/moe：test_glm53_shared_expert_fusion（2 测） | **生产模型零契约单测，本条最高价值项**；adapter 对它有 DUMP_LAYERS/mHC 活钩 |
| Qwen3.8 / Next | 无 qwen3_8 专名文件；qwen3_next.py(+mtp) 在 | 机制级=unittests 的 gdn/swa/mamba/hybrid_linear 四目录（Qwen3-Next=hybrid GDN 架，其注意力面全在这四目录）；unit 级=qwen3_5 四件（packed_weight_loader / mtp_quant_config / modelopt_fp4 / pipeline_parallel）+qwen4_exp_ple_table + embedding_registration | 上游无 "3.8" 专名测试——3.8 由 qwen3_5 族 arch 承载与否分诊时核；deterministic 类（test_qwen3_next/_qwen35_deterministic）=模型 boot e2e 面**不纳入**（归 R-10 模型轨） |
| KimiK3 | kimi_k3.py(+vl) | unit/models 三件：test_kimi_k3_packed_modules_mapping（1 测）/ test_kimi_k3_bfa_overlap / test_kimi_k3_vision | 无机制级专属目录（走 MLA 通用面）；AMD eval e2e 不纳入 |
| Hunyuan4 | hunyuan_v4.py(+nextn) | unit/models 两件：test_hunyuan_v4（4 测）/ test_hunyuan_v4_nextn_weight_loading | 件小，低风险批 |
| MiniMax-M3 | minimax_m3.py(+vl) | **上游无 unit 级测试**（只有 AMD mi35x/mi30x eval e2e） | 无可收物——登记"上游缺失"，不可能是我们补上游测试，归档即可 |

**功能区查证底数（评审方第二轮点名，同日实查）**

| 功能 | 上游家 | 底数 | 形态/风险 |
|---|---|---|---|
| **PD 分离**（disaggregation） | test/registered/disaggregation（21 件）+ unit/disaggregation（**33 件**） | unit 级 33 件=纯 python 单测（bootstrap/decode 池/kv 校验/role switch…，**首选批**）；顶层 21 件=多进程双服务 e2e（basic/dsv3 件等），挑核心件分诊 | unit 批零依赖可直跑；顶层批需双 boot，走 mock_model 或择件 |
| **dspark** | spec/dspark 目录（**13 件**） | block_accept_estimator/confidence_metrics/kernel_parity/scheduler/sps_table/stacked_ctx_kv_parity/ragged_verify 等——**纯单测大头** | kernel_parity/ragged_verify 可能踩 GPU 形状门（我们踩过 verify 崩点，预期有真实发现） |
| **MTP** | spec/ 顶层 f3 件 + mock_model/test_e2e_spec_eagle | test_frozen_kv_mtp / test_spec_mixed_chunk / test_spec_standalone(+extra)；无 python 级 mtp unit（EAGLE 路径走 e2e） | 项目红线认知：MTP×GLM5 图内 verify 在本栈 heisenbug（R-07 台账）——预期非全绿，登记有价值 |
| **dflash** | spec/dflash（2 件） | test_dflash.py + gemma4 e2e 件 | 主件单文件，小批量 |
| **EP**（高吞吐/低延迟） | ep/（8 件） | test_deepep_large（=LL 侧）/ test_deepep_small(+extra)（=吞吐侧）/ eplb_no_a2a / flashinfer_a2a / mooncake_ep_small | **多进程 requests 面为主（8/10 走 run_script/launch）**——boot 级，预期大部分 SKIP 或改 mock 后纳入；HCU_DEEPEP_BRIDGE 活桥正缺这层验证 |
| **hicache** | hicache/（11 件） | storage 四后端 file/mooncake/3fs/umbp + variants + spec_file/mooncake + pp_with_hicache + qwen35_hicache | storage 后端类纯 python 可直跑；import 依赖（mooncake/3fs/umbp 轮子缺失即 SKIP） |
| mock canary（补充发现，与 PD/CP/PP/TP 共振） | mock_model/（12 件） | test_e2e_{pd,cp,pp,tp,spec_eagle}.py + self_unit canary 六件 + perturb 两件 | **mock 模型 e2e= 无 GPU 权重也能跑部署拓扑的功能面**——hicache 有 qwen35 件同理：功能单测与模型解耦，正合 HCU 依托 mock 拓扑验证的路线 |

**工作内容**
1. 分批纳入（防一次吞撑爆 l15 段 2300s 超时）：
   - 批 A（机制目录）：**已完成（2026-10-19）**——l15_mech 段落地
     （gdn/hybrid_linear/mamba/swa/lightning/dsv4 六目录收集，含
     known-fail 预算绊线 passed≥90/failed≤42）；38 失败六族归因账本
     落 ci.md L1.5b，两跑稳定+全门 GREEN（证据与族表见 docs/ci.md
     L1.5b）；
   - 批 B（unit/models 族件）：按上表选件，新进 l15（或独立 units 段，实测定）；
   - 批 C（kernels/ops 族件）：glm53_shared_expert_fusion + dsv4 件全家；
   - 批 D（unit 功能面直跑优先）：unit/disaggregation 33 件 + hicache storage
     类 + spec/dspark 纯单测件 + spec/dflash 主件 + mock canary self_unit
     族——先扫"import 即 SKIP"清单定可跑面；
   - 批 E（多进程/boot 级，择件+改 mock）：disaggregation 顶层 21 件、ep 8 件、
     mock_model e2e_{pd,cp,pp,tp}——多数需双 boot 或多卡，逐件定"直跑/改造/
     登记不做"三态，**不硬塞 L1**；产出建议=独立功能段或夜间/发版档；
   - 批 F（红线预期管理）：MTP 三件 + dspark verify/parity 面——预期非全绿
     （R-07 台账在案），发现即登记，不设晋级门。
2. **首跑分诊三态制**（预期不做全绿承诺）：PASS / 合理 SKIP（AMD/无 GPU/
   依赖缺失）/ xfail-KNOWN_FAILURES 登记；非预期失败逐件归因（桥失配 vs
   上游已知 vs 平台限制）——mini KNOWN_FAILURES 账本落 docs/ci.md。
3. docs/ci.md L1.5 说明同步（批 A/B/C 落地各自同 commit）。

**验收标准**
1. 三批分发连续两跑稳定（不炸段、不拖垮 L1 全门 GREEN）。
2. 分诊三态清单 + KNOWN_FAILURES 账本落 docs/ci.md。
3. GLM-5.3-Next 五件与 qwen3_5 四件全部有处置（PASS/SKIP/登记三态之一，
   零静默忽略）。
4. 功能区（批 D-F）：批 D 可跑面清单 + 批 E 逐件三态处置表 + 批 F 发现
   登记（MTP/verify 类不设晋级门）；全部落 docs/ci.md，零静默忽略。

### R-15 性能回归门（相对阈值 + 历史锚定 + 三级防线）

**背景（2026-10-19 评审拍板，协调人 上游 perf 面查证后设计）**
- 上游有三层性能测试（basic_perf/ 11 件、perf/models/ 2 件、kernels/
  benchmark/）：`check_perf(at_least("output_throughput", …, 2660,
  amd=2100))`——boot 真模型→bench→对**硬编码绝对阈值**断言，跌破即红，
  等于把性能回归做成单测。**机制可偷，数值不可收**：阈值锚定 NVIDIA/AMD
  官方 CI 机器，BW1102 无参照意义，直收必假红。
- 我方已有性能资产（散在役里，非常驻）：A/B 臂间 bench（R-11/R-02）、
  catalog 晋级 bench 留档门、冷编译污染教训（R-11 验收实测 0.0×→热
  1.9×）、L1 pick_gpu 空闲选卡。缺的是把它们固化成机器。
- 设计原则（与北极星挂钩）：性能主战场=host 提交路径与图节点瘦身——
  该层 e2e bench 最钝、**op/图级微基准最敏锐**（单次 launch us 级差异
  e2e 不可见），故微基准常驻优先级高于 e2e 吞吐门。

**设计定案（三件）**
1. **相对阈值，不做绝对断言**：断言①臂间比（triton vs emu ≥N×，防
   "换装悄悄变慢"）②历史比（本轮 vs 基准记录不跌 M%；基准值入库可学
   catalog 档案，换机器/驱动重锚定）。绝对值只记录不设门。
2. **宿主三分离**：跑前 GPU 空闲检查（pick_gpu 现成）+ bench 预热段
   强制（冷编译假阳在案）+ 结果带环境戳（卡型/驱动/时钟/驻留占用）。
3. **防线分级**：L1 进**亚秒级 op/图级微基准**（量 launch 开销/图节点数/
   热点算子 us 级——R-02 vec 化 21.6ms 图发射那类信号的常驻哨）；e2e
   吞吐门放**夜间/发版档**，与 R-10 模型轨合流（同一次 boot 同时收四门
   与性能门）。

**工作内容**
1. 微基准工具面：op 级计时器 + 图节点/launch 计数器（prof_torch/
   hipGraphLaunch 前役手法固化）接进 L1 新段（阈值/source-of-truth
   沿 catalog 或独立 json）。
2. e2e 性能门脚本化：basic_perf 的 run_bench_serving/check_perf 机制
   借鉴，断言改相对；挂夜间档与 R-10 合流。
3. 基准档案：首跑留档各微基准/e2e 参考值（含环境戳），登记重锚定规则
   （换卡/换驱动/大版本必须重锚）。
4. docs/ci.md 段说明同步（同 commit）。

**验收标准**
1. L1 微基准段常驻 GREEN 且耗时可控（段级超时实测定）；无 GPU 扰动
   （跑前后卡况检查在案）。
2. 至少一次真实防尾示范：人为植入性能回归（如临时换 emu 臂）微基准
   翻红（红→绿回放）。
3. e2e 性能门在夜间/发版档跑通一次（哪怕值仅入档），基准档案+重锚定
   规则落 docs（ci.md 或 models.md，不新建文件）。

### R-16 skill/CONTRIBUTING 与门禁体系同步维护（多人协作上手守则）

**背景（2026-10-19 评审定：同事将加入分工协作）**
- skill 是新同事的第一入口；首轮补丁已随本行登记交付（见 R-16 总表行
  状态），本行跟踪**剩余同步面**，防止门禁体系演进后守则再次漂移。

**剩余工作**
1. CONTRIBUTING.md MR 门禁段与 docs/ci.md 对齐（kernels+adapter 收集面、
   l15 上游单测零拷贝条款、一键 L1 用法——CI 扩容 R-14/R-15 落地后复查）。
2. skill references/flash_attn wheel 兼容段（2026-10-10 实测）复核刷新
   （vendor 新轮就位即改写，勿留过期结论）。
3. 新同事实际上手一轮后回捞 skill 缺口（首跑踩坑即 one-pit-one-test
   补进黄金法则/参考文档）。

**验收标准**
1. CONTRIBUTING/ci.md/skill 三处门禁描述一致（一处改三处对）。
2. 无过期日期标注的"实测"结论（凡带日期的段落随复核刷新）。
3. 新同事首PR不因守则缺失返工（守则失效项即为 skill 待补项）。

## 项目北极星与长期方向（背景板，不单独开需求号）

**北极星（评审拍板，2026-10-17 重申）**：官方 sglang 代码零修改，拉下来挂上
adapter 即在 HCU 上跑通，且**性能正常**——能跑不算数，性能要对得起硬件。
当前性能主战场在 host 侧提交路径与图节点瘦身（hipGraphLaunch 线），不只算子本身。

三条推进线：

1. **调测效率**：官方单测（`l15_upstream` 段）代替端到端做日常验证；模型 e2e
   降频为发版/夜间。校准点：官方测试是**过滤器不是裁判**——发现适配层破坏
   即可，质量金标准是我们自己的三段式单测。
2. **自有单测资产**：三段式单测（参照/形状矩阵/容差）持续汇总，覆盖面**精心
   挑选**——不求全，锚定上游契约面 + 我们的接入面（影子符号面、桥的接缝面），
   目标是对新版本、新模型、新功能**快速支持**；新接入先补单测再上 e2e
   （CONTRIBUTING §3.4b test-first）。
3. **档位演进**：调试期 emu（稳、金标准）→ 性能期 triton 自研 → 成熟转
   vendor（aiter/lightop）。单向晋级、逐 catalog 条目按证据门禁推进
   （对拍 + bench + catalog 晋级）；emu/triton 换下不删、转 ref 角色只留
   离线测试。校准方向不变：emu 校准 triton；emu/triton 暴露 vendor bug；
   vendor↔triton 是性能竞争关系。
   **emu 清退清单注记（2026-10-17 协调人 对账，现役剩 6 处）**：①KDA prefill
   （triton 已写好 473–1009×，卡 R-01 门）②conv1d fn/update（16.4×/4.3×
   同卡 R-01 门）③fa3 prefill/varlen 面（decode 已 triton 化）④paged MQA
   logits（vec 化 emu 现役，vendor 替换=R-02）⑤kpool write ⑥flash_mla/
   dsa_paged_mqa 面。清零路径 ≈ R-01（双解锁）→ R-02 → fa3 prefill →
   收尾。已换装：silu/rms_norm/rotary/moe_gate（triton production）、
   GDN 线性注意力（aiter）、fa3 decode（triton）、MQA vec。
   **更新（2026-10-18）**：R-01 竞态定罪 custom AR（NCCL 缓解 60%→2.5%）
   已置待验收——若验收通过，①②两笔（最肥的）随即解锁换装。
4. **融合沿用（2026-10-17 评审定调）**：is_cuda 语义红利的兑现方式——上游
   融合入口（flashinfer/sgl_kernel 路径）默认**保持融合粒度**承接
   （vendor 融合核优先，其次自研 triton 融合核），拆解/降级为例外且必须
   显式登记（catalog 注记+理由），禁止静默拆解；正确性冲突时先保正确、
   vendor 融合核就绪后回接（mHC 先例：先 UNFUSE 保语义，deepgemm
   `tf32_hc_prenorm_gemm` 融合核可回接）。盘点见 R-09。

**硬件特性备忘（量化/内核选型的决策依据，2026-10-17 评审方提供）**：
HCU 卡的通用 core 与 tensor core **不能 overlap**——NVIDIA 上靠"矩阵核
乘 + 向量核同时处理 scale/量化"藏掉的开销，在本卡上**串行计费**。因此：
- 生产模型量化主路线 = **channel-wise**（scale 处理最少、可落 epilogue）
  + **Marlin W4A16**（量化成本离线 repack 挪走 + 自研 asm 内核按本硬件
  调度写）；block-wise 分组 scale 在 GEMM 主循环里的向量活结构性吃亏，
  对 block-wise fp8 路线的性能预期**不要按 CUDA 经验设目标**。
- 自研 deepgemm/marlin 与 CUDA 版不仅接口可能不同，**性能模型是另一套**
  （R-04/R-05 摸底与验收 bench 都以此为前提）。
- channel-wise int8 Linear 上游原生支持且已双卡型验证 e2e。

## 验收记录（在办任务的过程性证据）

> 已完成任务的验收记录随清行移除，证据沉淀于 models.md 战报 / ci.md /
> 对应 commit。下表只留在办任务的过程性证据。

| 日期 | ID | 结论 | 证据 |
|---|---|---|---|
| （在办过程证据追加处；完结即移除） | | | |
