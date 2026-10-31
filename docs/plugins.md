# plugins.md — 须走 plugin / 上游 / 镜像 / 专门窗口的需求清单

> 本文回答：**哪些已识别的性能与能力项不能（或不该）进 adapter 零侵入层，
> 各自的归属、现状与验收标准是什么。** adapter 的边界 = sitecustomize
> 运行时桥 + `hcu_adapter`（vendor 内核 / emu / 路由），**不修改宿主框架
> 内层代码结构**（sglang 的 srt 层；vLLM 的 platform/worker/runtime 层，
> 共建边界见 governance §7.2）。凡触及框架内层注册、预编译产物、上游合并、专门
> GPU 窗口的项，在此登记，不在 adapter 里强行实现。
>
> 配套：算子契约提报标准 [operator-requests.md](operator-requests.md)、
> 上游追新规约 [upstream-watch.md](upstream-watch.md)、adapter 环境键
> [env-vars.md](env-vars.md)、模型台账 [models.md](models.md)。
> 首版来源：2026-10-16 GLM-5.3-Flash 对照 sglang-das 首轮性能优化会话
> （commit 2907b14 / 2dcdac0；对照组 maxiaobetter/sglang-das
> `main_glm53_flash_feature_cleanup_tx` @ 364fabf，cookbook =
> inference-cookbook-das）。
>
> **2026-10-28 状态注**：plugin 载体已从"规划"落地为独立仓
> `sglang-plugin-das`，挂上游 sglang v0.5.22 官方 entry_points
> （`sglang.srt.plugins`）+ HookRegistry，首版只有诊断面——P1/P2 不再
> 依赖"等 plugin 机制"，逐项迁移即可开工；vLLM 侧按对称结构另立 plugin
> 仓（共建边界、接入时序、维护闭环见 [governance.md](governance.md) §7）。
> 下表"现状"列仍为 2026-10-16 口径，推进时逐行更新。

## 0. 判定规则（一项进本清单的门槛）

| 判据 | 含义 | adapter 为什么够不着 |
|---|---|---|
| **plugin** | 要改 srt 层结构：attention backend 注册、调度器、kpool 规划、spec-decode 框架挂载点 | sitecustomize 只能换绑函数/内核，挂不进类注册与调度决策 |
| **镜像** | 依赖预编译产物：aiter ASM/tuned 调优表、lightop wheel、fork 镜像内算子库 | 我们镜像 = dtk26041 + torch 2.11 + 官方 v0.5.21；产物在 sglang-das 镜像（0.5.18 + OpenDAS 栈）里，运行时不可生成 |
| **上游** | 上游官方分支已有/应有修复，adapter 侧只需合流监控或红门回归 | 修复合入点在上游源码，adapter 打补丁只会变成维护债 |
| **窗口** | 纯调优/对拍工作，需专门 GPU 窗口，产出回填仓库 | 日常窗口被 e2e 验证占满；sweep/复测要独占卡 |

## 1. 登记表

| # | 项 | 判据 | 现状（2026-10-16） | 归属建议 | 验收锚点 |
|---|---|---|---|---|---|
| P1 | **sglang-das srt 级集成**：glm5_next DSA attention backend 注册、kpool planner（indexer top-k 规划）、fork 专属 `SGLANG_*` 功能旗标族（`SGLANG_USE_LIGHTOP` / `SGLANG_USE_DEEPGEMM_MOE` / `SGLANG_NSA_FUSE_TOPK` 等）+ **spec KV 位置规划补丁族**（`SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO` / `SGLANG_ASSIGN_EXTEND_CACHE_LOCS` / `SGLANG_GET_LAST_LOC`(hcu 版) / `SGLANG_KVALLOC_KERNEL`——v0.5.21 全无此机制） | plugin | fork 在 srt 层直接改结构，无运行时挂载点；我们的 DSA 靠 adapter 桥 + `--dsa-*-backend` 旗标跑通（Berlin 等锚点绿）。**2026-10-17 升级为 MTP/dp-attention/EP8 的前置**：stock v0.5.21 上 GLM-5.3-Flash 三个特性臂全 RED（详见 models.md 当日行）——EAGLE 草稿 accept 0.01-0.02 + target 轨迹漂移；dp-attention/EP8 独立致病且单流 2165ms/步；adapter 侧三个单点修复尝试（kpool 写 emu / get_last_loc→torch[sitecustomize #14] / fa3 emu 空批守卫）均不治本 → 结构性缺口就在 das 的 srt 补丁面 | plugin 机制落地后逐项迁移；迁移一项 A/B 一项；spec 补丁族（allocation.py `alloc_for_spec_decode`/`get_last_loc` 差异面已量：~3 文件）优先 | e2e 四锚点 + 吞吐对比（同卡同 bs），旗标逐个开；MTP 臂加 accept len ≥3 门 |
| P2 | **MTP / spec-decode V2**（fork `SGLANG_ENABLE_SPEC_V2` + draft/target 框架改动） | plugin | 官方 v0.5.21 有自己的 spec-decode 面，fork V2 是 srt 层重写；未验证 | plugin；依赖 P1 的 backend 集成先行 | MTP 加速比 + 四锚点（spec 路径数值独立验证） |
| P3 | **PD 分离（1P1D）** | plugin / 部署 | cookbook glm-5.3 配方有 24 卡 1P1D 参照；PD 是部署形态 + srt 层支持；卡点分析工具 `tools/pd_flow_viz`（P/D 日志→请求级流转泳道动画+阶段归因，sglang 风格模式内置，真实日志待 `--probe` 适配）已入库 | plugin + 运维配方；先等单机 IFB 面完全转绿 | P/D 两角色四锚点 + 并发零串扰 |
| P4 | **aiter 预编译快路**（`SGLANG_USE_AITER_*` 开关族） | 镜像 | fork 默认关（我们镜像无产物也能跑纯 triton 路径）；aiter 在 HCU 需 gfx950，gfx938 上的融合 mhc 快路已证数值坏（词汤主病灶，`HCU_MHC_UNFUSE` 处置） | 等 DTK/aiter 官方出 gfx938 产物；逐算子过数值门再放行 | 每算子对拍 emu/参照 cos ≥0.9999 + e2e 锚点 |
| P5 | **E=288 MoE 调优表条目**（GLM-5.3-Flash E=288/N1=1024/N2=4096/K=4096/topk=8/block=128） | 窗口 → 镜像 | aiter `tuned_fmoe_asm_w8a8_group.csv` 已加载但无此形状条目 → 桥自动回退 stock triton + smem clamp，实测 36ms/步 @bs1（0.792ms/层×45） | 专门 GPU 窗口跑 sweep（`_probe_moe_decode_bench.py` 复测），产出条目进调优表 | decode ms/层显著下降 + 输出分布级锚点不劣化 |
| P6 | **mHC 融合 `not is_hip()` 修复合流** | 上游 | 上游已修（等效我们的 `HCU_MHC_UNFUSE`）；合流后我们的开关自然冗余 | upstream-watch 盯版本；合流后删桥 | 合流版直接起服 + 四锚点 |
| P7 | **DTK triton 编译栈 4 坏内核**（chunk_kda Aqk 全错 / conv1d update 噪声 / conv1d_fn varlen 初态死 / （第 4 点即 fn 臂）） | 上游（DTK） | adapter 已有 emu 止血 + 红绿门常驻；fork triton 内核 standalone 全绿（见 P8） | 按 operator-requests 标准向 DTK 提报；红门转绿即撤 emu | 红门测试（`test_kda_chunk_gfx938.py` / `test_conv1d_fn_varlen.py`）转绿 |
| P8 | **fork serving 门 RED 对拍**（vendored `fla_hcu` 内核 standalone 6/6 绿，serving 中 Berlin 确定性翻转） | 窗口（adapter 内部） | 默认 emu，fork 留仓 opt-in（`HCU_KDA_PREFILL_IMPL=fork`）；HCU_DUMP_LAYERS + `replay_kda_l0.py` 参考链已备（容器 /root） | 下个 GPU 窗口做 serving dump 对拍；转绿 = prefill 17-28s→~3s | serving 门 Berlin id=19808 等 + standalone 门双绿 |
| P9 | **DTK 大图发射（hipGraphLaunch ~435ms/步）** | 上游（DTK runtime） | 机制终版（2026-10-16 深夜 microbench 三连 + trace 复核）：launch API 是**纯宿主提交工作**（全同步后首调仍 6.8ms/千节点，GPU 实算仅 ~2ms）；**进程内全局串行**（2 线程分包 0.98×，piecewise 多线程死刑）；**跨进程独立**（两进程各 113/s=单进程原速）；**NUMA 不敏感**（scheduler 按 GPU 亲节点 taskset 绑核，e2e 零变化，`_launch_g53f_tp4_pinned.sh`）；env 全否决（见 env-vars.md）。服务 24k 节点/步 × 实参更重 ≈ 22µs/节点 = 540ms 提交墙；TP0 宿主额外职责致尾段 5.5ms 停顿量子，TP1-3 nccl 各等 2.5s/窗。附带发现：`torch.cuda.graph(g, stream=…)` 捕获路径本栈 segfault。**本地活路①（已落地，2026-10-17）：减图节点**——MQA logits emu vec 化（`HCU_MQA_LOGITS_IMPL=vec` 默认）图节点 23.9k→3.7k、launch API p50 540→21.6ms、单流 +52%（详见 models.md 2026-10-17 行）；**本地活路②（进行中）**：图剩余 3.7k 节点的下一刀（conv1d_update emu/mHC 链）。**新增 DTK 图 replay 崖证据（提报材料）**：图内大张量 fp32 物化与 (N,T,D) 转置物化单核 ~124ms（~0.8GB/s 定长，eager/小图同形状仅 µs 级；宿主窗口零 API 活动、每步 11 拍 151ms 设备侧等待——详析 prof_torch_vec/ V0-V2 三代对照）；根治仍在 DTK 提交/执行路径 | 按 operator-requests 向 DTK 提报（证据链：纯提交开销+串行性+跨进程独立+NUMA 不敏感+env 三连否决+stream 捕获 segfault+图 replay 大核崖） | 0.43s/步显著下降 + 四锚点不劣化 + 贪心确定性保持 |

| P10 | **自研 channel-wise 量化 scheme**（生产主力量化路线；kernel 归 `sgl_kernel` 共享层，scheme/apply 层归 plugin） | plugin | sglang 侧经 HookRegistry REPLACE quant scheme、vLLM 侧对称 plugin；格式规范与 conformance 单测**单源**（governance §7.2/§5.1 阶段 2）；checkpoint 格式先核能否参数化对齐上游 compressed-tensors 族参数化形态，对得上即白拿上游加载/转换工具链 | 每宿主一 plugin + kernel 共享；格式契约第一天固化文档+契约单测 | scheme 单测 + 宿主中立 parity 契约测试 + 两侧 serving gate 各一个 channel-wise checkpoint |

## 2. 与既有文档的边界

- **operator-requests.md**：P4/P5/P7/P9 向外提报时的**契约写法与受理标准**
  （本文档管"要不要提、提到哪"，那边管"怎么提算合格"）。
- **upstream-watch.md**：P6（及上游一切合流项）的**监控节奏与授权边界**。
- **models.md**：每项进展落模型台账行；本文档只维护清单状态，不记流水。
- **env-vars.md**：adapter 可控键的全量表；P9 的平台 runtime env（非
  adapter 键）在 env-vars.md 末节登记来源与拍板值。

## 3. 维护纪律

- 每完成一项：登记表改"现状"列 + 台账行落账 + （若涉及）红绿门/探针归位
  `tests/tools/`；**不在本文档写长篇过程**。
- 新增项必须过 §0 判定规则至少一条，且带"现状账"（今天怎么活下来的、
  代价多少）——与 operator-requests 的最小合格线同构。
