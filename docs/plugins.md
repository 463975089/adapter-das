# plugins.md — 须走 plugin / 上游 / 镜像 / 专门窗口的需求清单

> 本文回答：**哪些已识别的性能与能力项不能（或不该）进 adapter 零侵入层，
> 各自的归属、现状与验收标准是什么。** adapter 的边界 = sitecustomize
> 运行时桥 + `sgl_kernel/_adapters`（vendor 内核 / emu / 路由），**不修改
> sglang srt 层代码结构**。凡触及 srt 层注册、预编译产物、上游合并、专门
> GPU 窗口的项，在此登记，不在 adapter 里强行实现。
>
> 配套：算子契约提报标准 [operator-requests.md](operator-requests.md)、
> 上游追新规约 [upstream-watch.md](upstream-watch.md)、adapter 环境键
> [env-vars.md](env-vars.md)、模型台账 [models.md](models.md)。
> 首版来源：2026-10-16 GLM-5.3-Flash 对照 sglang-das 首轮性能优化会话
> （commit 2907b14 / 2dcdac0；对照组 maxiaobetter/sglang-das
> `main_glm53_flash_feature_cleanup_tx` @ 364fabf，cookbook =
> inference-cookbook-das）。

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
| P1 | **sglang-das srt 级集成**：glm5_next DSA attention backend 注册、kpool planner（indexer top-k 规划）、fork 专属 `SGLANG_*` 功能旗标族（`SGLANG_USE_LIGHTOP` / `SGLANG_USE_DEEPGEMM_MOE` / `SGLANG_NSA_FUSE_TOPK` 等） | plugin | fork 在 srt 层直接改结构，无运行时挂载点；我们的 DSA 靠 adapter 桥 + `--dsa-*-backend` 旗标跑通（Berlin 等锚点绿） | plugin 机制落地后逐项迁移；迁移一项 A/B 一项 | e2e 四锚点 + 吞吐对比（同卡同 bs），旗标逐个开 |
| P2 | **MTP / spec-decode V2**（fork `SGLANG_ENABLE_SPEC_V2` + draft/target 框架改动） | plugin | 官方 v0.5.21 有自己的 spec-decode 面，fork V2 是 srt 层重写；未验证 | plugin；依赖 P1 的 backend 集成先行 | MTP 加速比 + 四锚点（spec 路径数值独立验证） |
| P3 | **PD 分离（1P1D）** | plugin / 部署 | cookbook glm-5.3 配方有 24 卡 1P1D 参照；PD 是部署形态 + srt 层支持 | plugin + 运维配方；先等单机 IFB 面完全转绿 | P/D 两角色四锚点 + 并发零串扰 |
| P4 | **aiter 预编译快路**（`SGLANG_USE_AITER_*` 开关族） | 镜像 | fork 默认关（我们镜像无产物也能跑纯 triton 路径）；aiter 在 HCU 需 gfx950，gfx938 上的融合 mhc 快路已证数值坏（词汤主病灶，`HCU_MHC_UNFUSE` 处置） | 等 DTK/aiter 官方出 gfx938 产物；逐算子过数值门再放行 | 每算子对拍 emu/参照 cos ≥0.9999 + e2e 锚点 |
| P5 | **E=288 MoE 调优表条目**（GLM-5.3-Flash E=288/N1=1024/N2=4096/K=4096/topk=8/block=128） | 窗口 → 镜像 | aiter `tuned_fmoe_asm_w8a8_group.csv` 已加载但无此形状条目 → 桥自动回退 stock triton + smem clamp，实测 36ms/步 @bs1（0.792ms/层×45） | 专门 GPU 窗口跑 sweep（`_probe_moe_decode_bench.py` 复测），产出条目进调优表 | decode ms/层显著下降 + 输出分布级锚点不劣化 |
| P6 | **mHC 融合 `not is_hip()` 修复合流** | 上游 | 上游已修（等效我们的 `HCU_MHC_UNFUSE`）；合流后我们的开关自然冗余 | upstream-watch 盯版本；合流后删桥 | 合流版直接起服 + 四锚点 |
| P7 | **DTK triton 编译栈 4 坏内核**（chunk_kda Aqk 全错 / conv1d update 噪声 / conv1d_fn varlen 初态死 / （第 4 点即 fn 臂）） | 上游（DTK） | adapter 已有 emu 止血 + 红绿门常驻；fork triton 内核 standalone 全绿（见 P8） | 按 operator-requests 标准向 DTK 提报；红门转绿即撤 emu | 红门测试（`test_kda_chunk_gfx938.py` / `test_conv1d_fn_varlen.py`）转绿 |
| P8 | **fork serving 门 RED 对拍**（vendored `fla_hcu` 内核 standalone 6/6 绿，serving 中 Berlin 确定性翻转） | 窗口（adapter 内部） | 默认 emu，fork 留仓 opt-in（`HCU_KDA_PREFILL_IMPL=fork`）；HCU_DUMP_LAYERS + `replay_kda_l0.py` 参考链已备（容器 /root） | 下个 GPU 窗口做 serving dump 对拍；转绿 = prefill 17-28s→~3s | serving 门 Berlin id=19808 等 + standalone 门双绿 |
| P9 | **DTK 大图发射（hipGraphLaunch ~435ms/步）** | 上游（DTK runtime） | 机制已坐实（2026-10-16 用户 trace 分析 + 4×chrome trace 复核）：①launch API p50 540ms/次、每图 ≈24k kernel，AQL 队列固定容量→API 同步自旋；②rank0 短板——TP1-3 nccl 各烧 2.5s/窗纯等 TP0（通信不慢，是等齐）。env 处置在 dtk26041 **A/B 双双失败**：三件套捆绑与 `ROC_AQL_QUEUE_SIZE` 严格单变量均零收益+贪心非确定（详见 env-vars.md 平台 env 节）；runtime 确认支持 `ROC_AQL_QUEUE_SIZE`/`GPU_MAX_HW_QUEUES`/`HIP_USE_GRAPH_QUEUE_POOL`（后者未测，留作单变量候选）。trace 存证：容器 /root/logs/prof_torch/1792113313*.trace.json.gz（Perfetto 可开，本仓 prof_torch/ 有 md5 核对副本） | 按 operator-requests 向 DTK 提报（附 540ms launch API + rank 等待不对称 + 两轮 env A/B 数据，要求大图提交路径与队列扩容数值安全性修复） | 0.43s/步显著下降 + 四锚点不劣化 + 贪心确定性保持 |

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
