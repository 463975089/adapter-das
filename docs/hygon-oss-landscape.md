# HYGON-AI 开源库图谱：作用、边界与协作方法

> 本文回答：**github.com/HYGON-AI 组织的开源库各自是什么、与 adapter/plugin 两仓什么关系、
> 需要能力/修 bug 时去哪个仓、以什么方式参与。** 出镜像/选 vendor 档/提报算子需求前先查本文。
> 库清单为 2026-11-05 页面快照（该组织 55 个公开仓）；描述与 organ 页或仓库 README 实测一致，
> 标注 **[深读待补]** 的行是页面级定位、尚未逐仓核对（下轮补）。
> 配套：vendor 能力台账 [vendor-kernels.md](vendor-kernels.md)、算子提报契约
> [operator-requests.md](operator-requests.md)、治理边界 [governance.md](governance.md)。

## 0. 参与总原则

1. **组织内的库我们都是同一研发体系的下游与共建方**：`*-das` 后缀 = 基于上游开源的
   海光适配版（绝大多数 Apache-2.0 / MIT / BSD，均可直接参与开发和调试）。
2. **adapter 不自研 vendor 能力**（governance §0）：vendor 档的数学实现归 lightop /
   aiter-das / deepgemm-das / FlashMLA-das 这些库；我们出绑定桥 + 档位路由 +
   数值稽核。库的能力缺口按 [operator-requests.md](operator-requests.md) 契约提报，
   不在桥里造轮子。
3. **反向贡献**：adapter 侧沉淀的通用修复（如 triton 已知内核 bug 的 workaround、
   HIP 布局修正）应提 PR 回对应 `*-das` 仓——我们趟出来的坑让兄弟库受益。
4. **引用纪律**：第三方代码入库必须保留 upstream notice（`fla_hcu/` 是先例——
   Apache 代码可引用不可处分，软著申请须圈出第三方边界）。
5. 版本入镜规则：任何 vendor wheel 进部署镜像前过红绿门 / emu 锚
   （[ci.md](ci.md) L1），台账记 [vendor-kernels.md](vendor-kernels.md)。

## 1. 核心执行线（与我们同一日常链路）

| 仓库 | 是什么 | 与本仓关系 | 参与方式 |
|---|---|---|---|
| **sglang-das** | 基于 sglang 上游的下游适配 fork（das srt 补丁面所在，Apache-2.0） | **历史主线**：现役 rel 树的前身；P1 登记的 srt 级功能正逐步迁 plugin 仓。追新对照跑官方原样树 + 本 adapter | 对照基准；fork 里的修复逐条过 plugin 化判定 |
| **adapter-das**（本仓） | 影子库 + `hcu_adapter` 运行时桥（私有） | —— | MR 走本仓治理 |
| **sglang-plugin-das** | sglang 插件仓（私有，活跃线 develop）：`hcu_sglang` hook 迁移 + 模型层 `sglang_hcu_plugin` | plugin 侧真相源（分工见 reading-guide 姊妹仓地图） | 机制问题来本仓，能力开发去插件仓 |
| **vllm-plugin-das** | vLLM 上游的 das 插件仓 | vLLM 宿主线的 plugin 侧对应物（governance §7.2 共建边界） | vLLM 侧需求与该仓协作，adapter 的 `bridges/vllm.py` 承平台/运行时 |
| **inference-cookbook-das** | HCU 部署/调优配方库（K100_AI/BW100/BW1000 卡型 × 模型表 × Docker 镜像列表，MIT） | **官方版模型台账**：与我们的 docs/models.md 互为补充—— cookbook 按"卡型×框架"收录已验证配方 | 跑新模型先查它的表；我们的首行结果回填它 |

## 2. vendor 档能力源（adapter vendor tier 的上游）

这些是影子库四级路由 ①lightop ②aiter 的供给方；缺什么能力向它们提报：

| 仓库 | 是什么 | 覆盖（实测 README） | 注意 |
|---|---|---|---|
| **lightop** | 自研融合算子库（BitBlankSeries GEMM 等，Apache-2.0） | 路由第①档首选；BF16 GEMM 等生产档 | 本地有源码树，bug 直修直提 |
| **aiter-das** | ROCm aiter 的 HCU 适配 | 路由第②档；MoE/AR/w4a8 等生产档 | 依赖 **Moe** 仓（hipc 算子）；gfx950-only 门见 env-vars |
| **Moe** | hipc MoE 算子库（aiter 的依赖） | aiter-das 的底层 | 经 aiter-das 间接使用 **[深读待补]** |
| **deepgemm-das** | 量化 GEMM + MoE 分组矩阵内核（MIT） | DS 系 deepgemm 契约的 HCU 实现；`HCU_DEEPGEMM_BRIDGE` 的对端 | mxfp4/fp8 expert 消费端候选 **[深读待补]** |
| **FlashMLA-das** | DeepSeek FlashMLA 移植（MIT） | **gfx936/938**：BF16/FP8 稀疏 MLA prefill+decode、DSA BF16 稀疏解码、FP8 KV 656B/token 布局、split-KV/索引排序/topk_length | `mla_attention_dtk` 桥的对端；**评估 K3.1/DSv4.1 attention 档前先核本仓 wheel 是否含 sparse 能力** |
| **hytlass** | CUTLASS v3.5.0 系模板库（BSD-3） | HuTe 模型；TF32/FP16/BF16/I8/U8 GEMM/conv；组 GEMM/blocked-ELL 稀疏 | **无 fp8/fp4 GEMM**——量化 GEMM 短板，deepgemm-das 补位 **[深读待补]** |
| **DeepEP-das** | DeepEP 专家并行通信适配 | EP 通信档 | 现役 box 验证过 v1 面 |
| **UltraEP-das** | 专家并行通信（Python, MIT） | EP 通信新势力 **[深读待补]** | |
| **mooncake-das** | Mooncake KV 传输（fork kvcache-ai） | KV 分离传输；PD 分离链路 | P3（PD 分离）归属面的依赖 |
| **LMCache-das** | LMCache KV-cache 层适配 | 分层 KV | HIcache 族联动 **[深读待补]** |
| **fastsafetensors-das** | 快速模型加载 | 3TB 级权重（K3.1 类）冷启动优化候选 | **[深读待补]** |

## 3. 部署与运维配套（多机/集群形态用）

| 仓库 | 作用 |
|---|---|
| **hcu-dcgm** | 设备监控/诊断 CLI + REST（Go） |
| **hcu-exporter** | Prometheus 指标导出（温度/功耗/显存，Go） |
| **k8s-hcu-device-plugin** | K8s Device Plugin DaemonSet（注册 HCU 资源） |
| **k8s-hcu-dra-driver** | K8s DRA kubelet 插件（CDI/ResourceSlices） |
| **hcu-container-toolkit** | 容器设备注入工具 |
| **volcano** | 批调度适配 |
| **rocSHMEM-das** | rocSHMEM 通信适配（GPU 侧 partitioned memory） **[深读待补]** |

多机 TP/EP 部署（如 K3.1-Pro 4 机 32 卡形态）的整条配套官方件齐全，
镜像组编排直接复用，不自行拼装。

## 4. 训练/其他线（记录备查，非本仓职责）

Megatron-LM-das、Deepspeed-das、apex-das（训练三件）；xla-das、slime-das、
AReal-das（RL）、DiffSynth-Studio-das（扩散）、hcu-train-simulator（训练性能建模）；
opencv/opencv_contrib/opencv-python-das、spconv-das、faiss-das、
torchvision-das/tensorflow-das（多模态/稀疏卷积/检索/框架下游）。
均 **[深读待补]**——涉及模型训练侧需求时按第 0 节原则参与。

## 5. Skills：HYGON-AI SkillHub

组织技能发布渠道（agentskills.io 规范，Apache-2.0，PR 评审准入，trust 模型：
"CLI 发现性 ≠ 操作正确性"——安装前照例审脚本与权限）。

已发布（2026-11-05 快照）：

| 技能 | 用途 | 对本仓的价值 |
|---|---|---|
| **torch-trace-operator-profiler** | torch.profiler Chrome/Perfetto trace 的算子级归因（Python 管线 / ATen / kernel / runtime API / memcpy） | **直接可用**：P9 图发射提报、档位竞速取证正是这个面；本机已装同名技能 |
| **skillhub-contributor** | 技能创建/评审/入库（components.d 注册 + catalog 校验） | 想把技能提报组织库时的流程入口 |

安装/贡献：`npx skills add HYGON-AI/SkillHub [--skill <name>]`；
贡献路径 `scripts/contribute.py import|new` → `check` → 单 PR（内容+注册+catalog 再生成）。

与本仓关系：`skills/hcu-adapter-dev/` 是**仓内技能**（带 references/ 依赖本仓文档，
不满足 SkillHub 自包含准入）；若要组织级分发，先做自包含精简版再走 skillhub-contributor。

## 6. 维护纪律

- 本文的库清单按季核对一次组织页；新仓库入表时按第 1~4 节归类并补"与本文关系"。
- 第 2 节的库每次 wheel 换代，同步 [vendor-kernels.md](vendor-kernels.md) 台账
  （API 面/版本后缀失效信号）。
- 提报/PR 用语遵守组织规范（HCU 命名纪律同本仓 governance）。
