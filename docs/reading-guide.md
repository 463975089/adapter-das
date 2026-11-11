# 文档导读（AI / 新人导航）

> 目标：任何会话（人或 AI 助手）接手本仓工作时，按本页顺序读 5 份文件
> 就能不迷糊地干活。每份文档的一句话定位 + 何时必读。
>
> **宿主纪律**：本仓同时服务 **sglang 与 vLLM 两个宿主**（`HCU_ADAPTER_HOST`
> 分发）。公共机制、公共不变量一律用"宿主"指称（= sglang 或 vLLM）；
> 只在某宿主成立的内容必须显式特指（如 `bridges/sglang.py`、影子包
> `sgl_kernel/` 属 sglang 进程，`bridges/vllm.py`、HcuPlatform 属 vLLM
> 进程）。改文档时先问一句：这段在另一个宿主上成立吗？成立→写宿主；
> 不成立→写明哪个宿主、差异是什么。

## 按任务找文档（30 秒路由）

| 你要做的事 | 必读（按序） |
|---|---|
| **第一次接触本仓** | 本文 → README → docs/architecture.md（§1-§3 机制图）→ docs/governance.md |
| 提 MR / 改代码 | docs/governance.md（§1 分工授权 / §5.0 squash 合入 / §7 协作红线与共建边界）→ docs/ci.md |
| 加/改一个算子 | skills/…/SKILL.md → references/shadow-matrix.md → references/test-suite.md（命名与档位归档见 governance §3b，改完跑 gen_tier_matrix） |
| 加一座桥 / 新功能选位 | docs/governance.md（§0 双仓判别 → §3 裁决链 → §4 登记/dump → §7 接入时序/共建） |
| 向算子组/供应商提需求 | docs/operator-requests.md（锚定 sglang 契约面，机器数据先行） |
| 追上游新版本 / 日常 compatibility 巡检 | docs/upstream-watch.md（规则版：报告优先+自主修复边界+工具清单） |
| 配置后端 / 部署 | docs/config.md → hcu_adapter.example.yaml → docs/env-vars.md（调试覆盖） |
| 部署 vLLM 0.32 | docs/frameworks/vllm032.md（组件职责、启动时序、wheel 锁定与验证） |
| 迁移 vLLM 0.33 | docs/frameworks/vllm033.md（独立发布线、三仓定位、公开 attention/communicator 接口与验收边界） |
| 跑一个模型 | docs/cookbook.md（找配方）→ docs/models.md（看先例与坑）→ cookbook.md §4 平台约束 |
| **新容器/新镜像首次部署** | docs/bootstrap.md（依赖重打+tvm-ffi 预热+自检，照做 ≤10min） |
| 排查精度问题 | skills/…/references/precision-triage.md → references/kill-switches.md |
| 改 CI / 加门禁 | docs/ci.md → docs/test-route-governance.md（目标态：精度→功能→性能→路由→退役）→ ci/l0_gate.py |
| 梳理 L1.5 / 更新上游用例 manifest | docs/l15-test-inventory.md（逐文件候选与分诊）→ docs/l15-family-matrix.md（Dense/MLA/MoE/量化/Mamba 代表矩阵）→ docs/test-route-governance.md §1.3–§1.5 → docs/test-route-implementation-plan.md MR-02A～02E |
| 设计性能 A/B、路由统计或代码退役 | docs/test-route-governance.md → docs/test-route-implementation-plan.md（MR-00～MR-12）→ docs/autotune.md → docs/governance.md |
| 按测试治理路线开工 / 查依赖和验收 | docs/test-route-implementation-plan.md → 对应 MR 段的依赖、DoD 与验收命令 |
| 判断一项工作归哪仓 / 开发插件新能力 | docs/governance.md §0 判别 → 本文「姊妹仓地图」→ sglang-plugin-das docs/ |

## 算子目录入口

当前启动就绪门禁的执行顺序与严格入口见 [l15-readiness-plan.md](l15-readiness-plan.md)。

目录分工见 [governance 算子三居地图](governance.md#算子三居地图与宿主绑定层)：
`sgl_kernel/` 是 SGLang ABI 影子层，`hcu_adapter/` 顶层保留历史桥接和公共机制，
`hcu_adapter/kernels/` 放上游契约计算实现，宿主安装与契约分发位于 `bridges/`。
DeepInfer 的具体文件位置见[算子导航](frameworks/deepinfer-kernels.md)，
[计算覆盖](frameworks/deepinfer-compute.md)、[辅助功能](frameworks/deepinfer-auxiliary.md)
和[宿主接入](frameworks/deepinfer-bridge.md)统一收在 `docs/frameworks/`。

## 文档地图（谁管什么，一句话一份）

| 文档 | 一句话定位 | 失效信号 |
|---|---|---|
| README.md | 项目是什么、装怎么装、 empirically 跑通了什么 | 覆盖范围表与 sgl_kernel 实际导出不符 |
| docs/architecture.md | 分层机制：影子库/is_cuda 语义路由/三档路由/横向机制（配 architecture.html 一页图） | §3 图与 `hcu_adapter/` 目录结构不符 |
| **docs/governance.md** | **治理宪法**：双仓边界（adapter=行为还原 / plugin 独立仓=性能新能力）、seam→rebind→text-patch 裁决链、active-ref 角色、运行时注册表与 dump、plugin 接入时序 / 双宿主共建 / 维护闭环（§7） | 与 ci/l0_gate.py 三规则或 `hcu_adapter/registry.py` 行为不符 |
| docs/ci.md | CI 当前分层与实测账：L0 → L1/L1.5 GPU → L2 扩展验证（含双线三绑定铁律） | GATES 列表与 l1_gpu_gate.sh 不符 |
| docs/l15-test-inventory.md | L1.5 v2 逐文件候选账：分类、logical op、路由证据、纳入/排除和晋级条件 | 上游版本变更、nodeid 漂移或 manifest 已更新但清单未同步 |
| docs/l15-family-matrix.md | L1.5 Dense/MLA/MoE/量化/Mamba 五类无权重代表行为矩阵、runner 和 DoD | 固定 case、上游 exact nodeid 或 family 边界与实际执行不符 |
| docs/test-route-governance.md | 测试目标态路线：精度→功能→性能→运行态路由→报告→代码退役 | route schema、分层、生命周期或报告格式与实际落地不符 |
| docs/test-route-implementation-plan.md | 上述目标态的执行计划：MR 依赖、代码布局、DoD、验收、资源与回滚 | 任务已落地但状态未回写，或实现布局/schema/门槛与计划不符 |
| docs/config.md | hcu_adapter.yaml 部署配置 + BACKEND_CATALOG 状态语义 | catalog 键集与 hcu_config.py 不符 |
| docs/frameworks/vllm032.md | vLLM 0.32、Adapter 与可选 Plugin 的职责、部署和组合验证口径 | entry-point 所有权或版本门与代码不符 |
| docs/vllm-commit-fatal.md | 公共显式 commit-fatal 与只读 runtime health；PR17 最小基线 | 锁存/fork/selector 行为或最小依赖与代码不符 |
| docs/operator-requests.md | 算子需求提报标准（契约锚定上游 sglang、shm≠spec、验收判据、版本锚闭环） | 与 governance 判别问句/catalog 状态语义不符 |
| docs/plugins.md | 须走 plugin/上游/镜像/专门窗口的需求登记表（P1-P9，判定规则+验收锚点） | 完成项未更新现状列 / 未过 §0 判定就收录 |
| docs/upstream-watch.md | 上游追新巡检规约（对账工具+日流程+自主修复授权边界，可交接版） | 工具清单状态与 #349/#350 实际落地不符 |
| docs/env-vars.md | 全量环境变量手册（调试覆盖定位） | REGISTRY 键集与 hcu_env.py 不符（L0 会抓） |
| docs/frameworks/vllm-required-extensions.md | vLLM 必选 entry-point/capability 启动策略、来源与 bootstrap/HOSTnone 日志边界 | 环境解析或实际 common/general-plugin 时序与契约不符 |
| docs/cookbook.md | 已验证启动配方（按模型/特性），§4 平台约束必读 | 配方与最新 master 行为不符 |
| docs/sglang-spawn-device.md | SGLang scheduler GPU selection before adapter bootstrap; spawn hint and alias contracts | Launch context or interpreter startup order changes |
| docs/bootstrap.md | 新容器 Bootstrap 手册：依赖套件版本钉、tvm-ffi 手工预热、镜像差异表、四条自检 | 镜像换代后版本钉失效（快照与镜像不符） |
| docs/models.md | 模型验证记录表（一天一行，判定四门）+ 组合特性清单 | 新模型跑完未加行 |
| docs/tier-matrix.md | 算子档位矩阵（自动生成）：每算子有哪些档、缺哪档、什么状态——实现排期的直接输入 | 与 BACKEND_CATALOG 不一致（L0 `impl-naming` freshness 会抓） |
| docs/vendor-kernels.md | DTK vendor 算子库 API 梳理（aiter/lightop/boltops/flash_mla/flash_attn/deepgemm/kda/tokenspeed，实机内省+热点映射机会表） | wheel 换代后 API 面/版本后缀失效 |
| skills/hcu-adapter-dev/SKILL.md | 开发规约入口 + 任务路由（references/ 下五份专篇） | 黄金法则与代码内守卫不符 |
| tests/tools/README.md | 物料命名语法（_test_/_probe_/_launch_/_dbg_） | 新物料未按前缀归位 |
| sglang-plugin-das 仓 docs/ | 插件方法侧导航与契约（见下节姊妹仓地图） | 与本文同名文档分工表冲突 |

## 姊妹仓地图（sglang-plugin-das）

插件仓在独立 git 仓维护，机制层与插件层的文档各自成套。三条分工铁律：

- **真相源分工**：机制层事实（加载时序、影子库、平台语义、算子档位与
  目录、env 键、后端选择）以本仓为准；插件策略层与开发流程（Hook 注册
  契约、`original_fn` 语义、插件工程/迁移台账、组合验证矩阵）以 plugin
  仓为准。同名文档回答同一件事的两个侧面，不互为备份。
- **判据单源**：一项工作归哪仓，以 governance §0 判别问句为准；
  `sglang-plugin-das/docs/governance.md` §1 与本仓 `docs/plugins.md` §0
  是它的两个应用视角，三处任一改动同笔同步（governance §7 前言有同款
  铁律）。组合兼容契约在 plugin 仓 governance §2-§4。
- **跨仓引用写"仓坐标"**（仓 + 文档路径 + 节号），不放跨仓相对链接。

同名文档分工对照（6 对）：

| 文档 | adapter-das（本仓） | sglang-plugin-das |
|---|---|---|
| architecture.md | 机制分层：影子库 / is_cuda 语义 / 三档路由 / 桥结构 | 插件架构 + 未来运行时目录规划 |
| governance.md | 治理宪法：双仓判别（真相源）、裁决链、§7 共建 | 组合兼容契约：目标所有权、验证矩阵 |
| bootstrap.md | **宿主**新容器部署手册（依赖钉 / tvm-ffi 预热 / 自检） | 方法库开发环境准备 |
| ci.md | L0/L1/L2 宿主门禁体系 | 方法仓 CI 阶段约束 |
| plugins.md | 需求登记表 P1-P10：**哪些项必须走 plugin** | 能力开发指南：**拿到项之后怎么做** |
| reading-guide.md | 机制侧导航（本页） | 插件方法侧导航 |

"改哪仓"速断：行为还原 / 内核实现 / 函数换绑 / 配置与环境键 → 本仓；
新增能力 / srt 层注册与调度结构 → plugin 仓；拿不准先过 governance §0
判别问句，再到本仓 plugins.md §0 与 plugin 仓 governance §1 复核。

## 不变量（写在多处，改代码前背一遍）

1. **宿主 python 零修改**（sglang 与 vLLM 同律）——一切经 import 面进入
   （影子包/元路径桩/sitecustomize）。
1b. **平台语义走 is_cuda 分支**（不走 is_hip 路线；fnuz 等语义护栏白名单锁定）——
    NVIDIA 库符号按四级路由：lightop → aiter → triton 先行 → 响亮 stub
    （governance §0.5）。
2. **算子数学只在内核层**：sglang 侧在影子包 `sgl_kernel/`，vLLM 侧经
   `bridges/vllm.py` 绑定同批内核；桥与编排器只写 glue
   （L0 `glue-numerics` 强制）。
3. **一个 catalog 算子一个 active 实现**；ref 实现只活在离线测试里。
4. **sitecustomize 不新增实现体**（新桥直接落 `hcu_adapter/<域>.py`）。
5. **GPU 路径零主机同步**（`.item()`/`nonzero`/布尔掩码）——graph capture 保命。
6. **共享节点只动自己前缀的容器/进程**（每人用自己名字缩写做前缀，
   不碰别人的）；长任务 `timeout N`；令牌永不落盘。

## 历史答案速查（别再踩的坑）

文档里所有"（hash）/案例/教训"标注都是真实事故的沉淀，遇到类似症状先搜：
rope cache 半分布局（fbc540c）、aiter AR 图重放损坏（1eb038d）、HIPC 权重必须
预打包、AWQ 必须 `--quantization moe_wna16`、恒显式 `--attention-backend`、
moe_align capture 崩（527ad97）。金索引法：docs/models.md 备注列 + commit log。

## 文档维护规则

- **改行为必改文档，同 PR**：三门禁（env↔env-vars、catalog↔config、
  规则↔governance）由 L0 兜底，其余靠 review。
- 每份文档表头的"失效信号"列即自查清单：改到相关代码时顺手核对。
- 本导读的"30 秒路由"表新增文档时必须加行。


vLLM 0.33 mHC 行为适配候选：`docs/frameworks/vllm033-mhc.md`；数学委托官方 Torch，公开 OOT 注册归宿主桥，post/head 保留，完整四维及模型门禁未闭合。
