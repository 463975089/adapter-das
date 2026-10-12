# 文档导读（AI / 新人导航）

> 目标：任何会话（人或 AI 助手）接手本仓工作时，按本页顺序读 5 份文件
> 就能不迷糊地干活。每份文档的一句话定位 + 何时必读。

## 按任务找文档（30 秒路由）

| 你要做的事 | 必读（按序） |
|---|---|
| **第一次接触本仓** | 本文 → README → docs/architecture.md（§1-§3 机制图）→ docs/governance.md |
| 提 MR / 改代码 | CONTRIBUTING.md（§1 流程、§3 checklist、§5 红线）→ docs/ci.md |
| 加/改一个算子 | skills/…/SKILL.md → references/shadow-matrix.md → references/test-suite.md |
| 加一座桥 / 新功能选位 | docs/governance.md（§0 双仓判别 → §3 裁决链 → §4 登记/dump）→ CONTRIBUTING §3b |
| 配置后端 / 部署 | docs/config.md → hcu_adapter.example.yaml → docs/env-vars.md（调试覆盖） |
| 跑一个模型 | docs/cookbook.md（找配方）→ docs/models.md（看先例与坑）→ cookbook.md §4 平台约束 |
| 排查精度问题 | skills/…/references/precision-triage.md → references/kill-switches.md |
| 改 CI / 加门禁 | docs/ci.md → ci/l0_gate.py（治理三规则在此实现） |

## 文档地图（谁管什么，一句话一份）

| 文档 | 一句话定位 | 失效信号 |
|---|---|---|
| README.md | 项目是什么、装怎么装、 empirically 跑通了什么 | 覆盖范围表与 sgl_kernel 实际导出不符 |
| docs/architecture.md | 分层机制：影子库/import 缝/三档路由/横向机制（配 architecture.html 一页图） | §3 图与 `_adapters/` 目录结构不符 |
| **docs/governance.md** | **治理宪法**：双仓边界（adapter=行为还原 / plugin 独立仓=性能新能力）、seam→rebind→text-patch 裁决链、active-ref 角色、运行时注册表与 dump | 与 ci/l0_gate.py 三规则或 `_adapters/registry.py` 行为不符 |
| CONTRIBUTING.md | 协作流程：分支/MR/checklist/节点纪律（红线） | 与 Gitee 实际门禁行为不符 |
| docs/ci.md | CI 分层：L0 秒级 → L1 GPU ~10min → L2 夜间 | GATES 列表与 l1_gpu_gate.sh 不符 |
| docs/config.md | hcu_adapter.yaml 部署配置 + BACKEND_CATALOG 状态语义 | catalog 键集与 hcu_config.py 不符 |
| docs/env-vars.md | 全量环境变量手册（调试覆盖定位） | REGISTRY 键集与 hcu_env.py 不符（L0 会抓） |
| docs/cookbook.md | 已验证启动配方（按模型/特性），§4 平台约束必读 | 配方与最新 master 行为不符 |
| docs/models.md | 模型验证记录表（一天一行，判定四门）+ 组合特性清单 | 新模型跑完未加行 |
| skills/hcu-adapter-dev/SKILL.md | 开发规约入口 + 任务路由（references/ 下五份专篇） | 黄金法则与代码内守卫不符 |
| tests/tools/README.md | 物料命名语法（_test_/_probe_/_launch_/_dbg_） | 新物料未按前缀归位 |

## 不变量（写在多处，改代码前背一遍）

1. **sglang python 零修改**——一切经 import 面进入（影子包/元路径桩/sitecustomize）。
2. **算子数学只在 `sgl_kernel/`**；桥与编排器只写 glue（L0 `glue-numerics` 强制）。
3. **一个 catalog 算子一个 active 实现**；ref 实现只活在离线测试里。
4. **sitecustomize 不新增实现体**（新桥直接落 `_adapters/<域>.py`）。
5. **GPU 路径零主机同步**（`.item()`/`nonzero`/布尔掩码）——graph capture 保命。
6. **共享节点只动自己前缀的容器/进程**（每人用自己名字缩写做前缀，
   不碰别人的）；长任务 `timeout N`；令牌永不落盘。

## 历史答案速查（别再踩的坑）

文档里所有"（hash）/案例/教训"标注都是真实事故的沉淀，遇到类似症状先搜：
rope cache 半分布局（fbc540c）、aiter AR 图重放损坏（1eb038d）、HIPC 权重必须
预打包、AWQ 必须 `--quantization moe_wna16`、恒显式 `--attention-backend`、
moe_align capture 崩（527ad97）。金索引法：docs/models.md 备注列 + commit log。

## 文档维护规则（与 CONTRIBUTING §3 呼应）

- **改行为必改文档，同 PR**：三门禁（env↔env-vars、catalog↔config、
  规则↔governance）由 L0 兜底，其余靠 review。
- 每份文档表头的"失效信号"列即自查清单：改到相关代码时顺手核对。
- 本导读的"30 秒路由"表新增文档时必须加行。
