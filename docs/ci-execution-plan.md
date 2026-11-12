# adapter-das CI 执行规划

本文确定 adapter-das 的 CI 测试范围、门禁等级、资源使用规则和触发方式。
本文是执行方案，先用于统一范围；在方案评审完成前，不自动扩大 GPU 测试范围，
也不把真实模型测试挂到普通 PR。

## 1. 现状结论

当前 `.github/workflows/quality-gate.yml` 只在 PR 的
`opened`、`synchronize`、`reopened` 和 `ready_for_review` 事件触发，并调用
`HYGON-AI/quality-gate` 的 reusable workflow。仓库内的 `ci/l0_gate.py`、
`ci/l1_gpu_gate.sh` 和 L1.5 工具并没有在这个 workflow 中显式接线。因此需要把
“仓库有测试入口”和“CI 实际执行了测试”分开记录，避免误把外部工作流的结果当作
本仓库全部测试已经覆盖。

当前测试设计的方向是合理的：

- L0 结构和治理检查适合每次提交，在 CPU runner 上执行；
- L1 CPU/adapter 回归适合普通 PR；
- L1 GPU 和 L1.5 需要真实 HCU，不能在托管 CPU runner 上执行；
- L2 真实模型启动、生成和精度验证应作为扩展验证，不应阻塞普通 PR；
- `l15_v0522` 的 exact nodeid manifest 和五家族 58 case matrix 是正确的
  固化方向。

当前需要调整的地方：

1. `l15_mech` 不能长期只按 passed/failed 总数预算放行；旧失败消失、新失败出现时，
   总数可能不变，造成假绿。
2. `l15_v0522` 默认 `observe` 只能发现漂移，不能保护合入；应按批次逐步转为阻塞。
3. family matrix 默认 `off`，因此普通 L1 gate 实际不检查五家族 58 case；应先
   `observe`，稳定后再升为阻塞的受控 HCU job。
4. `l15_sglang_main.json` 是历史 main 快照，不能作为固定版本门禁；应归档为升级
   参考，或改为由升级 workflow 动态生成 manifest。
5. CI 必须把 adapter commit、SGLang commit、容器/工具链版本、GPU 型号和 manifest
   hash 绑定到 artifact，否则数值结果无法复验。
6. “无空闲卡”必须是排队或明确的 `resource-unavailable`，不能当作测试通过，也不能
   为抢卡自动选择正在使用的设备。

## 2. 测试层次和最终范围

| 层 | 范围 | 资源 | 门禁定位 |
|---|---|---|---|
| L0 | Python 语法、Shell 换行、敏感信息、registry/catalog/docs、源码治理和 manifest schema | CPU | 每次 PR 必须阻塞 |
| L1-CPU | `tests/test_ops_cpu.py`、adapter 生命周期/配置/导入/打包等无 GPU 用例 | CPU | 每次 PR 必须阻塞 |
| L1-GPU | adapter 自有 kernel/bridge 回归、精选工具 gate、vendor import sanity | 1 张 HCU | 受控 HCU job；稳定后对相关 PR 阻塞 |
| L1.5-upstream | SGLang attention core：DSA/KDA/Dense/MLA；机制：GDN、Hybrid、Mamba、SWA、Lightning、DSV4 | 1 张 HCU | exact manifest；逐 nodeid 判定 |
| L1.5-v2 | `l15_v0522` 的 Batch 1–4 候选，按 core/contract/diagnostic/platform-skip 分类 | 1 张 HCU | 迁移期 observe，分批升级为 blocking |
| L1.5-family | Dense/MLA/MoE/Quantization/Mamba 五家族 58 case | 1 张 HCU | 代表性组件门；不能替代模型启动 |
| L1.5-comm | CP/AR/通信相关用例 | 2 张 HCU | 独立受控 job；默认不挂普通 PR |
| **L1.6** | 指定 dummy 架构或真实 checkpoint 的 Engine 加载、eager/graph 生成、TP/显存配置和源码/权重完整性 | 1–4 张 HCU | L1.5 通过后受控执行；模型启动失败阻塞，完整精度另计 |
| L2-F | cookbook 模型启动、eager/graph 生成、短上下文和长上下文、量化/并行特性 | 1–8 张 HCU | 合入后、夜间或手动 |
| L2-P/L3 | 性能、压力、多机、RDMA、PD、EP、外部存储和长期稳定性 | 专用资源 | 专项 workflow，不作为普通功能 PR 门禁 |

L1.5 的有效覆盖必须分别报告：

- `core-green`：真实执行 adapter/HCU 路径且有 route evidence；
- `contract-green`：验证构造、metadata 或配置契约；
- `diagnostic-known-fail`：精确 nodeid、错误指纹、owner 和转绿条件齐全；
- `platform-skip`：平台守卫和 skip 原因稳定；
- `unclassified`：不能计入通过率，必须产生后续分诊任务。

报告不能只输出总 passed 数。至少要输出 collected、executed、各分类数量、
新增/消失 nodeid、skip reason 漂移、错误 fingerprint 漂移和 route evidence 数量。

## 3. 触发矩阵

### 3.1 普通 PR

普通 PR 的默认路径只使用 CPU runner：

1. L0；
2. L1-CPU；
3. manifest/schema 静态校验；
4. family matrix 的参数和 nodeid 校验，不执行 GPU；
5. upstream diff 的静态检查（如果本 PR 修改了上游绑定、manifest 或升级工具）；
6. 仓库敏感信息和真实环境信息扫描。

普通 PR 默认不自动申请共享 HCU。需要 HCU 证据时，通过受控的
`workflow_dispatch` 或带有明确权限的 label 触发 L1-GPU/L1.5 job。label 触发器
必须设置并发限制、资源锁和最大运行时间，不能由普通 PR 自动绕过。

以下改动建议自动标记为“需要 HCU 复核”，但不直接抢卡：

- `hcu_adapter/`、`sgl_kernel/`、`sitecustomize.py`；
- `tests/kernels/`、`tests/adapter/` 中的 GPU 或 graph 用例；
- `tests/manifests/`、`tools/l15_*`、`ci/l1_gpu_gate.sh`；
- SGLang binding、源码完整性检查或升级工具。

### 3.2 合入 main 后

合入 main 后由独立的 post-merge workflow 执行，和 PR 快速门分开：

1. 立即执行 L0 + L1-CPU，失败则标记 main 回归；
2. 在受控 HCU runner 上执行 L1-GPU；
3. 对 adapter/kernel/manifest 相关提交执行 L1.5-family 五家族矩阵；
4. 对 upstream binding 或 SGLang 适配相关提交执行 exact manifest collect/verify；
5. 只有前述结果满足门禁，才启动 L1.6 模型启动就绪；
6. L1.6 通过后，才启动 L2-F cookbook 模型冒烟。

post-merge 的 HCU job 必须允许排队，不能自动改用正在运行测试的节点。结果必须
上传为 artifact，并记录实际分配的资源、等待时间和未执行原因。

### 3.3 手动 dispatch

手动 workflow 用于复现、分诊和单独验证，输入至少包括：

- `suite`：`l1-gpu`、`l15-manifest`、`l15-family`、`l15-comm`、`l2-model`；
- `adapter_ref`：待验证的分支或 commit；
- `sglang_ref`：tag、分支或 commit；
- `manifest`：固定 manifest 版本；
- `families`：可选的 family 子集；
- `device_count`：1 或 2/更多；
- `mode`：`observe` 或 `blocking`。

手动任务必须把输入参数写入报告，避免“同名测试、实际版本不同”的结果混淆。

### 3.4 SGLang 源码升级

SGLang 升级使用独立的 `upstream-upgrade` workflow，不复用普通 PR 的默认门。
触发方式按以下优先级：

1. 手动 dispatch，指定 SGLang commit/tag；
2. adapter 的 upstream binding 或 pristine source 版本变更时自动触发；
3. 上游 release/tag 事件接入后再增加自动触发；
4. 每日追新暂不启用，等当前功能和精度问题收敛后再启用。

升级 workflow 的输入和报告必须绑定：

```text
adapter commit
sglang commit/tag
container image and DTK/Torch/Triton/vendor versions
manifest hash
GPU architecture and device count
```

升级流程固定为：

1. 创建 pristine SGLang checkout，不修改上游源码；
2. 运行 `tools/upstream_source` 和 `ci/upstream_diff.py`；
3. 做完整 collect，比较新增、消失、重命名 nodeid 和 binding 文件 hash；
4. 先执行 L0、manifest validate/inventory/collect/verify；
5. 执行 L1.5-family 五家族矩阵；
6. 只对受影响域执行 targeted L1.5 和 adapter 回归；
7. 通过组件门后，使用 cookbook 配方执行 L2 模型启动、eager 生成、graph 生成和
   精度检查；
8. 对错误指纹变化、路由变化或源码完整性变化生成升级报告；
9. 只有结果可复现且人工确认后，才更新 SGLang binding、manifest 分类和版本文档。

SGLang 升级 workflow 不应直接修改正在被其他调试任务使用的共享源码目录，也不应把
临时源码、模型路径、节点地址或原始日志提交到仓库。

## 4. 门禁晋级顺序

门禁按证据晋级，不按测试数量晋级：

### 阶段 A：静态和 CPU 基线

- 把 L0、L1-CPU、manifest validate 接入普通 PR；
- 把 artifact schema 和环境指纹格式固定；
- 任何 collection error、manifest schema error、源码完整性错误均阻塞。

### 阶段 B：L1.5 观察

- 保留现有目录门作为迁移期参考；
- `l15_mech` 同时输出逐 nodeid 结果和旧预算结果；
- `l15_v0522` 默认 observe，收集完整 Batch 1–4；
- family matrix 在受控 HCU post-merge job 中 observe；
- 连续两次结果中 nodeid、skip reason 和 known-fail fingerprint 稳定，才进入下一阶段。

### 阶段 C：精确门禁

- 逐步移除 `l15_mech` 的总数预算作为放行依据；
- 对已完成分诊的 `core-green` 节点启用 blocking；
- `contract-green` 单独报告，不能充当 kernel 覆盖；
- `diagnostic-known-fail` 只有 fingerprint 未变时允许继续观察，fingerprint 漂移、
  新增失败或节点消失应阻塞升级复核；
- family matrix 先 post-merge blocking，再考虑对高影响 PR 受控 blocking。

### 阶段 D：模型就绪

- L1.5 组件门通过后才执行 L2-F；
- L2-F 至少覆盖启动、eager 生成、graph 生成、checkpoint 字节绑定和独立精度参考；
- 模型通过不能反向替代缺失的组件 route evidence；
- 真实模型结果不得写成普通 L1.5 “全部通过”。

## 5. 资源与并发规则

- CPU job 可以并行；
- 单卡 L1/L1.5 job 使用专用 HCU runner，最多一个任务占用一张卡；
- 双卡通信 job 必须声明 `device_count=2`，不得从单卡 job 偷用第二张卡；
- runner 不得通过扫描显存后强行抢占正在运行的进程；
- 资源不足返回 queued/resource-unavailable，不能返回 PASS；
- 同一 adapter commit + SGLang commit + 容器版本的重复任务可以复用缓存，但必须在
  artifact 中声明 cache hit；
- CI 产生的报告和日志放在 job artifact 或仓库外结果目录，代码仓库只保存 manifest、
 规则和脱敏后的摘要。

## 6. 本阶段暂不做的事情

在范围确认前，不执行以下动作：

- 不在当前开发节点跑全量 adapter/GPU 测试；
- 不把 3292 个历史分类 nodeid 直接变成全部 blocking；
- 不把 58 个 family case 误称为真实模型覆盖；
- 不把 vllm 问题混入 adapter/SGLang 升级门；
- 不启用每日追新；
- 不新增包含真实节点地址、容器名、用户信息或模型路径的代码和 CI 配置。

## 7. 实施顺序

1. 先确认本规划中的分层、触发矩阵和 blocking 规则；
2. 将普通 PR workflow 接入 L0、L1-CPU 和 manifest 静态校验；
3. 新增受控 HCU workflow，先只做 L1-GPU 和 L1.5 observe；
4. 将 `l15_mech` 改为 exact nodeid 分类报告；
5. family matrix 连续稳定后从 observe 升为 post-merge blocking；
6. 新增独立 upstream-upgrade workflow；
7. 最后接入 cookbook 驱动的 L2-F 模型验证；
8. 其他功能和精度问题收敛后，再评估每日追新。

## 8. 当前用例审计结论

截至当前工作树的静态审计：

- `l15_v0522.json` 有 28 个候选源文件，路径没有重复；
- `l15_family_matrix_v1.json` 有 5 个家族、58 个 exact case，显式 nodeid 没有重复；
- `l15_comm_v1.json` 的通信 case 与 family matrix 没有 exact nodeid 重复；
- `l15_v0522_candidate_baseline.json` 与 `l15_v0522_classified.json` 保存的是同一候选体系
  的历史基线和分类快照，不是两套需要重复执行的测试；
- `l15_sglang_main.json` 没有实际 case，当前不能作为有效测试清单，应归档或改成升级时
  动态生成的临时 manifest。

因此，当前主要问题不是“manifest 中写了重复 case”，而是同一个 case 可能被多个入口重复
执行：

1. `l15_upstream` 按目录执行 attention core，`l15_mech` 按目录执行 mechanism；
2. `l15_v0522` 的 selectors 又覆盖这些 upstream 目录，并额外覆盖 Batch 1–4；
3. family matrix 复用了部分 upstream nodeid，同时包含 adapter-owned 的组件合同；
4. `ci/l1_gpu_gate.sh` 的 `tests/kernels tests/adapter` 宽目录执行会再次包含 family matrix
   中的 adapter-owned case。

这些重复在迁移期有价值，可以用于交叉验证；进入稳定 CI 后应改为“单一执行、多个报告”：

- 普通 L1 运行基础 adapter 回归；
- L1.5 manifest 运行精确 upstream nodeid；
- family matrix 只运行代表性组件合同；
- 同一 nodeid 在一次 job 中只执行一次，由报告聚合器把结果映射到多个覆盖域；
- 宽目录回归和 family matrix 不在同一个 blocking job 中重复运行。

当前 gate 仅在 family matrix 为 `blocking` 且上游树存在时，按
`L15_FAMILY_MATRIX_FAMILIES` 选中的家族排除宽目录中的 adapter-owned nodeid。
未选择的家族仍由宽目录执行；observe/off 或没有上游树时保留完整宽目录覆盖。
清单校验或家族选择错误在排除前响亮失败，不能静默减少覆盖。

当前高优先级缺口按顺序为：

1. 为 `core-green` 补真实 backend/route evidence，避免“测试通过但实际走 upstream 或
   torch fallback”；
2. 补 KV cache 的读写、页表、复用、边界和 graph replay 生命周期；
3. 补 sampling、scheduler、prefill/extend/decode 状态转换和 speculative/MTP 契约；
4. 将 FP8/INT8/W4A16/W4A8 的权重加载、scale/layout 和小模型端到端验证放入 L2-F；
5. 使用 cookbook 增加真实模型 launch、eager/graph generation 和独立精度参考；
6. 将 CP/EP/PD、通信、RDMA、外部存储继续保持在独立扩展层，不混入单卡 L1.5 core。

L1.6 的固定入口是 `ci/l16_model_readiness_gate.sh`，机器可读清单是
`tests/manifests/l16_model_readiness_v1.json`。它复用
`tools/l15_model_smoke.py` 的运行合同，但拥有独立的 manifest 和 artifact；
因此 L1.6 的通过只表示模型能够按指定配置加载并完成基础生成，不表示完整模型
精度、性能、并发或多机功能已经通过。

不建议为了“增加数量”直接扩大 L1.5。每个新增 case 都应先回答它覆盖的 logical op、
阶段、形状边界、预期 backend、是否需要模型/服务/多卡，以及现有 case 为什么不能覆盖；
否则应优先合并参数或删除重复的包装测试。



## CPU workflow 接线进度（2026-11-11）

`.github/workflows/adapter-cpu.yml` 将 L0、四份 manifest 校验、Shell 语法和显式 CPU
回归接到 PR、main push 和手动触发。runner 为托管 Ubuntu，安装 CPU Torch，
隐藏全部设备并禁用 pytest 第三方插件自动发现。测试报告以 JUnit artifact 保存。
这条 workflow 不涉及 HCU 分配；既有外部 Quality Gate 保留。

该显式测试清单在远端隔离快照、隐藏全部 GPU、禁用第三方插件自动发现的环境中
**407 passed、34 skipped**（包含双 wheel 构建/互锁、FP8 factory 别名与私有编译输入合同；GPU 探针不在
CPU 环境执行）。这验证了测试命令；全新托管 runner 的依赖安装和 GitHub 事件运行
仍须在推送后观察，不能把容器结果称为 GitHub Actions 已通过。

## 受控 HCU 接线进度（2026-11-11）

`.github/workflows/adapter-hcu.yml` 当前仅支持手动 `workflow_dispatch`，不自动申请
共享设备。专用 runner 使用 `self-hosted, linux, hcu-adapter` 标签，并通过
`hcu-validation` environment 的配置提供 `HCU_UPSTREAM_TREE`、`HCU_CHECKPOINT` 和
`HCU_GPU_IDS`。这些值不得写进仓库。管理员须事先独占保留设备；SMI 空闲检查只用于
拒绝已忙设备，不能代替资源租约，也不会自动换卡。

- `components`：先按选中的五家族生成 ownership，再执行严格 residual exact 门、
  独立 partial-subtest observation 和 58 case family matrix；仅允许单卡。
- `readiness`：执行六类 L1.6 eager/graph，用外部真实 checkpoint 完成权重绑定；
  TP 数量来自显式保留的设备集合。
- `upgrade`：要求 baseline/candidate ref 和真实 checkpoint，先检查静态接口漂移及
  实际 pristine HEAD，再执行组件矩阵与 L1.6。接口漂移未审阅时先失败，不进入 GPU 阶段。

严格 residual 清单与 family matrix 的 transferred nodeid 必须互斥并覆盖原 green
集合。当前分类修正后的 residual 为 153 个 nodeid，11 个由 family matrix 承担；
6 个包含通过和跳过 subtest 的节点单独观察。已确认全跳过的 19 个节点标记为
`platform-skip`，不能因为父测试返回 passed 就计入严格通过。

本地宽目录 gate 仅在 family matrix 为 blocking 且上游树存在时转移执行所有权，
off/observe 或未选择的家族保留原覆盖。当前 workflow 尚未在 GitHub 上实际运行；
完成专用 runner、environment 和外部资源保留配置后，先手动验证，再决定哪些 HCU
检查升为分支保护条件。每日追新和自动 post-merge HCU 调度仍未启用。
