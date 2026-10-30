# 测试、路由观测与代码退役治理方案（test-route-governance）

> 本文回答：**adapter 的测试体系如何从“精度能跑”演进到“功能可用、性能可比、
> 路由可审计、陈旧代码可退役”。**目标不是继续堆测试，而是让每次门禁都能回答
> 七个问题：对不对、能不能、快不快、实际走了谁、各档差多少、谁没有被路由、
> 谁已经可以删除。
>
> 配套事实源：后端目录 `sgl_kernel/_adapters/hcu_config.py::BACKEND_CATALOG`，
> 安装态注册表 `sgl_kernel/_adapters/registry.py`，分层门禁 [ci.md](ci.md)，
> 算子治理 [governance.md](governance.md)，本机选档 [autotune.md](autotune.md)。
> 本文是**目标方案与落地路线**；未标“已落地”的机制不得当作当前能力宣称。
> 可执行的 MR 拆分、依赖、验收命令和回滚见
> [test-route-implementation-plan.md](test-route-implementation-plan.md)。

## 0. 目标终态与五条硬原则

目标闭环：

```text
上游 sglang 测试 / 模型请求
        ↓
精度门 → 功能门 → 性能门
        ↓
运行态路由账（op / backend / shape / phase / count / reason）
        ↓
emu · triton · vendor 同窗 A/B
        ↓
report.json + report.md
        ↓
缺口补齐 / 档位晋级 / fallback 收敛 / 陈旧代码退役
```

五条 HARD 原则：

1. **正确性先于性能**：任何档位先过独立参照的精度门和 graph-replay 门，
   才能进入性能竞速；“更快但不对”不进入候选集。
2. **路由必须被证明**：环境变量或配置写了 `triton` 不等于实际走了 triton；
   运行报告必须给出可观察的 dispatch/调用证据。
3. **性能只做同窗比较**：同机、同卡型、同镜像、同代码、同 shape、同输入、
   同 graph 模式；性能数字不跨机器迁移。
4. **零命中不等于死代码**：先区分 missing / fallback / unobserved / orphan；
   只有无调用、无测试、无 owner、上游 target 已消失等证据闭环后才退役。
5. **生产路径零主机同步**：运行态观测不得在热路径引入 `.item()`、同步计时或
   破坏 graph capture；常态只计数，性能计时必须显式开启并采样。

## 1. 分层体系：L0 → L1 → L1.5 → L2-F → L2-P → L3

| 层 | 回答的问题 | 测试对象 | 资源 | 触发与判定 |
|---|---|---|---|---|
| **L0** | 结构和治理是否合法 | 语法、catalog/registry/docs、孤儿/重复/过期资产 | CPU、零外部依赖 | 每次提交；HARD 规则任一红即阻塞 |
| **L1** | adapter 自有实现是否正确 | kernel、bridge、bootstrap、配置、打包、graph safety | 1 卡，MR 级 | 独立参照 + shape/dtype 矩阵 + replay |
| **L1.5** | 接入社区 sglang 后真实调用面是否兼容 | 上游官方 pytest，固定 nodeid manifest | 1 卡，MR/追新 | `core-green` 硬门；其余三态分账 |
| **L2-F** | 模型和特性组合能否工作 | launch/生成/锚点/长文/Graph/TP·EP·PD 等 | 1–8 卡 | 扩展验证/发版；功能矩阵判定 |
| **L2-P** | 热点路径是否变快或回退 | op microbench + serving A/B | 1–8 卡 | 扩展验证/档位晋级；同窗阈值 |
| **L3** | 长期稳定性和复杂拓扑是否健康 | 压测、多机、RDMA、故障恢复 | 按场景 | 专项触发，不挂普通 MR |

### 1.1 L0 增强：从“结构合法”扩展为“资产可治理”

在现有 `glue-numerics` / `text-patch` / `registry-ops` 三条硬规约上，逐步增加：

- `catalog-unbound`：catalog 可选 op 无安装态绑定且无 planned 理由；
- `code-not-catalogued`：实现存在但没有 catalog/registry 身份；
- `duplicate-active`：同一 logical op 有多个 production active；
- `ref-routed`：ref 实现被 bridge 路由进生产；
- `target-missing`：上游 target/needle 已消失但桥或 text-patch 仍在；
- `env-declared-not-read`：环境变量只在文档/REGISTRY 中存在，代码无读取点；
- `testless-active`：active 实现没有精度判据；
- `deprecated-unresolved`：已标 deprecated，但退役证据或残留清理尚未闭环。

新增规则先以 WARN 观察，达到“结果稳定、误报归零、owner 确认”的晋级条件后再升
HARD；不得一上来用静态启发式批量删代码。

### 1.2 L1：每个 catalog op 的最低合格线

每个可运行 op 至少具备：

1. 一个独立 Torch/FP32 或上游参照的数值判据；
2. 一个真实 active 档位测试；
3. 一个 fallback/ref 判据；
4. 一个路由开关验证（显式强制档位后证明实际命中）；
5. decode 链路 op 的 graph capture→换新输入→replay；
6. shape 矩阵含最小 M、graph pad、奇数边界和真实 TP 切片。

断言沿用 [test-suite.md](../skills/hcu-adapter-dev/references/test-suite.md) 的
`cos + maxrel` 纪律；FP8、索引、MoE TopK 等使用各自语义指标，禁止用一个宽松
`assert_close` 覆盖所有类型。

### 1.3 L1.5 v2：从 Attention 精选扩到 adapter 主要调用面

当前已落地的 L1.5 主要是 `attention/unittests`：dense/dsa/kda/mla 与
gdn/hybrid_linear/mamba/swa/lightning/dsv4。它是有效基线，但结构明显偏科：能发现
attention backend 和机制层回归，不能系统发现 `sgl_kernel` 公共 ABI、基础 fused-op、
量化、KV Cache、Sampling、MoE 和底层 Conv1D 的上游漂移。

L1.5 v2 保持上游测试本体零拷贝，扩展目标不是“把 `test/registered` 全跑一遍”，而是
覆盖社区 sglang 经 adapter 进入 HCU 的主要调用面：

| 域 | 首批候选 | 主要验证 |
|---|---|---|
| Attention core | 现有 dense/dsa/kda/mla 的有效绿色 nodeid | backend 构造、metadata、FA3/MLA/DSA/KDA 路径 |
| Mechanism | gdn/hybrid_linear/mamba/swa/lightning/dsv4 的绿色子集 | 线性注意力、状态、窗口和 DSV4 接缝 |
| Activation/Norm | fused-op parity、RMSNorm、fused-add RMSNorm | `sgl_kernel` AOT ABI、backend dispatch、融合语义 |
| RoPE/Sampling | `test_rope_rocm`、renorm/top-p/temperature | HIP/native/vendor 路径和 sampling 影子面 |
| KV Cache | store-cache、MLA KV buffer、FP8 KV write | 索引、页表、布局、写入与 graph safety |
| Quantization | FP8 aiter utils、per-token/group quant | 顶层符号、scale/layout、bit/dequant 精度 |
| MoE | TopK padded/degenerate、align/gate | expert 选择、padding、renorm 和 graph replay |
| Attention kernels | KV indices、decode metadata、Triton kernels、verify KV | prefill/extend/decode 的底层真实调用面 |
| Mamba/Conv1D | causal-conv、SConv metadata/cache/replay | state/cache/页边界和 Conv1D 路由 |
| Model contracts | 无权重的构造、quant config、weight mapping/loader | 模型 API 漂移；只计 contract，不冒充 HCU kernel 覆盖 |

以下仍不进入 L1.5 core：完整 `launch_server`/HTTP/eval、模型权重和外部数据集、多卡
TP/EP/CP/PD、RDMA/NIXL/Mooncake/3FS、性能压测，以及只会稳定平台 skip 的
FA4/TRT-LLM/SM 专属/gfx950-only 路径；它们分别归 L2-F、L2-P、L3 或 skip 账。

### 1.4 L1.5 v2 分类与判定

收集集合使用按上游版本/commit 固定的机器可读 nodeid manifest：

| 集合 | 含义 | 是否阻塞 | 是否计入 HCU 有效覆盖 |
|---|---|---:|---:|
| `core-green` | 单卡、无服务、真实命中 adapter/HCU 路径、连续稳定全绿 | 是 | 是 |
| `contract-green` | metadata/构造/配置等上游契约有价值，但不一定执行 adapter kernel | 是或按域定 | 否，单列契约覆盖 |
| `diagnostic-known-fail` | 精确 nodeid + 错误指纹 + owner + 解除条件 | 否 | 否 |
| `platform-skip` | FA4/TRT-LLM/SM 专属/gfx950-only 等稳定平台性 skip | 否 | 否 |
| `unclassified` | 已收集但尚未完成依赖、路由和结果分诊 | 否，只 WARN | 否，必须有清零计划 |

每个候选至少登记：`nodeid / domain / ops / expected_backends / route_evidence /
device_count / needs_server / needs_model / external_deps / timeout / owner / current_result`。
没有 route evidence 的纯契约测试只能进入 `contract-green`；合法 skip 不进入有效覆盖分母。

过渡期保留 `passed≥N / failed≤M` 预算，但目标终态禁止只按总数放行：一个旧失败
修好、一个新失败出现时总数不变，会产生假绿。manifest 必须检查集合漂移、skip
理由漂移和错误指纹漂移。

### 1.5 扩充顺序与规模约束

L1.5 v2 按批次进入，不能一次把数百个候选挂到普通 MR：

1. **Batch 0 盘点**：对候选 collect-only，记录依赖、卡数、模型/服务需求、平台守卫、
   adapter seam、当前结果和耗时；
2. **Batch 1 基础 ABI**：activation/norm/RoPE/sampling/KV Cache；
3. **Batch 2 Quant/MoE**：先把 ABI 缺口作为 diagnostic，修复后转 core；
4. **Batch 3 Attention kernels**：KV indices、metadata、FP8 KV、verify 路径；
5. **Batch 4 Mamba/Conv1D**：state/cache/metadata/replay；
6. **Batch 5 Model contracts**：无权重构造和加载契约，单列 contract coverage。

MR 快车的 v2 `core-green` 首轮目标约 250–350 个稳定实例，保持单卡、可分段、可
配置超时；contract、diagnostic 和扩展矩阵由扩展验证入口运行。数量只是资源护栏，判定以调用面和
route evidence 为准，不以测试数作为覆盖成绩。

示例 schema（路径和 nodeid 以对应上游版本 collect-only 结果为准）：

```yaml
version: 1
upstream: sglang-v0.5.22
suites:
  l15_core:
    blocking: true
    device_count: 1
    tests:
      - nodeid: test/registered/attention/unittests/dsa/test_dsa.py::TestX::test_y
        ops: [paged_mqa_logits, flash_attention]
        expected_backends: [triton, emu]
        route_evidence: required
        owner: attention
        timeout_s: 300
  l15_diagnostic:
    blocking: false
    tests:
      - nodeid: test/...::test_x
        fingerprint: "llvm.call op operand type mismatch"
        owner: dtk
        review_condition: "DTK toolchain supports the required lowering"
        promotion_condition: "two stable green runs with route evidence"
```

## 2. 精度阶段：先把“对”做成机器事实

### 2.1 指标按语义选，不按文件统一

| 算子类型 | 主判据 | 补充判据 |
|---|---|---|
| BF16/FP16 普通算子 | cosine + maxrel/atol | shape/dtype/原地语义 |
| FP8/INT8 量化 | scale allclose + bit-match | dequant cosine、饱和边界 |
| 索引/布局/页表 | bit exact 或集合/排列约束 | sentinel、奇数页、越界护栏 |
| MoE TopK | expert 集合、权重、padding、tie 规则 | shared expert、非法 expert |
| Attention | output cosine/maxrel + LSE | causal/window/GQA/varlen/FP8 KV |
| Graph | capture 成功且 replay 跟随新输入 | 多次 replay、地址稳定性 |
| 模型 e2e | 文本金标/语义锚点/分布级判定 | 长文事实、并发零串扰 |

### 2.2 测试下沉顺序

```text
CPU/ref → adapter kernel → 上游调用面 → 小模型 e2e → 大模型/复杂功能
```

模型或 L1.5 发现的每个适配根因，必须补一个更低层的离线复现（one pit, one test）；启动脚本里的规避参数不能替代回归测试。

### 2.3 精度报告最低内容

每条结果至少包含：`op/backend/dtype/shape/reference/metric/tolerance/result`，并绑定
adapter commit、sglang commit、arch、DTK/Torch/Triton 和测试 nodeid。报告没有
环境指纹时，数值不具备可复验性。

## 3. 功能阶段：L2-F 组合矩阵

精度通过后再扩功能，按维度组织而非为每个模型复制一份脚本：

- 阶段：warmup / prefill / extend / decode / graph capture / replay；
- 结构：dense / MoE / MLA / DSA / KDA / GDN·Mamba / hybrid / SWA；
- 数值：BF16 / FP16 / FP8 / INT8 / W4A16 / W4A8 / FP8 KV；
- 并行：TP / DP-attention / EP / CP / PD / MTP；
- 缓存：Radix / HiCache / KV transfer。

每个功能场景输出统一判定：

```json
{
  "launch": "pass",
  "generation": "pass",
  "accuracy": "pass",
  "long_context": "pass",
  "graph_capture": "pass",
  "route_coverage": 0.98
}
```

L1.5 只收单卡、无 server、无外部资产的官方单测；完整模型、HTTP、多卡、RDMA、
外部存储和数据集归 L2-F/L3，不得为了扩大 L1.5 数字混入。

## 4. 运行态路由账：安装了谁 ≠ 实际调用了谁

现有 registry 记录**安装态**：哪个 target 被哪个 implementation 通过 seam/rebind/
text-patch 接管。新增的 route telemetry 记录**运行态**：请求真正调用了谁、多少次、
什么 shape、处于哪个阶段、为什么 fallback。

### 4.1 统一 backend 分类

只允许：

- `vendor`：aiter / lightop / flash_mla / deepgemm 等；
- `triton`：adapter 自研或移植的 Triton 实现；
- `emu`：为行为还原拆解/模拟的路径；
- `torch`：参考或保底路径；
- `upstream`：未替换，直接走社区实现；
- `stub`：符号可 import、真实调用响亮失败；
- `unknown`：无法归因，治理目标必须为 0。

`emu` 不与 `torch` 合并：emu 可能由多颗 Triton/vendor/Torch kernel 组成，其性能
和退役意义不同。

### 4.2 事件 schema

```json
{
  "op": "silu_and_mul",
  "backend": "triton",
  "implementation": "sgl_kernel.elementwise.silu_and_mul",
  "vendor_version": null,
  "phase": "decode",
  "dtype": "bfloat16",
  "shape_signature": {"m": 1, "hidden": 7168},
  "graph_mode": "replay",
  "count": 128,
  "latency_us_sum": 5320,
  "latency_us_p50": 40.8,
  "latency_us_p95": 44.2,
  "reason": "catalog-default",
  "evidence": "<pytest nodeid or model request id>"
}
```

### 4.3 两种工作模式

**常态计数模式**：仅累计 `op/backend/shape_bucket/phase/count/reason`，不做 GPU
同步；进程退出或显式 flush 时写 JSON。该模式可进扩展验证和 L2，不默认挂普通生产。

**性能采样模式**：显式 `HCU_ROUTE_PROFILE=1` 后启用 Event 计时，并用采样率
控制开销；只在 benchmark/扩展验证使用。计时窗口内禁止每次 synchronize，统一在
采样批次结束后收割。

### 4.4 Graph 计数口径

必须区分：

```text
python_call_count / capture_count / replay_count / logical_call_count
```

Python wrapper 可能只在 capture 时执行一次，而图会 replay 数千次；只统计 wrapper
会严重低估路由频次。首版先保证 capture 与普通调用分账，第二版在 graph runner/
scheduler seam 补 logical replay 乘数。

## 5. 性能阶段：emu / triton / vendor 同窗竞速

### 5.1 Shape 来源与标准集

每个 op 的 benchmark shape 取四类并集：

1. 上游官方测试的契约形状；
2. 已验证模型 route telemetry 的高频形状；
3. graph bucket 与 decode `M=1`；
4. 奇数边界和 TP/EP 切片后的真实尺寸。

shape 集进入机器可读 catalog，并记录来源；不得只挑对某档有利的理想尺寸。

### 5.2 执行顺序

```text
强制 emu/torch → 精度门
强制 triton    → 精度门
强制 vendor    → 精度门
所有可用档精度绿 → 同输入同窗 benchmark
```

强制档位后必须有 route evidence；“配置写了 vendor”但运行计数为 0 的 benchmark
判无效。

### 5.3 性能指标

至少记录：首次编译时间、冷/暖启动、warmup/iters、p50/p95/max、吞吐、显存峰值、
workspace、graph capture/replay、shape 覆盖率，以及相对当前 production 的加速比。
JIT cache seed 必须进入环境指纹；冷编译成本与暖态执行时间分列，不能混成一个数字。

### 5.4 档位晋级

| 状态迁移 | 必要条件 |
|---|---|
| candidate → validated | 算子精度矩阵、graph replay、对应 L1.5 全绿；性能不差于 production；连续两跑稳定 |
| validated → production | 目标模型 L2-F 全绿；目标 shape 有收益；显存可接受；有 kill switch 和回退档 |
| production → deprecated | 新档完全覆盖且稳定；保留旧档直到回退观察条件全部满足 |
| deprecated → retired | 无活跃调用、无回退需求、配置/文档/测试/代码全链删除 |

性能阈值按 op 定义；首版可用“不得回退 >5%，热点至少提升 10%”作为模板，但不得
写成全仓统一铁律。小 kernel 的噪声和大 GEMM 的收益尺度不同。

## 6. 未路由与缺口分类

| 类别 | 定义 | 处置 |
|---|---|---|
| `missing` | 上游真实调用触发 stub/NotImplementedError | 阻塞对应支持声明；按 lightop→aiter→triton→stub 顺序进入实现队列 |
| `fallback` | 成功执行但走 upstream/torch/emu/低性能拆解 | 功能可过；按频次×耗时排性能优先级 |
| `unobserved` | catalog 有实现，但当前矩阵没有触发 | 先补场景/确认上游调用点；不得直接删除 |
| `orphan/dead` | 无 catalog/registry/test/import，或 target 已消失 | 进入退役候选，按 §7 完成三证复核 |
| `unknown` | 真实调用无法归入任何标准 backend | 治理缺陷；核心报告中必须为 0 |

算子需求优先级使用 `调用频次 × 单次代价 × 影响面`，数据进入
[operator-requests.md](operator-requests.md) 的现状账；路线图阻塞可在无生产频次时
替代，但要注明证据等级。

## 7. 代码退役：防止陈旧代码和重复实现堆积

### 7.1 生命周期补全

在现有 `production / validated / candidate / broken / planned` 基础上增加：

- `deprecated`：已由新实现覆盖，仍保留为回退路径，直至退役三证闭环；
- `retired`：实现已删除，只在迁移/历史账中留记录，不可选。

每个实现逐步补齐：`owner / introduced / last_verified / last_hit_release / tests /
replaces / remove_after`。owner 允许是模块/团队，不强制个人名字。

### 7.2 扩展资产对账

生成以下清单：

```text
catalog-unbound
code-not-catalogued
registered-not-hit
tested-not-hit
hit-not-tested
duplicate-active
ref-routed-production
text-patch-target-missing
env-declared-not-read
deprecated-expired
```

其中 `registered-not-hit` 只说明覆盖不足；只有同时满足“完整矩阵零命中 + 上游 target
消失/无调用点 + 无测试需求 + 无 owner/保留理由”才升级为 dead 候选。

### 7.3 退役流程

```text
发现候选 → owner/调用点复核 → 标 deprecated
        → 完整测试矩阵连续稳定且无活跃调用
        → route/test/upstream 三证确认无需求
        → 同 MR 删除实现、import、catalog、env、配置、文档和测试
        → L0 验证零残留
```

删除必须全链完成；只删实现文件、留下环境变量或文档，等价于制造下一批陈旧资产。

## 8. 统一报告：一次运行产两件

每次 gate/扩展验证输出：

```text
report.json   # 机器消费、趋势对比、门禁判定
report.md     # 人工评审、MR 评论、夜报
```

### 8.1 固定章节

1. **环境指纹**：adapter/sglang commit、镜像、arch、DTK、Torch、Triton、vendor
   版本、GPU、cache seed；
2. **测试结果**：pass/fail/skip/xfail，新失败与 known-fail 分离，manifest 漂移；
3. **精度**：op/backend/dtype/shape/metric/tolerance；
4. **功能**：模型、量化、Graph、TP/EP/PD 等矩阵；
5. **路由**：次数、shape、phase、fallback reason、missing/unknown；
6. **性能**：各档 p50/p95、显存、编译时间、加速比；
7. **治理**：orphan/stale/duplicate/deprecated、owner、复审条件与建议动作。

### 8.2 路由汇总表

| Op | 总调用 | emu | triton | vendor | torch | upstream | stub | unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `<op>` | `<n>` | `<n>` | `<n>` | `<n>` | `<n>` | `<n>` | `<n>` | `<n>` |

核心门的 `unknown` 必须为 0；`stub>0` 是否阻塞按测试声明，但不能静默；fallback
必须有 reason。

### 8.3 性能汇总表

| Op | Shape/阶段 | emu p50 | triton p50 | vendor p50 | production | 最优档 | 加速比 |
|---|---|---:|---:|---:|---|---|---:|
| `<op>` | `<shape>` | `<us>` | `<us>` | `<us>` | `<tier>` | `<tier>` | `<x>` |

## 9. 执行节奏与资源

| 触发 | 必跑 | 目标资源 |
|---|---|---|
| 普通 MR | L0 + L1 + L1.5 core + 路由覆盖检查 | 1 卡；可分段、可中断 |
| 扩展验证 | L1 全量 + L1.5 core/contract + 小模型 L2-F + op 性能采样 + 资产扫描 | 1–4 卡 |
| 发版/专项窗口 | 大模型/多卡功能矩阵、扩展性能矩阵、deprecated 审核、cache seed 更新 | 共享资源窗口 |
| 上游版本切换 | upstream diff、完整 L1.5、关键模型 L2-F、热点 L2-P、路由差异 | 专门窗口 |

JIT 预编译包只能缩短时间，不能替代精度与路由验证。cache seed 必须匹配 arch、
镜像、DTK、Torch、Triton、sglang/adapter commit；不匹配时允许自然重编，禁止强行
复用旧 `.so` 制造假绿。

## 10. 分阶段实施计划

### P0：可观测底座

- 定义 route event schema 与 backend 枚举；
- 建立 L1.5 v2 候选 inventory，并固化版本化
  `core/contract/diagnostic/skip/unclassified` manifest；
- 现有 Attention/Mechanism 先精确化，新增基础 ABI、Quant/MoE、Attention kernels、
  Mamba/Conv1D 按 observe 批次进入；
- 扩展 registry dump：安装态与运行态分区；
- 建立环境指纹和统一 `report.json` 骨架；
- 只计数，不做热路径计时。

**验收**：现有 L1/L1.5 与 v2 inventory/manifest 可稳定重跑；已埋点试点 op 能给出
backend/count/reason 且 `unknown=0`，未埋点 op 明确标 `not_instrumented`；集合、skip 和
错误指纹漂移可见。

### P1：精度硬门

- 首批 10 个高频 op 补齐参照、shape、graph 和路由强制判据；
- known-fail 从总数预算迁到精确 nodeid+指纹；
- 补齐上游 v0.5.22 已暴露的公共 ABI 缺口；
- 建立精度结果的 JSON schema。

**验收**：`core-green=100%`；新增失败不能被旧失败数量抵消；每个 active 有测试。

### P2：功能矩阵

- 小模型 dense/MoE/混合注意力三条 smoke；
- Graph、量化、KV、MTP、TP/EP/PD 按层接入；
- route coverage 与功能结果绑定；
- 模型脚本仍按 `tests/tools` 命名规约管理。

**验收**：功能报告可回答“哪个组合失败、失败前实际走了哪些 backend”。

### P3：性能与路由竞速

- shape catalog 与 telemetry 高频形状回灌；
- emu/triton/vendor 强制路由 runner；
- Event 采样计时、冷编译与暖态分账；
- candidate→validated→production 自动产证据包，最终晋级仍需 review。

**验收**：首批 10 op 具备可复验三档 A/B；性能回退和档位收益可机器判读。

### P4：报告与退役治理

- `report.md` 自动生成并接 MR/扩展验证；
- stale/orphan/duplicate 扫描从 WARN 升 HARD；
- 完成一次旧实现全链退役演练；
- deprecated 项满足退役三证后触发审核。

**验收**：孤儿实现 0、重复 active 0、ref 生产路由 0；删除无残留。

## 11. 首批闭环范围与任务拆分

首批不要追求全 catalog，选择高频且已有测试基础的 10 个 op：

```text
silu_and_mul / rms_norm / rotary_embedding / per_token_quant /
kv-cache store / flash_attention / paged_mqa_logits /
moe_gate / moe_align / conv1d
```

工程任务建议：

| ID | 任务 | 产物 |
|---|---|---|
| M1 | L1.5 manifest 与 collect-only 漂移检查 | `core/contract/diagnostic/skip` 清单 |
| M2 | registry 安装态报告增强 | active/unbound/target/source 统一输出 |
| M3 | runtime route counter | op/backend/shape/phase/count/reason |
| M4 | graph logical replay 计数 | capture/replay 分账 |
| M5 | benchmark shape catalog | 上游形状+生产高频形状 |
| M6 | 三档强制路由 runner | emu/triton/vendor 正确性与性能 |
| M7 | 统一报告生成器 | JSON + Markdown + MR 摘要 |
| M8 | stale/orphan/duplicate 扫描 | 扩展资产账与 L0 WARN/HARD |
| M9 | MR/扩展验证/发版三套入口 | 分层资源与超时策略 |
| M10 | 首批 10 op 闭环与退役演练 | 一次完整晋级、一次完整删除 |

## 12. 反模式

- 只看配置值就宣称“已走 vendor”，没有运行态证据；
- 把合法 skip 算成 HCU 适配覆盖；
- 用 passed/failed 总数掩盖 nodeid 集合变化；
- 性能测试不先过精度门，或跨机器比较数字；
- 在每次调用中 synchronize/`.item()` 做统计，破坏 graph 和真实性能；
- 仅凭一次零命中删除代码；
- 新实现只加代码，不加 catalog、测试、owner、快关和报告身份；
- 删除实现却保留 env/config/docs/import，制造“幽灵功能”；
- 把完整模型、多卡、外部服务测试硬塞进 L1.5，拖慢 MR 且难归因。

## 13. 文档维护与失效信号

本文随机制落地更新，以下任一发生即说明文档需同 MR 修订：

- CI 层级或触发策略变化但 §1/§9 未更新；
- route event 字段或 backend 枚举与实现不符；
- catalog 生命周期增加/删除状态但 §5/§7 未同步；
- 报告 schema 改动但 §8 未同步；
- 首批 op 或任务拆分完成后仍停留在计划态；
- L1.5 继续使用总数预算，但文档误称已经实现精确 manifest。
