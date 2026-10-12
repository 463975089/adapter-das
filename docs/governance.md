# 算子治理规约（governance）

> 本文回答两个问题：**一个算子需求进来，落在哪一层、谁拥有、怎么证明它对。**
> 以及本仓的**目标终态**：adapter 的职能是让开源 sglang 在 HCU 上的行为与
> AMD/CUDA 参照一致——做影子库（同名包补齐 AOT 算子面）、算子路由（三档
> triton→vendor→torch 与 catalog 裁决）、质量检测（分级门禁回归），并在
> 行为一致的约束下最大化性能。**行为还原**所需的桥（换绑、断言放宽、
> 判定词 wrap）属于 adapter 本职，落在本仓；**上游没有的功能**（新算子、
> 新后端、新机制）以 **plugin 形态**承载——plugin 是与 adapter **不同的
> 代码仓**，经上游 seam / 注册机制接入，其代码不进本仓。两条线如果混进
> glue 单体，就是历史上"同一个算子好几个库都有"的土壤。
> 配套机制代码：`sgl_kernel/_adapters/registry.py`（运行时注册表 + dump）、
> `ci/l0_gate.py`（三条静态规约）。本文的每条 HARD 规则都应能被 L0 检查。

## 0. 目标终态（sitecustomize 收敛 + 双仓边界）

- sitecustomize.py 的终态定位是**编排器**（<200 行：探测 → 按序调用各
  域桥的 `install()` → log_summary/dump），不含任何具体桥的实现体。
- 判据一句话：**改官方代码行为**（换绑、断言放宽、判定词 wrap）是行为
  还原手段，其实现体留在本仓、文件化到 `_adapters/<域>.py`；sitecustomize
  里只准剩调用行。
- **双仓边界**：为行为一致服务的改动归 adapter 仓；上游没有的新能力归
  plugin 仓——**plugin 的目的同样是性能优化**（算子融合、专项内核、新调度
  路径等官方代码没有的功能），只是它以"新增能力"的形态存在，与 adapter 的
  "补齐既有行为"形态不同。功能扩展的 MR 不进 adapter 仓。判别时问一句：
  "上游在 AMD/CUDA 上有没有这个行为？"有而我们没有 → adapter 补齐到一致
  （其中性能优化照做：三档档内演进、vendor 竞速）；上游也没有（哪怕动机
  就是性能，如算子融合）→ plugin 仓。
- 现存长桥按 architecture.md §9 的 P1-P3 迁出，每次迁移容器 e2e 复验；
  **不新增**：review 拒收往 sitecustomize 加实现体的 MR（L0
  `glue-numerics` 是下界，不是目标线）。

## 1. 分工与授权边界

| 层 | 仓库 | 允许 | 禁止 |
|---|---|---|---|
| **内核库** `sgl_kernel/*.py` | adapter | 算子的数值实现（triton 首档 / vendor 封装 / torch 兜底） | 依赖 sglang 内部模块（保持 framework-free） |
| **域桥** `_adapters/<域>.py` | adapter | 行为还原 glue 的实现体：函数换绑、签名适配、断言放宽、判定词 wrap——install() 入口 + 开关 + register 登记 | **任何数值逻辑**（数学放内核库）；上游没有的新能力（那归 plugin 仓） |
| **编排器** `sitecustomize.py` | adapter | 探测 + 按序调用各域桥 `install()` + log_summary/dump | 实现体（目标 <200 行；§0） |
| **上游 seam** | sglang | 正式插件接入（`--quantization`、attention-backend registry、moe-runner-backend 等） | — |
| **plugin** | **独立仓** | 上游没有的新能力，目的即性能优化：算子融合、专项内核、新后端、新调度机制；优先经上游 seam 接入，无 seam 时提供自己的注册入口 | 把行为补丁反向塞回 adapter 职责面 |

一句话红线（L0 `glue-numerics` 强制）：**算子数学只准出现在内核库；
sitecustomize.py 不 import triton、不出现 `@triton.jit`、不引用 `tl.*`。**

## 2. 算子的两个角色（解决"重复实现"）

同一符号允许最多两份实现，但**角色不同、登记在册**：

- **`active`（生产指针）**：运行时真正服务的实现。每个 catalog 算子**至多一个**
  active 绑定——上游实现能用，active 就指上游（我们的 triton 不重写）。
- **`ref`（诊断指针）**：framework-free 参照实现 + 数值对照测试
  （`tests/kernels/`），用于不拉起 sglang 复现精度问题。**ref 永远不准被
  桥接进生产路径**；它存在的意义就是当 active 可疑时的裁判。

常被误认为"重复"的典型场景由此正名：sglang 自带 triton 核 + 我们 sgl_kernel
也有一份——前者若是 active，后者必须是 ref（且带离线复现测试），或者反之；
两者都是 active、或有一份无角色，即是治理违规。

**ref → active 升级**就是"坏算子写 triton 绕过"的操作化定义：带已过的数值
门禁把注册表的生产指针从上游实现改指我们的实现，catalog 状态随迁
`candidate → production`，同一 MR 内完成（转正与退役条款见 §5）。

## 3. 实现手段裁决链（新需求按序选位，不许跳级）

1. **上游 seam 优先**。上游给了正式挂载点就绝不 monkeypatch。这是唯一
   无技术债的位置。plugin 仓提供的新后端/新机制也优先为此准备 seam 实现。
2. **rebind（函数换绑，adapter 仓内）**：上游无 seam 但入口是纯 Python，
   且缺失的行为上游在 AMD/CUDA 上存在。实现体全量放本仓
   `_adapters/<域>.py`（内核函数在 `sgl_kernel/`，域桥只做绑定动作），
   `install()` 入口 + env/yaml 开关 + 安装时 `registry.register(...)`
   登记，sitecustomize 里只留一行调用。
3. **text-patch（源码文本替换+重编译，adapter 仓内）**：断言放宽类。
   **白名单制**：每个 needle 必须登记在 `registry.py` 的
   `TEXT_PATCH_ALLOWLIST`（含 op 与上游原文），新增条目 = 治理评审 =
   MR 说明"为何 1/2 不可行"。L0 `text-patch` 规约扫 sitecustomize 里的
   `.replace("assert _is_cuda...")` 目标串，不在册即红。当前在册 2 处：
   fused_moe swiglu-clamp 断言放宽（GLM-5.3-Flash/DeepSeek-V4 MoE）、
   prefill-CP rope 守卫断言。

**需求先过双仓判别**（§0）：上游 AMD/CUDA 上也有的行为缺口才走 2/3；
上游同样没有的能力不修桥、不放宽断言，落 plugin 仓经 seam 接入或直调
底层库——即便动机是性能优化（算子融合等）也不例外：**性能不决定选仓，
"上游有没有这个行为"才决定**。裁决链 2/3 的产物是"让 HCU 表现得像
参照平台"，不是"绕过参照平台的缺失"。

全局翻转平台身份（把 `is_hip` 判定改成 `is_cuda`）**禁止**：它把所有
`elif _is_cuda:` 的不可移植分支（flashinfer/trtllm 独有符号）一并打进
导入路径，且砍断依赖 `is_hip()=True` 的正常分支（fnuz 表、gfx workaround）。
逐条放宽 + 已验证实现背书，是一个个来的。

## 4. 运行时注册表与 dump（"现在谁在跑"必须一行可答）

- 每个成功安装的绑定调用 `registry.register(op, impl, target, mechanism)`。
  `mechanism ∈ {seam, rebind, text-patch}`；target 是调试器可下断点的
  点符号。register 自带防重复（同 target 换 op 抛错）、catalog 交叉校验
  （op 必须在 `BACKEND_CATALOG`，无 catalog 条目的绑定不存在）。
- **人读**：启动日志 `HCU adapter runtime:` 每算子一行
  （`op <- impl [role/mechanism] via target`），接在 `HCU adapter env:` 之后。
- **机读**：`HCU_ADAPTER_DUMP=路径` 启动时写 JSON（算子→活跃实现+机制+来源、
  未绑定的 catalog 算子清单、全部 env 生效值+来源）；`=1` 打进日志。
  CI 归因脚本与配置级二分（一次翻一个 catalog 条目重跑 e2e）消费它。

### CI 测试分层（归因金字塔）

| 层 | 内容 | 成本 | 回答 |
|---|---|---|---|
| L0 静态 | 三条治理规约 + 既有结构检查 | 秒级 | 结构烂没烂 |
| L1 单算子 | ref_impl 的 framework-free 数值对照 | 分钟级 | **这个算子数学错没错** |
| L1.5 | 单模型单路径 e2e（现行 GPU 门禁） | ~10min | 方案整体通不通 |
| L2 组合 | 垫片×后端矩阵，**受影响切片触发**（变更碰算子 X → 只跑含 X 的方案，映射由 catalog 推） | 按需 | 叠加有没有互相污染 |

组合测出问题不重跑矩阵：从 dump 看现场 active 实现 → yaml 一次翻一个条目
二分，每次十几分钟，天然把问题限制在小范围。

## 5. 生命周期条款

- **注册即存在**：不在 catalog、不在注册表的实现，等于不存在——review 和
  排障都不认。
- **退役**：上游官方支持某算子时，对应桥的删除与默认路径切换必须在同一个
  MR；catalog 状态 `production → deprecated → removed`，不许留"注释掉的桥"。
- **变更面**：改 KERNEL 实现 → 跑该算子 ref 测试 + L1.5；改 sitecustomize/
  `_base.py`/注册表 → 加 e2e 冒烟（CONTRIBUTING §1 既有规则）。

## 6. L0 静态规约速查（ci/l0_gate.py）

| 检查名 | 级别 | 规则 |
|---|---|---|
| `glue-numerics` | HARD | sitecustomize.py 无 triton import/jit/`tl.`（注释里的提及不触发） |
| `text-patch` | HARD | 每个 `_is_cuda` 系 `.replace` needle 都在 `TEXT_PATCH_ALLOWLIST` |
| `registry-ops` | HARD | 每个 `registry.register(...)` 的 op 都是 catalog 条目 |
