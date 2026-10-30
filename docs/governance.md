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
> 配套机制代码：`hcu_adapter/registry.py`（运行时注册表 + dump）、
> `ci/l0_gate.py`（三条静态规约）。本文的每条 HARD 规则都应能被 L0 检查。

## 0. 目标终态（薄入口 + 宿主分发 + 双仓边界）

- `sitecustomize.py` 固定为三行自动加载入口，只调用
  `hcu_adapter.bootstrap.install()`；宿主探测和 common/host 顺序由
  `bootstrap.py` 负责，宿主实现位于 `hcu_adapter/bridges/`。
- 判据一句话：**改官方代码行为**（换绑、断言放宽、判定词 wrap）是行为
  还原手段，其实现体留在本仓、按宿主放入 `hcu_adapter/bridges/`，按算子/
  机制放入 `hcu_adapter/<域>.py`；`sitecustomize` 里只准保留入口。
- **双仓边界**：为行为一致服务的改动归 adapter 仓；上游没有的新能力归
  plugin 仓——**plugin 的目的同样是性能优化**（算子融合、专项内核、新调度
  路径等官方代码没有的功能），只是它以"新增能力"的形态存在，与 adapter 的
  "补齐既有行为"形态不同。功能扩展的 MR 不进 adapter 仓。判别时问一句：
  "上游在 AMD/CUDA 上有没有这个行为？"有而我们没有 → adapter 补齐到一致
  （其中性能优化照做：三档档内演进、vendor 竞速）；上游也没有（哪怕动机
  就是性能，如算子融合）→ plugin 仓。
- 现存 `bridges/sglang.py` 长桥按 architecture.md §9 的 P1-P3 迁出，每次
  迁移容器 e2e 复验；**不新增**：review 拒收往 sitecustomize/bootstrap
  加实现体或往宿主桥加数值内核的 MR（L0
  `glue-numerics` 是下界，不是目标线）。

## 0.5 平台语义路线（2026-10-12 定）：走 is_cuda 分支，不走 is_hip 分支

**目标**：新模型/新功能在上游总是先落在 CUDA 分支（背后是 flashinfer /
trtllm 等 NVIDIA 生态）——卡商的核心竞争力就是新模型覆盖速度。因此本仓的
平台路线定为：**上游的平台判定一律回答 CUDA 语义**（`_is_cuda`、capability、
版本探测由 adapter 平台层统一接管），让新模型代码**不经适配天然进入
CUDA 分支**；**不维护独立的 is_hip 分支路线**。

**HCU 没有的 NVIDIA 库（flashinfer / trtllm / marlin 等），靠影子库四级
路由接住**——影子库不走"适配 HIP 生态分流"的路线，走路由路线：守住
CUDA 语义的调用面，把每个符号路由到 HCU 侧最合适的实现：

| 级 | 归宿 | 说明 |
|---|---|---|
| ① | lightop | vendor 首选：DTK 官方维护，长期成本最低 |
| ② | aiter | vendor 次选：语义对上即接（含签名适配） |
| ③ | **triton 自研先行** | fp32 累加、graph-safe——新模型"先跑通"，正确性优先 |
| ④ | 响亮 stub | import 成功、调用报错带指引；永远不做静默假实现 |

**语义护栏（强制白名单）**：个别判定翻转后会静默改变数值语义，必须保持
已验证的 HIP 语义、不得随 CUDA 语义翻转——

- `_is_fp8_fnuz`：HCU 实测走标准 e4m3fn（False），已过 fp8 全链锚点；
  翻转会错配 ROCm fnuz 约定的 vendor config 表 → **白名单锁定**；
- fp8 scale 布局（row/tensor-wise vs 128-block）等已定案量化契约同理；
- 白名单本身登记在平台层（随 P1 拆分收口成单点），每条带锚点出处。

**与现有形态的关系**：
- 已验证模型的生产部署**行为不变**——翻转的是"上游问我们是不是 CUDA"，
  数值语义由护栏锁定，已过的锚点不回头重验；
- 现存的"放宽 is_cuda 判定"类桥（capability 桥、AOT spec 桥等）由平台层
  统一接管、逐步退役（P1 拆分收口）；
- 性能在"跑通"之后按 §3 裁决链逐点做：triton 先行产物转 ref 备胎，
  lightop/aiter 等价核就位后路由切 vendor（catalog 状态演进）。

**落地形态（2026-10-12 bw7 首验）**：`hcu_adapter/platform_seam.py`
（编排器在 sitecustomize 最先调它——`common.py` import 时就用 is_hip()
冻结 FP8_E4M3_MAX，判定函数必须抢在所有 sglang import 前换掉）。合成态
三原则：**模型层 CUDA 语义 + 内核层 HIP 真值 + 硬件层不动**——只翻
sglang 的判定 seam（is_cuda/is_hip/alike、platforms 解析、
is_flashinfer_available、external-event 口、CUDA 版本特性门统一 False、
FP8 常量重算 fn），torch.version / triton 编译后端 / vllm HIP 构建 /
kernels 选核层全保持真值。开关 `HCU_PLATFORM_SEMANTICS=cuda|hip`（hip=
legacy 路线，A-B 对照）。is_cuda 语义放大的缝隙类型（缺 vendor 库裸
import、特性探测 hasattr 恒真、dlpack/numpy 标量、JIT hipcc nvcc-isms）
逐点按 §3 裁决链落桥，registry dump 逐条可见。

**验收口径**：新模型在 CUDA 语义下过四判定（启动/生成/精度/长文）即算
支持达成，性能不设门槛。stub 每次真实触发记录为**缺口清单**条目
（符号+调用点+频次）——这张表就是 triton 排期与向 DTK 提 vendor 需求的输入。

## 1. 分工与授权边界

| 层 | 仓库 | 允许 | 禁止 |
|---|---|---|---|
| **内核库** `sgl_kernel/*.py` | adapter | 算子的数值实现（triton 首档 / vendor 封装 / torch 兜底） | 依赖 sglang 内部模块（保持 framework-free） |
| **宿主桥** `hcu_adapter/bridges/<host>.py` | adapter | 宿主级行为还原 glue、ABI 适配和安装顺序；common 只放跨框架守卫 | 数值内核；把一个宿主的补丁装进另一个宿主；上游没有的新能力 |
| **算子/机制桥** `hcu_adapter/<域>.py` | adapter | 函数换绑、断言放宽、判定词 wrap——install() 入口 + 开关 + register 登记 | **任何数值逻辑**（数学放内核库）；上游没有的新能力（那归 plugin 仓） |
| **分发器** `hcu_adapter/bootstrap.py` | adapter | 探测宿主 + 先 common 后唯一宿主桥；失败隔离 | 具体桥实现、宿主 eager import、数值逻辑 |
| **自动入口** `sitecustomize.py` | adapter | `from hcu_adapter.bootstrap import install; install()` | 探测、分支、桥实现 |
| **上游 seam** | sglang | 正式插件接入（`--quantization`、attention-backend registry、moe-runner-backend 等） | — |
| **plugin** | **独立仓** | 上游没有的新能力，目的即性能优化：算子融合、专项内核、新后端、新调度机制；优先经上游 seam 接入，无 seam 时提供自己的注册入口 | 把行为补丁反向塞回 adapter 职责面 |

一句话红线（L0 `glue-numerics` 强制）：**算子数学只准出现在内核库；
`sitecustomize.py`、`hcu_adapter/bootstrap.py` 与 `hcu_adapter/bridges/*.py`
不 import triton、不出现 `@triton.jit`、不引用 `tl.*`。**

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
   `hcu_adapter/bridges/<host>.py` 或 `hcu_adapter/<域>.py`（内核函数
   在 `sgl_kernel/`，桥只做绑定动作），
   `install()` 入口 + env/yaml 开关 + 安装时 `registry.register(...)`
   登记，由宿主桥在原顺序点调用。
3. **text-patch（源码文本替换+重编译，adapter 仓内）**：断言放宽类。
   **白名单制**：每个 needle 必须登记在 `registry.py` 的
   `TEXT_PATCH_ALLOWLIST`（含 op 与上游原文），新增条目 = 治理评审 =
   MR 说明"为何 1/2 不可行"。L0 `text-patch` 规约扫全部 routing glue 里的
   `.replace("assert _is_cuda...")` 目标串，不在册即红。当前在册 2 处：
   fused_moe swiglu-clamp 断言放宽（GLM-5.3-Flash/DeepSeek-V4 MoE）、
   prefill-CP rope 守卫断言。

**需求先过双仓判别**（§0）：上游 AMD/CUDA 上也有的行为缺口才走 2/3；
上游同样没有的能力不修桥、不放宽断言，落 plugin 仓经 seam 接入或直调
底层库——即便动机是性能优化（算子融合等）也不例外：**性能不决定选仓，
"上游有没有这个行为"才决定**。裁决链 2/3 的产物是"让 HCU 表现得像
参照平台"，不是"绕过参照平台的缺失"。缺失的能力若属于**外部算子组/
供应商的交付范围**（向其提需求而非自己实现），按
[operator-requests.md](operator-requests.md) 标准提报——契约锚定上游
sglang 符号面，shim 妥协不进契约。

全局翻转平台身份（把 `is_hip` 判定改成 `is_cuda`）**禁止**：它把所有
`elif _is_cuda:` 的不可移植分支（flashinfer/trtllm 独有符号）一并打进
导入路径，且砍断依赖 `is_hip()=True` 的正常分支（fnuz 表、gfx workaround）。
逐条放宽 + 已验证实现背书，是一个个来的。

### 3b. 模块命名与档位标识规范（2026-10-28 定）

> 目标：**不看内容、只看文件名**，就能回答"这个算子有哪些档、缺哪档"。
> L0 `impl-naming` 执法；档位全景的司法事实源是 BACKEND_CATALOG，
> 渲染表 `docs/tier-matrix.md`（`tools/gen_tier_matrix.py` 生成，L0
> freshness 钉同步）。

1. **具体实现文件 = `<op>[_<变体>]_<tier>`**，tier 词表封闭：
   `emu`（torch 直算/仿真臂）· `triton`（自研 triton）· `cpp`（C++ JIT）·
   `fork`（vendor 源码 fork 承接）· `ref`（诊断参照，位置约束见 §2）；
   vendor 库绑定安装器用**库名**：`aiter` / `lightop` / `dtk` / `rccl`（通信库）。
   例：`conv1d_update_emu` · `mqa_logits_triton` · `paged_attn_cpp` ·
   `quant_aiter` · `deep_gemm_dtk`。**平台词（hip/cuda/rocm）不算档位**，
   禁作实现后缀。
2. **路由器 / 机制 / 门禁文件禁用档位后缀**（裸 op 名或机制名：
   `watchdog_cold`、`kv_fix`、`spec_h2d_pin`）；路由目标需体现时用泛词
   `_vendor`（`fa_vendor`、`mqa_vendor`）。前缀为主题而非档位的机制
   补丁不受限（`triton_libdev` / `triton_pdl` = 补 triton 编译栈本身，
   非 triton 档实现）。
3. **同族实现共享 op 词根**：`paged_mqa_logits` 家族 =
   `dsa_mqa_logits_emu` / `mqa_logits_emu` / `mqa_logits_triton` /
   `mqa_vendor`——一眼可辨全家族档位覆盖与缺口。
4. **`_ref` 红线**：ref 文件只准被 `tests/` 引用（§2）；生产路径出现
   `_ref` import 即违规。
5. **sgl_kernel 影子契约文件豁免**：文件名 = 上游符号面
   （elementwise / moe / gemm_fp8 …）；包内自研 helper 同样适用本规范
   （`paged_attn_triton` / `paged_attn_cpp` / `fa_varlen_triton`）。

6. **DeepInfer 内核门面与机制豁免清单**（与 L0 同步）：
   - `gemm.py`、`gather.py`、`norm.py`、`index_ops.py`：校验或选择后端的门面。
   - `fused_transform.py`：按原有顺序组织 norm/RoPE/量化、输出写回和缓存写入的复合执行门面；
     单档位 RoPE 与 KV 计算分别位于 `rope_{emu,lightop}`、`kv_cache_emu`。
   - `runtime_config.py`：宿主侧配置、对齐规则及运行参数状态，无张量数值实现。
   - `lightop_usage.py`：共享调用计数机制，无张量数值实现。
   L0 对 `hcu_adapter/kernels/` 按此清单或档位后缀检查；新增豁免需同步说明职责。
7. **实现文件档位与配置值分开管理**：`_emu` 是 Torch/仿真实现的文件后缀；
   已发布的后端选择值 `torch` 仍是合法配置值，不是文件命名档位。
   结构整理不修改 catalog 算子键、配置默认或已接受的配置值；此类接口迁移须单独评审。

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
  `_base.py`/注册表 → 加 e2e 冒烟（ci.md 门禁节既有规则）。

### 5.1 plugin 的演进轨迹（接入 → 上游化 → 退役）

plugin 承载的"上游没有的新能力"（自研量化格式、新算子、新后端）不是终点
形态，而是一段演进路径的起点。标准轨迹四阶段，每阶段的准入/退出都有实证
要求：

1. **plugin 接入**：独立代码仓，经上游 seam（若该类能力在上有挂载点，
   如 `--quantization` 注册、attention/moe-runner backend registry）或
   自带注册入口接入；直调 lightop/aiter 等底层库不强制穿 adapter 路由。
   准出条件：端到端跑通 + 数值门禁绿（ref 实现落 `tests/kernels/`）。
2. **证据积累**：性能对照（vs 现有 active 路径，如 channel-wise vs
   block-wise、W4A8 vs W4A16 的格点数据）、生产窗口稳定运行记录。数据
   是后续一切的弹药——**格式契约（scale 布局、packed 排布、量化 axis
   约定）从第一天起固化成文档 + 契约单测**，这是将来上游 PR 的核心内容。
3. **上游 push（择机）**：以正式 PR 向上游贡献格式解析 + runner 注册点
   （核实现可留在 plugin 仓，经 seam 引用）。参照范本：上游 AWQ-MoE 的
   `quantization_config.quant_method` → `--quantization` 解析 → runner
   注册三级结构。合并即意味着该能力的"上游在 AMD/CUDA 上有"成立。
4. **桥/影子实现退役**：上游合并后，adapter/plugin 里对应的过渡实现
   （影子库寄居代码、monkeypatch 级桥）逐 MR 降级删除，active 指针归位
   上游实现——§5 退役条款在此生效（`production → deprecated → removed`，
   不留注释掉的桥）。HCU 侧受益于上游生态对格式的原生认知，非 HCU
   部署同样可用该格式。

反模式：**跳过阶段 3 长期停在阶段 1**——plugin 永久靠注册入口自挂载、
格式永不固化为契约，结果是该能力永远只有单一平台认识它，上游升级时
独自承担全部兼容成本。若某能力确定无上游化价值（纯平台私有优化），须
在 plugin 仓 README 明示"不上游化"决定及其理由，避免默认处于悬置态。

### 5.2 算子档晋级门（2026-10-26 定）：接入即晋级，对比测试案例是发现器

**vendor 算子档（aiter/deepgemm/lightop/boltops 等轮子家的核）的晋级
不以性能对比为门——能接入即晋级**（接入判据 = 真实调用面跑通 + 数值
对拍过参照）。性能/精度问题不是晋级阻塞，而是**对算子组提需求、提 bug
的输入**（通道 [operator-requests.md](operator-requests.md)，提报标准
§0 已有）。此前散见的「production 晋级 = serving A/B」表述只适用于
**自研实现档**（emu→triton 家族——核是我们自己的，性能是我们的账）；
vendor 档的性能账在轮子同事，adapter 不背。

**但 adapter 必须配对比测试案例，让这类问题能被"跑测试"发现**，而不是
靠肉眼猜或用户投诉：

- **数值对拍**（vs 参照路径/金标准，容差明示）——失配即档位回撤/哨兵
  转红，属阻塞；
- **计时对拍**（vs 现 active 路径，同形状逐 M 计时）——比值只记录在案
  不阻塞，作为 operator-requests 的「现状账」证据（慢多少倍、挡什么）；
- 惯例参照 R-02 哨兵双事实模式：positive-matches-ref + 缺陷-persists
  两条钉子——**轮子修好后哨兵意外转绿即自动解除回撤**，重跑门禁晋级。

对照现状：mqa vendor 档（`tests/kernels/test_mqa_vendor_gfx938.py`）、
quant aiter 档（`tests/kernels/test_quant_aiter_gfx938.py`）已是此形态；
moe_c w4a16 档目前只有 CPU 契约单测（wiring/ABI），GPU 对比哨兵
（parity vs stock triton 臂 + 逐 M 计时，box16 探针一次性数据在案未固化）
为欠账，接入 serving 后补。

## 6. L0 静态规约速查（ci/l0_gate.py）

| 检查名 | 级别 | 规则 |
|---|---|---|
| `glue-numerics` | HARD | sitecustomize/bootstrap/宿主 bridges 无 triton import/jit/`tl.`（注释里的提及不触发） |
| `text-patch` | HARD | 每个 `_is_cuda` 系 `.replace` needle 都在 `TEXT_PATCH_ALLOWLIST` |
| `registry-ops` | HARD | 每个 `registry.register(...)` 的 op 都是 catalog 条目 |

## 7. 双宿主共建、plugin 接入时序与维护闭环（2026-10-28 定）

> 承接原任务看板中的治理面（看板已移除，任务跟踪归 MR 与团队现用系统）。
> 本节沉淀 sglang / vLLM 双宿主共建与长期维护的裁决规则，与 §0/§1 的
> 分仓边界同源。任务看板原有的北极星与硬件备忘迁入 §7.3/§5.1。
>
> 与 plugin 仓的文档呼应：本仓（adapter-das）持有**机制层真相源**与
> **归仓判据**（§0 判别问句）；`hcu-sglang-plugin` 仓的
> `docs/governance.md` 持有组合兼容契约（目标所有权、组合验证矩阵、
> `original_fn` 包装链语义），其 §1 判据表是 §0 的应用视角——三处判据
> （本节 §0 / 本仓 plugins.md §0 / plugin 仓 governance §1）任一改动
> 同笔同步。同名文档的分工对照见本仓 `docs/reading-guide.md`
> 「姊妹仓地图」。

### 7.1 进程接入时序（红线）

两个宿主的扩展点按同一时序进入进程，桥/hook 的可实现性由它决定：

**sitecustomize（adapter，解释器启动）→ 宿主框架 import → 宿主 plugin
机制 → 引擎启动**。

- sglang 侧：`load_plugins()`（entry_points 组 `sglang.srt.plugins`）在
  `launch_server` / `Engine.__init__` / scheduler **每个进程**启动早期执行，
  先于 `ServerArgs` 构造，加载后统一 `HookRegistry.apply_hooks()`；
- vLLM 侧：`vllm.platform_plugins` / `vllm.general_plugins`（entry_points）。

由此三条推论：

1. **必须在宿主框架 import 之前生效的东西**（平台 seam、sys.modules 别名、
   import 时常量冻结）**永远只能住 adapter**——plugin 机制上够不着（它
   随宿主 import 而来）；
2. **adapter 补丁只准在启动期安装**：HookRegistry wrap 的是 apply 时刻的
   绑定；apply 之后若再 rebind 同一目标会**静默吞掉 plugin hook**（无
   报错）。新桥禁止"用到才装"的惰性 rebind；
3. plugin hook 对 adapter 已换绑的路径**用 AROUND 不用 REPLACE**（REPLACE
   是谁后写谁生效，两仓各自 review 都看不出冲突）；热路径差异化用配置期
   选择（backend / scheme 替换），不用逐调用 wrapper。

### 7.2 双宿主共建边界

- **共享层 = `hcu_adapter/` 包本体（bridges/common + 机制层/域桥 + `sgl_kernel/`）**，
  framework-free 是硬契约（§1 表）；共享层变更两个宿主都要回归；共享算子
  配**宿主中立 parity 契约测试**（`tests/kernels/`，不依赖 sglang/vllm
  测试套的存在）；
- **宿主桥保持薄**（平台身份 + 版本门 + install 编排），实现体按 §0 归
  `hcu_adapter/<域>.py`；`bridges/sglang.py` 长期向薄走是共建验收线
  （P1-P3 拆分见 architecture.md §9）；
- **纪律互相吸收 = 下沉到共享层**：vLLM 桥的版本门（不支持在选择 HCU 前
  raise）、fail-closed 硬件探测（PCI vendor，不初始化 runtime）、失败锁存
  三项抽入 common 供 sglang 腿复用；sglang 侧 jit_build_log /
  aiter_jit_cache / 逐补丁命中计数等可观测性反向供 vLLM 用；
- **分层 import 契约**：`bridges/sglang.py` 禁 import vllm，
  `bridges/vllm.py` 禁 import sglang（common 与包内机制层禁宿主已由 §1 覆盖）；
- **vLLM 扩展宿主权责边界（extension API v1）**：Adapter **独占**
  `vllm.platform_plugins` / `vllm.general_plugins` 两个官方入口；独立 plugin
  经 `hcu_adapter.extensions` entry-point 组接入，只交付可选优化实现
  （`ExtensionSpec` + 实现），不得复制版本/平台/DTK 门禁——宿主适配只有
  一个启动裁决点。激活发生在版本、平台和 DTK runtime guard **之后**；
  optional 扩展失败降级为进程内报告，仅显式 required 扩展/capability
  失败才响亮阻断；无 plugin wheel 时保持官方 vLLM 基线路径可启动；
- **宿主仲裁待改**：argv 无信号且双框架并存时 bootstrap 现静默取 sglang
  （legacy-default）——应改为 loud fail 要求显式 `HCU_ADAPTER_HOST`，
  在此之前两框架混装的节点必须显式设该变量。

### 7.3 维护闭环（官方单测 = 行为契约；统计 = 清理输入）

- **契约子集**：宿主官方单测的精选面（ci.md L1.5 段）是 adapter 的
  验收标准；**skip 数环比与失败数同级**——skip 异常上涨 = 适配在变相降级
  （"转 skip"类修复必须逐项登记、可审计）；
- **单测只证正确、不证不慢**：性能面由微基准 + serving 门守（R-15 三级
  防线节奏，台账归 ci.md / models.md）；
- **死分支清理以联合覆盖统计为输入、人工分类后执行**：统计负载 = 契约
  子测 ∪ serving 门（旗舰模型 + bs 扫描 + graph capture + spec decode）；
  每补丁留命中计数（顺带充当补丁存活检查：没装上 = 零命中）。
  **反例固化（红线）**：`moe_dualstream_gate` 的 capture-only 分支单跑
  单测必然零命中——禁止以"单跑覆盖率没用到"为删除依据；删除一律可逆
  （commit + 证据留档）；
- **北极星两只表**（评审对照）：① adapter 补丁数随上游版本**单调下降**
  （清理 + upstream PR 的净效果，见 §5.1 演进轨迹）；② skip 率稳定。
  两只表持续恶化 = 在长回第二个侵入式分叉。

### 7.4 版本与门（宿主 / 影子包）

- 影子包 `sgl-kernel` 的版本号必须携带上游基线（目标形态 `0.5.22+hcu.N`）；
  现值 `0.1.0` 待改——不带基线信息时 pip 层与官方 wheel 可互替、静默换内核；
- vLLM 支持门现钉 v0.32.0 release line（`bridges/vllm.py`）；抽 common 时
  sglang 线同享"不支持在选择前 fail"；
- **已知悬案**：vLLM platform plugin 在本栈 import order 下挂死（nmz26
  实证；sglang 进程内以 `VLLM_PLUGINS=none` 静音规避）——standalone
  vLLM serve bring-up 销案前不得移除该规避；
- channel-wise 量化（本硬件 core/tensor core 不重叠 → channel-wise 是
  主力量化路线，见 §5.1 阶段 2）归 **plugin 仓**：每宿主一个 plugin、
  kernel 共享 `sgl_kernel`、格式规范与 conformance 单测**单源**——判别
  仍走 §0 问句，与"动机是性能"无关。


## 算子三居地图与宿主绑定层

| 目录 | 当前职责 | 新代码归属规则 |
|---|---|---|
| `sgl_kernel/` | SGLang 宿主 ABI 契约层；影子包名称由上游锁定 | 契约文件按 §3b 豁免；不为本次 DeepInfer 整理改动此目录 |
| `hcu_adapter/` 顶层 | 历史 SGLang 侧桥接实现，以及多宿主共用的配置、registry、引导等机制 | 现有实现按任务逐步整理，新计算实现不再增加顶层平铺文件 |
| `hcu_adapter/kernels/` | 当前由 DeepInfer 桥消费的上游契约实现；不依赖宿主模型类 | 新计算实现进入此目录，使用算子族＋档位命名；门面须有真实校验、选位或组合职责 |
| `hcu_adapter/bridges/_deepinfer/` | DeepInfer 宿主绑定、安装和契约分发 | `tilelang_ops.py`、`tile_kernels.py` 对接上游符号面；不放纯计算实现 |

`kernels/` 独立命名空间避免导入 `sgl_kernel.__init__` 时提前启动 SGLang 的
AOT/stub 注册。当前消费方是 DeepInfer；未来其他宿主复用时，仍需各自 bridge
明确张量布局与签名，不能按同名算子自动互换。模型新量化方案与执行策略仍归 plugin。

新算子计算代码只进入 `kernels/`，`bridges/` 只保留绑定、安装和契约分发。
L0 检查档位命名与绑定层数值代码；代码评审还需检查职责，L0 GREEN 不代表目录归属免审。
不符合分层的变更应在 L0/评审阶段打回。跨层别名壳与无职责的旧路径转发不再保留；
迁移模块时须同步 bridge、catalog、plugin 导入及 provider 身份检查。

本轮逐文件核验：`tile_kernels.py` 创建上游模块并登记实际 provider，
`tilelang_backend/factories.py` 导出契约 factory 并为未实现项生成报错闭包，
`tilelang_ops.py` 校验 factory 参数、选择 provider 并生成调用闭包；三者留在桥层。
它们当前没有需要移走的独立数值实现，不能把报错闭包或绑定表误当成计算算子。
详细更名和归属表见[DeepInfer 算子导航](frameworks/deepinfer-kernels.md)。
