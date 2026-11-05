---
name: hcu-adapter-dev
description: hcu-sglang-adapter 仓库开发规约：算子垫片/影子库新增、三档实现策略、自动化测试要求、高危算子快关与精度定位流程、sglang cookbook (NVIDIA 参数) 直接支持的降级矩阵。改这个仓库的任何代码前先读本 skill。
---

# hcu-sglang-adapter 开发规约

不修改 sglang 官方代码，让社区版 sglang 直接跑在海光 HCU（HCU/DTK/HIP）上。
本仓库的价值主张：**上游 import 什么算子名，我们就提供什么算子名，内部可
路由到 triton / vendor / torch 三档实现，并对全部出口 API 做统一质量验证**。
"adapter 测试全绿 ⇒ 官方 sglang 无大问题" 是本仓库的验收标准。

## 黄金法则（历史 bug 换来的，违反必踩坑）

当前 SGLang 迭代（2026-11-03 用户要求）：社区源码文件不改写；
以功能和精度为验收标准，性能不设门槛；每日追新推迟到其余优化完成后。
历史磁盘 text/JIT 补丁逐项迁移，不能把运行时写回源码称为“零修改”。
迁移函数绑定需测试已缓存别名和 spawn 子进程；仅换 module 属性可能遗漏
早先的 `from ... import ...`。`dist_bc` 已以保留函数对象和私有 numpy
facade 的运行时桥替代磁盘补丁；不可重新引入 common.py 写回。
`moe_dualstream_gate` 同样已迁移为保留函数对象的运行时分支适配；
仅作用于 MoE forward 的私有 capture predicate，不能改全局 capture 状态
或临时改实例的 alt_stream；保留 skip_shared_experts 与 DeepEP 等分支。
`spec_h2d_pin` 通过原函数私有 Torch facade 分配 pinned KV-lens，
保留函数对象与已有别名，不写回 eagle_utils.py。不能以固定 ring 深度
推测 DMA 已结束；使用 Torch 管理的 pinned 分配并验证多流异步复制。
`fp4_expert_load` 保留社区权重加载函数与已有别名，以私有 cast 接口
对齐 FP4→FP8 的 128×128 block 配置，并支持显式 DeepEP BF16 反量化；
不写回 fp8.py。block 元数据必须同时用于当前加载和后续 runner，
不能只改 method.weight_block_size 而让 quant_config 仍为 32×32，
也不能修改其他层共享的 checkpoint config。
`dsv4_wqkv_scale_patch` 仅扩展原加载函数两处 `name.endswith`
的参数常量，以 tuple 接纳 CT-int8 `.weight_scale`，保留原函数对象、
指令字节及 PP/MTP、异步加载、流式 buffer 与原融合路径，不写回
`deepseek_v4.py`；签名或调用位置漂移须拒绝，不能重写权重流绕开原过滤。
`jit_sources` 将 JIT `csrc/include` 的编译资产及 CUDA→HIP forwarders
生成到 adapter 的内容寻址缓存，不能写社区树或 DTK include，也不能复制
社区 Python 包或伪造模块 `__file__`。全部变换完成并校验后原子发布；
key 包含输入字节和变换实现，已有缓存损坏须拒绝。保留原
`resolve_sources` / `load_jit` 函数对象和别名，覆盖绝对/相对源路径、
默认/显式/依赖 provider 的 include。保持官方 BuildSpec/ninja/cache
与依赖追踪；量化先生成 packed shuffle 分支，再由 warp 修正 generic T。
`jit-input:` 注册仍按原 needle 白名单治理，但表示缓存中的编译数据。
`jit_fp8_ffi` 只在编译导出边界对 e4m3fn 输出做零拷贝 uint8 view；
不能把输入或 scale 一并转换，也不能改变上层 dtype 选型。保留原
cache_once factory 与已有别名，warm cache 和 cold key 均须覆盖，
签名/闭包漂移拒绝；快关通过环境注册表解析。FFI 通过不代表量化数值通过。
DSV4 的 gfx938 编译输入使用 DTK OCP 转换，避免社区软件转换在
exp8=15 过早饱和；不扩大到未验证架构或改变 gfx942/FNUZ 分支。
此类编译成功后的错误必须以独立数值和变输入 graph 回归识别。
`tools.l15_checkpoint_reference` 在隐藏 GPU、HOSTnone 的独立进程用
CPU FP32 Transformers eager 对拍真实 checkpoint smoke 的所选 token 概率。
它目前是短输入探针，不输出精度验收 passed；新 smoke 在 controller 与
worker 加载/执行前后绑定实际 checkpoint 字节，参考执行也必须匹配此清单。
历史 smoke 的缺失绑定不能靠参考时 fingerprint 倒推补齐，须明确标记
legacy-unbound。精度门禁还须补输入/长度/分布和模型族覆盖，不能把 top1 或有限 logprob
单独当成完整模型精度。
checkpoint 哈希结束还须重新读取资产字节确认一致；同 tick 的等长写入可能
保持相同 stat 元数据，不能仅依赖时间戳判断已读文件是否变化。
固定 `--profile accuracy` 另覆盖双语、512/2048 token、复用/变输入，
以 CPU FP32 的所选概率与 top-20+tail 合并分布按版本化策略验收；
必须匹配 HCU 权重字节并执行完整 eager/graph，不能把合并分布称为
全词表散度，也不能外推为其他模型/长度通过。聊天模板明确请求 list，
并在 Engine 前校验 token IDs，避免 Transformers 返回映射时误传字符串 key。
复合 checkpoint 的文本词表从 text_config 获取，不能读外层/视觉词表；
独立参考通过公开条件生成 AutoModel 入口加载并记录实际类。Qwen3.5 的
纯文本验证不外推为图像/NEXTN/量化或多卡通过，家族 cookbook 配方与
较小 checkpoint 的参数差异必须单独记录。
完整目标与证据见 `docs/l15-readiness-plan.md`。
输出层归因可显式使用 smoke 的 `--enable-fp32-lm-head`，它只转发社区
公开选项；controller、worker、实际 server info 和 CPU 参考报告必须一致。
默认保持社区默认配置，不能因局部指标改善而启用为 adapter 默认或改精度阈值。
`tools.l15_checkpoint_dtype_probe` 在隐藏 GPU 的 HOSTnone 进程，对已绑定
accuracy 请求固定同一 prompt/输出序列，比较独立 CPU FP32/BF16 eager
及 HCU 概率；记录具体输出位置、token 和三方 logprob。CPU BF16 包含
BF16 输出层，不等价于 SGLang/HCU 实现；报告仅作诊断，不输出 passed，
不能用 CPU 自身的 dtype 差异豁免原 FP32 精度门禁。
smoke 的 `--dtype float32` 仅是显式整模型精度诊断，默认仍为 BF16；
controller/worker/server 与参考报告须绑定同一执行 dtype。历史缺失字段
只能按原 BF16 默认解释。FP32 配置在原 V1 阈值内不代表 BF16 通过，
也不等同于建立了更严格的 FP32 数值策略，不能用改默认 dtype 替代定位。
真实量化启动须核对实际执行路径：config 声明或 server 的未显式设置
`quantization=None` 不足以判断路由。可用社区公开 profiler 接口获得
scheme/量化/scaled-mm/设备内核调用证据，报告绑定原始 trace 字节摘要。
CT-FP8 的组件回归必须经过真实 scheme、加载后处理与默认 padding；
仅 `apply_fp8_linear` 的非 CT、显式关闭 padding 分支不能代替。
CPU BF16 参考也受 CPU 架构/内核实现影响；跨节点比较需保留环境，
不能把不同 CPU 的 BF16 数值差异直接归因 HCU。
smoke 的 `--quantization` 只转发社区公开选项，默认仍按 checkpoint 自动检测；
controller/worker/server 及参考报告必须绑定同一显式选项。配置未声明量化
但显式在线量化的执行也须拒绝普通 CPU FP32/dtype 参考，不能据此绕过量化限制。
INT8 MoE 合成对拍须保留原地 runner 的输入副本；量化先用 FP64 检查整数与
尺度契约，FP32 reciprocal 的误差界内半整数可取相邻整数，其余须精确取整。
再用已验证整数/尺度做独立 GEMM 参考，不把浮点除法在半整数的差异误判为
graph 错误；最终输出阈值保持原样，routing 在 down-GEMM 存 BF16 前相乘。
显式 `w8a8_int8` 和自动 CT 的 Linear 都需绑定既有 `int8_scaled_mm`；
只修 CT 模块不能覆盖前者。`int8_linear` 在验证两个原方法的全局符号契约后
补 module global，保留方法和已有别名，不新增 GEMM 或改社区源码。
退役自研 all-reduce 后，vendor 拒绝门不得再引用其旧安装状态；
`HCU_DISABLE_OPS=aiter_ar` 必须在新解释器真实 bootstrap 中拒绝并回退 NCCL。
CPU 参考工具的单测也必须在干净 HOSTnone/隐藏 GPU 的进程执行；
GPU 全套已加载宿主时，以独立进程复测，不能放宽生产工具的隔离守卫。
量化 CPU 参考默认拒绝；显式 `--quantized-reference ct-fp8-token-channel`
仅支持已验证的 Dense Qwen3 CT channel-FP8 权重与动态 token-FP8 输入。
必须匹配 HCU 绑定字节，逐个检查全部加载 tensor 与完整 Linear 量化参数，
并观察每层实际激活 QDQ；只反量化权重不足以建立参考。保留原精度策略，
greedy 一致不能豁免分布 FAIL，不能绕过社区 CT 对整模型 FP32 的拒绝。

1. **同名 ≠ 同语义**：每个垫片必须对拍参考实现（cos/rel 断言），不能只测
   "能跑不崩"。案例：rope GPT-J cache 恒为半分 [cos|sin] 布局（fbc540c）。
2. **eager 对 ≠ graph 对**：凡进 decode 链路的算子必须过 capture-replay
   变输入探针。案例：AiterCustomAllreduce 图内重放损坏（1eb038d）。
3. **GPU 路径禁主机同步**：`.item()`/`nonzero`/布尔掩码索引会毁掉 CUDA
   graph capture。
4. **高危 vendor 算子默认不信**：custom all-reduce、flash attention 这类
   "size 灵活、场景多" 的 vendor kernel，接入即配压测 + 快关。
5. **import 永不失败**：未覆盖算子在**调用时**抛带指引的 NotImplementedError。
6. **环境变量三原则**（客户反馈"变量太多"的对策）：一个手册
   （`docs/env-vars.md`）；默认自动适应（arch/wheel 探测，正确部署零
   变量）；显式设置优先且启动日志标注来源（`hcu_env.py` 注册表，
   禁止在行为接线处裸读 `os.environ`）。
7. **平台语义走 is_cuda 分支**（governance §0.5）：上游平台判定统一回答
   CUDA——新模型/新功能上游先落 CUDA 分支，追新是卡商关键路径；HCU 缺的
   NVIDIA 库符号按四级路由接（lightop → aiter → triton 先行 → 响亮 stub）；
   fp8 fnuz 等语义护栏白名单锁定不随翻。
8. **新容器先灭镜像包**：每次创建/新用容器，先把镜像自带的 sglang(das
   定制线)和 sgl_kernel/sglang_kernel 从 dist-packages 删干净再干活；
   残留会与 PYTHONPATH 的官方源/ adapter 混栈（症状：奇怪的
   ImportError 如 hcu_alloc_decode_kernel、module 解析到镜像路径）。
   清理命令见 references/cookbook-matrix.md。

9. **测试先行顺序**：算子/数学类新实现——单测先于 e2e（ref 对拍+形状矩阵
   +容差，脚手架见 references/test-suite.md 三段式）；seam/接缝类——e2e 探路
   可以，但 "one pit, one test"：探明的坑修复时必带离线复现单测，否则不予
   合入。实证对照：contig-triton/fa3-emu 单测先行 e2e 首过 vs
   moe_align/slimquant 反序付 e2e 倒查代价（governance §7 维护闭环）。

10. **tests/ 下禁止脚本级副作用**：门禁会 pytest 收集 `tests/kernels`
   与 `tests/adapter`——模块级 `sys.exit`/裸执行会让收集期直接
   INTERNALERROR（两周内两撞，ae0698b 后仍有回潮）。脚本式测试必须
   函数化 + `__main__` 守卫；新写测试一律 pytest 兼容形态。

11. **性能 bench 三分离**：跑 bench 前 GPU 空闲检查、**预热段强制**
    （冷编译曾把 triton bench 测成 0.0×，热跑实为 1.9×）、结果带环境戳
    （卡型/驱动）。断言用**相对阈值**（臂间比/历史比），不吃绝对
    tok/s——绝对值锚在别人的机器上。性能判别与晋级依据 catalog 档位
    证据门，不做无档案的"感觉变快了"。

12. **模块命名随档位走**（governance §3b，L0 `impl-naming` 执法）：具体
    实现文件 = `<op>[_<变体>]_<tier>`——tier ∈ emu/triton/cpp/fork/ref
    或 vendor 库名（aiter/lightop/dtk）；路由器/机制/门禁裸名；平台词
    （hip/cuda/rocm）不是档位；`_ref` 只准被 tests/ 引用；同族实现共享
    op 词根（mqa_logits_{emu,triton}/mqa_vendor 一眼看全家族）。加档位
    后跑 `python tools/gen_tier_matrix.py`，`docs/tier-matrix.md` 一眼
    看全"谁有哪些档、缺哪档"（L0 freshness 钉同步）。

13. **vLLM 专项后端保留官方语义**：普通 MHA/GQA 可在已验证条件下默认
    `TRITON_ATTN`，但 `use_mla` / `use_sparse` 等专项配置必须保留 `None`
    交给官方 `RocmPlatform` priority，禁止强改成普通 attention backend；
    Plugin 的普通 FlashAttention 优化也不得接管 MLA/Sparse MLA。禁止把
    `gfx936` 等 HCU 架构伪装成 CDNA3 来让单个 AITER 检查通过——平台代际会
    同时放开 FP8/GEMM/MHA/MLA 等不相关能力。只在最窄 seam 表达真实能力，
    测试至少覆盖普通、MLA、Sparse MLA 路由以及安装态 wheel 的实际 backend 日志。

## 任务路由（做某类事之前先读对应参考文档）

vLLM 0.33/gfx936 的 C4 indexer 候选见
`docs/frameworks/vllm033-sparse-indexer.md`。复用官方 logits/top-k 和 public
metadata/OOT，不能用伪架构或全局 helper 重绑准入。Graph padding 取 host
metadata 边界并同步打包 query/weights；便利 pack/unpack wrapper 的 GPU
长度读回不可带进 capture。默认 stock；native 显式 opt-in，快关优先。
源码、安装态、spawn/fork、两层缓存和完整模型结果分别记录，不以组件晋级。
C4 缓存须遵守官方物理页 pitch：8448-byte payload 可位于8640-byte对齐页，
BLHNC/BLNHC 还可在物理块间交错其他层。只要求页内 block-flat payload
稠密、页间不重叠且 FP32 scale 对齐；不得以整张 tensor 的 is_contiguous
拒绝官方视图，也不得复制 cache 来绕过布局。设备探针覆盖官方
create_kv_cache_views 的 aligned/packed 视图、填充/其他层不变和变输入
Graph；metadata 的动态长度探针仅支持其原始 MTP1 配方。

vLLM 0.33 mHC 先读 `docs/frameworks/vllm033-mhc.md`：恢复官方既有行为归
Adapter，优先 public OOT 委托官方 Torch pre/post 和公开 IR RMS；原生签名
接受 norm 参数不证明执行了 RMS。保留已通过的官方 post/head，vendor
小批次通过不覆盖大 prefill 舍入差异。capture 后须先 replay 建立基线，
空 Graph 与非空分开。候选默认 stock；注册、两层缓存、late drift、
spawn/fork 和四质量维度须留证，CPU lifecycle 不能作为模型/设备通过。
注册后直接重试遇到非法策略也须锁存失败；恢复配置不得重新准入。
多个内置 owner 组合时，宿主健康边界须逐个检查；同名函数导入必须独立绑定，
避免后一次覆盖前一次。回归同时覆盖各 owner 失败和全健康路径。

共享机制在**调用时**也必须 framework-free，不能用惰性 `from vllm import
envs` 隐藏宿主决策。模型 lifecycle 的声明/上下文/同步留在 `model_load`；
vLLM worker/executor/env-cache 覆盖查询归 `bridges.vllm_model_load`，薄 Worker
通过该公开门面调用。迁移公开路径时同步真实 Plugin 的 ABI 预检和 selector
导入，禁止共享层反向导入宿主桥作兼容壳。验证必须含双宿主 import-blocking
下的真实共享 dispatch，而不只检查 import 成功；详见
`docs/frameworks/vllm032-post-model-load.md`。

vLLM public registry/extension health 修改先读 `docs/vllm-flash-registry.md`。
共享 lifecycle 机制放 `runtime_lifecycle.py`，不能长在宿主薄桥内；改动须
回归双宿主 bootstrap/import 与共享 consumer 契约，并保持真实 vLLM probes。
回归必须覆盖真实官方 selector 在 clear/foreign 后的拒绝、零实参签名预检、
精确 report/PID 绑定、sink 失败和错误格式化中实际 fork；不得只直接测 owner
checker。CPU 生命周期通过不代表 HCU 数值或模型通过。

宿主启动失败须先锁存固定拒绝，再格式化异常；`__str__` 也可能抛
`BaseException`、重入或 fork。扩展宿主先判原始 fatal 链，再安全生成唯一
FAILED 报告；诊断失败不能把已发布错误变成普通 optional，也不能把干净
optional 误判为 fatal。保留 callback 原有捕获边界，不新增恢复接口。
回归需有真实 metadata/官方 loader、安装态坏字符串、父子拒绝和重试证据；
详见 `docs/vllm-commit-fatal.md` 与测试参考。

> 全仓文档地图与失效信号清单见 `docs/reading-guide.md`（新会话先读它）。

| 你要做的事 | 先读 |
|---|---|
| 新增/修改一个算子垫片或影子库 | `references/shadow-matrix.md` |
| 加一座桥（改官方代码行为：换绑/断言放宽/判定词 wrap）或新能力选位 | `docs/governance.md`（双仓判别 + 裁决链 + registry 登记） |
| 写或改任何 kernel / 测试 | `references/test-suite.md` |
| 加一个快关（kill switch） | `references/kill-switches.md` |
| 排查精度问题（输出垃圾/分叉/漂移） | `references/precision-triage.md` |
| 端到端性能优化（开箱基线/profiler 归因/算子优化需求/回归） | `references/e2e-performance.md`（细节检索个人知识库 kb，手册内有 kb:<id> 引用） |
| 评估 sglang cookbook NVIDIA 参数能否直接用 | `references/cookbook-matrix.md` |
| 加/改一个环境变量 | `docs/env-vars.md`（手册+维护规则）与 `hcu_env.py` 注册表 |
| vLLM required 启动策略 / 日志来源 | `docs/frameworks/vllm-required-extensions.md`（正常 bootstrap→common 才自动汇总；空 required 默认不单列、不得为补日志另造策略；HOSTnone 直接 general-plugin 仍 enforce，但不承诺 common 日志；须测真实两条调用链及官方 general-plugin loader/真实 metadata，不能用手动 log_summary 代替） |
| vLLM 官方 AR 的 HCU 兼容 | `docs/frameworks/vllm032-allreduce.md` + `references/kill-switches.md`（先真实 runtime ABI，再 FP8/AR/Plugin；默认不选 AR，不能继承 SGLang GraphSafe 证据或放开全局 AITER/伪装架构；注册 fused ops 不代表 fusion pass 已启用） |
| 改上游行为补丁（text patch） | `docs/governance.md` TEXT_PATCH_ALLOWLIST（L0 硬门） |
| 加/改算子档位实现文件（命名/归档） | `docs/governance.md` §3b + `python tools/gen_tier_matrix.py`（档位矩阵随 catalog 同 commit） |
| vLLM 0.32 平台、attention/MLA backend 选型 | `docs/frameworks/vllm032.md` + `docs/governance.md`（先区分普通/MLA/Sparse MLA，保持真实架构能力） |
| vLLM 0.33 升级、公开后端与版本准入 | `docs/frameworks/vllm033.md`（版本门通过不代表 ABI/设备通过；官方没有旧 fork 的 Flash 私有 hooks，须核对公开 provider 接口；INT8 专家准入不得扩大到其他格式） |
| vLLM HCU 架构清单 | `docs/frameworks/vllm033.md`（`gfx928/gfx92a/gfx936/gfx938`；身份匹配与 Core DTK 编译清单一致，provider 能力仍按各架构证据单独准入） |
| vLLM 扩展发布状态后失败 / 捕获错误后继续 / fork 健康 | `docs/vllm-commit-fatal.md` + `docs/governance.md` §7.2（干净 optional 失败仍可继续；显式 commit-fatal 不得恢复或伪装回滚） |
| 需求/任务领取 | 任务跟踪不走仓内看板（board.md 已移除），归 MR 与团队现用任务系统；协作红线见 `docs/governance.md` §7 |

## 治理闭环（改代码必走的登记链，L0 机器查）

一处行为落地，多处登记同步——**同 commit 内完成**，否则 L0 红：

1. **catalog**：BACKEND_CATALOG 加/改条目（状态机 planned/candidate/
   validated/production/broken，晋级必须有对拍+bench+A/B 证据）；
2. **env 手册**：新 env → `hcu_env.py::REGISTRY` + `docs/env-vars.md`
   行（手册在册不是可选动作）；
3. **UI 快照**：`tools/hcu_config_ui.html` 与 catalog 同 commit（快照是
   治理链最后一公里，曾吃过 a983af3 亏）；
4. **运行时注册**：行为接线用 `registry.register`（验收/排障以
   `HCU_ADAPTER_DUMP` 可见为准）。

平台语义走 is_cuda 路线的推理：judgement 层翻译、硬件层保持真（见
governance §0.5）；上游行为补丁（源码 text-patch）必须进
TEXT_PATCH_ALLOWLIST 白名单，白名单外不合法。

## 仓库结构

```
hcu_adapter/           唯一适配包：bootstrap.py 宿主探测 + common→host 分发
                       （sglang/vllm/deepinfer/none）；bridges/ = common 跨框架守卫 +
                       sglang 宿主桥顺序链（按 architecture.md §9 P1-P3 向包内
                       域桥拆分中）+ vllm 平台身份/版本门/ABI 桥，安装序列有
                       专属回归钉 tests/adapter/test_sitecustomize_sequence.py；
                       deepinfer 宿主桥与契约内核独立命名空间；包内其余为按契约
                       复用的跨宿主机制层与域桥：
                       aiter / flashinfer / flash_attn 填空+守卫 + vendor 桥
                       （quant/ar-norm/mqa/w4a16）+ fla_hcu 路由；
                       hcu_env.py REGISTRY + hcu_config.py BACKEND_CATALOG；
                       kill_switch.py HCU_DISABLE_OPS 快关注册表；
                       registry.py 运行时实现注册表（register/dump，治理 §4）
sgl_kernel/            同名影子包（sglang 的 AOT 默认内核路径；宿主 import 契约，
                       包名不可改）——elementwise/layernorm/rotary/moe/kvcacheio.py
                       triton/torch 实现；fa_varlen_triton.py 等顶层自研 triton 内核
sitecustomize.py       三行自动加载入口（governance §0 终态）；整链 hip 门控
tests/
  test_ops_cpu.py        CPU 契约面（任何机器）
  tests/kernels/         自有算子+vendor 桥契约（GPU；pytest 收集面）
  tests/adapter/         seam/编排层测试（platform_env/model_env/sitecustomize
                         序列；脚本式测试须函数化+__main__ 守卫，pytest 收集）
  tests/vendor/          高危 vendor 压测
  tests/tools/_*.py      排查探针（手动，路径可用 HCU_ADAPTER_ROOT/HCU_CKPT 覆盖）
  tests/tools/run_triage.py  精度问题四级快速定位入口
  tests/bench/           性能/并发评测（gsm8k_eval，非门禁）
ci/l1_gpu_gate.sh      一键 L1 全门（l0/cpu/kernels/adapter/l15 上游单测/
                       gates/vendor 分段；上游单测零拷贝——收集列表在此维护）
docs/governance.md     算子治理：双仓边界/裁决链/角色/注册表/catalog 状态机/
                       接入时序/共建边界/维护闭环（§7）
skills/hcu-adapter-dev/  本 skill
```

## 协作规约

- 自动化门禁 = `tests/test_ops_cpu.py` + `tests/kernels tests/adapter` +
  `tests/vendor/` + ci/l1_gpu_gate.sh 分段（pytest 可跑、断言带数值谓词）；
  `tests/tools/_*.py` 是排查工具（手动、可硬编码路径但须支持 env 覆盖）；
  `tests/tools/` 不进门禁。
- 改 kernel 必须附带或更新对应契约测试；涉及 vendor 高危面的同时更新压测
  与快关注册表，三件一起提交。
- 桥/plugin 选位与登记（先分仓、裁决链、registry.register、L0 三规则、
  四处登记同 commit）见上文治理闭环与 `docs/governance.md`（§0/§3/§7）
  ——sitecustomize 不新增实现体。
- push 前最低标准：`pytest tests/test_ops_cpu.py`（任何机器）+
  节点上 `pytest tests/kernels tests/adapter tests/vendor -k "not stress"` +
  `python tests/tools/run_triage.py --stage ops`（或直接 `bash
  ci/l1_gpu_gate.sh` 一键全门）。
- commit 信息带根因短句（参考 fbc540c / 1eb038d 风格），便于回溯 bug 类。
- 文档与代码同库同 PR：发现可复用的新坑后立即在修复所在 MR 同步更新本
  skill/对应参考文档，不等最终总结；一次性环境日志只留验证证据。README
  是指针型入口，静态资产表只活在 catalog/models.md（已立规约：README
  不维护清单、不留本地任务编号）。
