# L1.5 v2 上游测试用例清单与分诊规则

> 本文是 L1.5 v2 的**候选账和分类规则**，回答“选哪些上游测试、为什么选、进入哪个
> 集合、还缺什么证据”。当前已落地门禁和历史实测结果以 [ci.md](ci.md) 为准；目标原则
> 以 [test-route-governance.md](test-route-governance.md) 为准；工程任务以
> [test-route-implementation-plan.md](test-route-implementation-plan.md) 为准。
>
> 本文列出的 `candidate-observe` 不代表已经支持或已经进入硬门。测试本体始终来自绑定的
> 社区 sglang 源码树，本仓只保存 manifest、分诊结果和 adapter 侧回归，不复制上游用例。

## 1. 目标与边界

L1.5 验证的是：**社区 sglang 的公开调用面经过 adapter 后，在单卡 HCU 环境中仍满足
上游契约**。它位于 adapter 自有 L1 与模型/服务 L2 之间，不承担完整 serving 验证。

纳入 L1.5 的用例必须同时满足：

1. 来自绑定版本的上游官方测试；
2. 单卡可运行，不启动长期服务；
3. 不下载模型权重或外部数据集；
4. 不依赖 RDMA、外部存储或独立服务；
5. 能验证 adapter/HCU 路径，或能验证对适配有价值的构造/配置契约；
6. 测试结果可归为 pass、精确 known-fail 或稳定平台 skip；
7. nodeid、上游树指纹、分类理由和 owner 可机器记录。

以下不进入 L1.5 core：HTTP/eval、完整模型启动、模型权重、多卡并行、外部服务、性能
压测和长稳测试。纯 NVIDIA/特定 gfx 架构用例只进入 `platform-skip`，不计入 HCU 有效
覆盖。

## 2. 集合定义

| 集合 | 准入判据 | 裁决 | 有效覆盖 |
|---|---|---|---|
| `core-green` | 真实执行 HCU/adapter 路径，连续稳定通过，有 route evidence | 阻塞 | 计入 |
| `contract-green` | 验证构造、配置、metadata 或 weight mapping，但不证明 kernel 命中 | 独立裁决 | 单列，不计 kernel 覆盖 |
| `diagnostic-known-fail` | 精确 nodeid、稳定错误指纹、归因、owner、复审和转绿条件齐全 | 非阻塞，指纹漂移告警/阻塞 | 不计入 |
| `platform-skip` | 平台守卫明确且 skip reason 稳定 | 非阻塞，理由漂移告警 | 不计入 |
| `unclassified` | 已收集但证据不足 | 仅观察，必须继续分诊 | 不计入 |

候选的初始状态统一为 `candidate-observe`。只有完成真实 collect、执行和路由证明后，才可
转入上述四个稳定集合；不能因为文件名看起来相关就直接标 `core-green`。

## 3. 当前已落地集合

### 3.1 Attention core

当前 `l15_upstream` 以目录运行：

```text
test/registered/attention/unittests/dsa
test/registered/attention/unittests/kda
test/registered/attention/unittests/dense
test/registered/attention/unittests/mla
```

现状价值：覆盖 DSA/KDA/Dense/MLA backend 构造、metadata、decode/extend 和部分 FP8 KV
调用面。当前问题是按目录收集、依赖 `-k "not tilelang"`，没有逐 nodeid 固化有效绿色、
合法 skip 和 diagnostic。

### 3.2 Mechanism

当前 `l15_mech` 以目录运行：

```text
test/registered/attention/unittests/gdn
test/registered/attention/unittests/hybrid_linear
test/registered/attention/unittests/mamba
test/registered/attention/unittests/swa
test/registered/attention/unittests/lightning
test/registered/attention/unittests/dsv4
```

现状价值：覆盖线性注意力、窗口、状态和 DSV4 接缝。当前门禁按 passed/failed 总数预算
裁决，无法识别“旧失败消失、新失败出现但总数不变”，因此必须迁移为逐 nodeid 清单。

### 3.3 当前集合的口径

v0.5.22 已知 collect-only 基线约为：Attention core `101/103 collected`（2 个 tilelang
排除），Mechanism `134 collected`，合计约 235 个实例。这是**收集规模**，不是有效 HCU
覆盖数；平台 skip、known-fail 和未证明路由的测试必须从有效覆盖中剔除。

## 4. v2 候选清单

以下路径以社区 sglang v0.5.22 为版本锚。最终 manifest 必须使用实际 collect 得到的
完整 nodeid，不能只保存文件路径。

### 4.1 Batch 1：基础 ABI

| 上游文件 | logical op | 期望证明 | 初始分类 | 进入 core 前置条件 |
|---|---|---|---|---|
| `test/registered/kernels/ops/activation/test_fused_op_gpu_parity.py` | `silu_and_mul` 等 activation | `sgl_kernel` AOT ABI、融合输出 | candidate-observe | 收集成功；真实 HCU 执行；active backend 路由可证 |
| `test/registered/kernels/ops/layernorm/test_fused_op_gpu_parity.py` | `rms_norm`/layernorm | fused-op ABI 与数值 | diagnostic 候选 | 先修复并固定当前 collection error，再转 observe |
| `test/registered/kernels/ops/layernorm/test_rmsnorm.py` | `rms_norm` | dtype/shape/数值 | candidate-observe | 非纯构造；命中 HCU 实现；稳定通过 |
| `test/registered/kernels/ops/layernorm/test_fused_add_rmsnorm.py` | `fused_add_rms_norm` | 原地/融合语义 | candidate-observe | 输出和 residual 语义通过；路由可证 |
| `test/registered/rotary/test_rope_rocm.py` | `rotary_embedding` | ROCm/HIP 路径、布局 | candidate-observe | 平台守卫确实执行而非 skip；backend 可归因 |
| `test/registered/kernels/ops/sampling/test_renorm.py` | sampling renorm | 概率归一化与边界 | candidate-observe | HCU tensor 路径真实执行；无外部资产 |
| `test/registered/kernels/ops/sampling/test_top_p_renorm.py` | top-p renorm | cutoff/tie/归一化 | candidate-observe | HCU tensor 路径真实执行；语义断言稳定 |
| `test/registered/kernels/ops/kvcache/test_store_cache.py` | `kv_cache_store` | 页表、索引、布局写入 | candidate-observe | bit/布局判据通过；命中 adapter/HCU 路径 |
| `test/registered/kernels/ops/kvcache/test_set_mla_kv_buffer.py` | MLA KV buffer | buffer ABI 与写入语义 | candidate-observe | 真实设备执行；无模型或服务依赖 |

Batch 1 首先回答公共符号和基础 fused-op 是否兼容。文件内若同时包含 CPU-only、平台 skip
和 GPU parity 参数实例，必须逐 nodeid 分类，禁止整文件一刀切。

### 4.2 Batch 2：Quantization 与 MoE

| 上游文件 | logical op | 期望 backend | 初始分类 | 已知风险/转绿条件 |
|---|---|---|---|---|
| `test/registered/kernels/ops/quantization/test_fp8_utils_aiter.py` | FP8 utils | vendor/aiter | candidate-observe | vendor 缺失应分为环境 skip，不可伪装 core |
| `test/registered/kernels/ops/quantization/test_per_token_quant_fp8.py` | `per_token_quant` | vendor/triton | diagnostic 候选 | 当前已知顶层 `sgl_per_token_quant_fp8` ABI 缺口；补独立回归后再转绿 |
| `test/registered/kernels/ops/quantization/test_per_token_group_quant.py` | group quant | triton/vendor | candidate-observe | scale/layout/bit 或 dequant 指标通过；route evidence 必需 |
| `test/registered/moe/test_topk_padded_region.py` | `moe_gate`/padding | triton/vendor | candidate-observe | expert/padding bit-exact；非法区不泄漏 |
| `test/registered/moe/test_topk_renormalize_degenerate.py` | `moe_gate`/renorm | triton/vendor | candidate-observe | degenerate/tie 规则一致；非 torch-only 参考路径 |

量化用例不能只用宽松 `allclose`：manifest 需绑定 scale、bit-match 或 dequant 指标。MoE
用例需记录 expert 集合、权重、padding 和 tie 规则，避免“数值接近但专家选择错误”。

### 4.3 Batch 3：Attention 底层调用面

| 上游文件 | logical op/阶段 | 期望证明 | 初始分类 |
|---|---|---|---|
| `test/registered/attention/test_create_kvindices.py` | KV indices | 页表/索引 exactness | candidate-observe |
| `test/registered/attention/test_normal_decode_set_metadata.py` | decode metadata | decode metadata 契约 | candidate-observe |
| `test/registered/attention/test_triton_attention_kernels.py` | attention kernel | Triton prefill/decode 数值 | candidate-observe |
| `test/registered/attention/test_fused_fp8_kv_write.py` | FP8 KV write | 写入、scale、layout | candidate-observe |
| `test/registered/attention/test_kda_kernels.py` | KDA kernels | KDA 低层 kernel 路径 | candidate-observe |
| `test/registered/attention/test_verify_shared_kv.py` | verify/shared KV | speculative verify KV 共享语义 | candidate-observe |
| `test/registered/attention/test_verify_splitkv.py` | verify/split KV | split-KV verify 语义 | candidate-observe |

该批与现有 `attention/unittests` 不重复：现有集合偏 backend/机制集成，本批验证被其调用的
KV index、metadata、写入和 verify 低层契约。若某 nodeid 最终只测试纯 Python metadata，
应进入 `contract-green`，不能计为 kernel route coverage。

### 4.4 Batch 4：Mamba 与 Conv1D

| 上游文件 | logical op | 期望证明 | 初始分类 |
|---|---|---|---|
| `test/registered/kernels/ops/mamba/test_causal_conv1d.py` | `conv1d` | causal conv 数值与状态更新 | candidate-observe |
| `test/registered/kernels/ops/mamba/test_fused_replay_state_indices.py` | replay state | replay 索引 exactness | candidate-observe |
| `test/registered/kernels/ops/mamba/test_sconv_cache.py` | SConv cache | cache 读写与边界 | candidate-observe |
| `test/registered/kernels/ops/mamba/test_sconv_decode_metadata.py` | SConv decode metadata | decode 构造契约 | contract/core 待分诊 |
| `test/registered/kernels/ops/mamba/test_sconv_extend_metadata.py` | SConv extend metadata | extend 构造契约 | contract/core 待分诊 |
| `test/registered/kernels/ops/mamba/test_sconv_tracking.py` | SConv tracking | state tracking | candidate-observe |
| `test/registered/kernels/ops/mamba/test_verify_commit_triton.py` | verify commit | Triton commit 与 replay | candidate-observe |

该批必须覆盖非零 prefix、页边界、batch size 1、换新输入 replay 等历史敏感形状。仅验证
metadata 对象字段的实例归 contract；真正执行 Conv1D/SConv kernel 的实例才可归 core。

### 4.5 Batch 5：模型构造与权重契约

当前历史分诊已经证明以下类型有价值：

- 模型构造与模块注册；
- quant config 解析；
- packed modules/weight mapping；
- packed weight loader；
- pipeline/embedding registration；
- 不加载权重的 MoE gate、NextN 和配置契约。

这些用例初始统一进入 `contract-green` 候选，必须满足“无模型下载、无服务启动、无外部
数据、稳定完成”。manifest 中必须保存实际 nodeid；不能按 `unit/models` 整目录收入，
也不能把构造契约数量并入 HCU kernel 有效覆盖。

### 4.6 已有功能单测候选的边界

历史分诊中，PD/disaggregation unit、spec/dspark 纯单测和 mock-model canary 存在大批
绿色用例。它们可建立独立 `contract-green` 扩展段，但不优先进入普通 MR core：

- mock 通过只能证明 wiring/协议，不证明真实 HCU kernel；
- PD、HiCache、DFlash 中涉及外部服务、文件后端、HF 资产或 boot 的用例排除；
- 多卡和真实传输路径归 L2-F/L3；
- 只有无服务、无网络、单卡且对 adapter 契约有直接价值的 nodeid 才纳入。

## 5. Manifest 每条记录的最小字段

```yaml
nodeid: test/registered/.../test_x.py::test_case[param]
domain: quantization
logical_ops: [per_token_quant]
classification: diagnostic-known-fail
blocking: false
expected_backends: [vendor, triton]
route_evidence: required
requirements:
  device_count: 1
  server: false
  model_weights: false
  external_network: false
platform_guard: null
current_result:
  outcome: collection_error
  fingerprint: "cannot import name 'sgl_per_token_quant_fp8'"
ownership:
  owner: quantization
review_condition: "public ABI is implemented with an adapter-owned regression"
promotion_condition: "stable green runs and expected route observed"
```

强制规则：

- `classification=core-green` 时，`route_evidence` 必须为 `required` 且实际报告非空；
- `classification=diagnostic-known-fail` 时必须有 fingerprint、owner 和 promotion condition；
- `classification=platform-skip` 时必须有稳定 platform guard 和 skip reason；
- 结果未采集写 `not_collected`，不得用 pass、skip 或 0 代替；
- nodeid 参数变化、消失或新增必须产生 manifest diff。

## 6. 从候选到硬门的状态机

```text
candidate-observe
  ├─ 真实执行 + 预期路由 + 稳定通过 ───────────→ core-green
  ├─ 只验证构造/配置/metadata，稳定通过 ──────→ contract-green
  ├─ 稳定失败 + 精确指纹 + owner/解除条件 ───→ diagnostic-known-fail
  ├─ 稳定平台守卫 + 理由明确 ────────────────→ platform-skip
  └─ 证据不足 ───────────────────────────────→ unclassified
```

晋级 `core-green` 必须同时具备：

1. 上游版本与树指纹匹配；
2. collect nodeid 稳定；
3. HCU 上真实执行，非 xfail/skip；
4. 数值或语义判据通过；
5. 实际 backend 与预期一致；
6. 无 server/model/network/multi-GPU 隐含依赖；
7. owner 完成分诊确认。

降级或移出 core 的条件包括：路由证据消失、变为平台 skip、引入外部资产、错误指纹变化、
上游删除 nodeid 或测试不再覆盖原 logical op。降级必须保留原因，不能静默从 manifest 删除。

## 7. Collect 与执行输出

每次 inventory 至少产出：

```text
inventory.json       # 所有候选和静态依赖标签
collect.json         # 实际 nodeid、参数、collect error
results.json         # pass/fail/skip/xfail 与错误指纹
routes.json          # logical op/backend/count/evidence
classification.json  # 最终分类及理由
summary.md           # 新增、消失、漂移和待分诊摘要
```

报告必须分别给出：

- collected 数；
- executed 数；
- core/contract/diagnostic/platform-skip/unclassified 数；
- 有 route evidence 的 core 数；
- 有效覆盖分母和通过率；
- 新增/消失 nodeid；
- skip reason 和 diagnostic fingerprint 漂移。

禁止只报告“总共多少 passed”。

紧凑清单压缩前先解析文档级 `classification`/`route_evidence` 默认值；
未声明时分别为 `candidate-observe`/`required`。压缩再展开不得改变生效值，
尤其不能把 `core-green` 的证据要求降为 `not_collected`。
设备数、返回码与分类计数字段必须是整数，JSON 布尔值不作为整数接受。

通信 runner 以非零返回码保留真实失败：pytest 的 failed+skipped 混合摘要
仍记为 failed，仅成功退出或 pytest 无用例（rc=5）允许判为 skipped。
observe 保持非阻塞，但 artifact 的失败结果及 `hard_failures` 必须如实记录；
torchrun 脚本的 rc=5 不具有 pytest 的“未收集”语义。

## 8. 首轮梳理结论

1. 保留现有 Attention core 和 Mechanism 作为迁移基线，但停止把目录总数视为最终裁决；
2. 优先分诊 Batch 1，先补公共 ABI、Norm/RoPE/Sampling/KV Cache 的明显空白；
3. `test_per_token_quant_fp8.py` 和 LayerNorm parity 收集问题先进入 diagnostic，不静默排除；
4. Batch 2–4 按域进入 observe，逐 nodeid 证明 route 后再晋级；
5. 模型构造、PD/spec/mock 等绿色单测单列 contract，不抬高 kernel 覆盖率；
6. 所有纯平台 skip 从有效覆盖分母剔除；
7. 现有 `l15_mech` 总数预算只作为迁移期兼容门，精确 manifest 稳定后退出裁决。
