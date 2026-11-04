# L1.5 Dense / MLA / MoE / Quantization / Mamba 代表矩阵

该矩阵用于回答“社区 sglang + adapter 是否能在不加载模型权重、不启动服务的条件下，走通五类主要模型行为”。它不是性能测试，也不替代真实模型 L2。

## 固定边界

- 绑定社区 sglang v0.5.22 源码树；
- adapter 优先进入 `PYTHONPATH`，不修改上游源码；
- 恰好 1 张 HCU；
- 无服务、无模型权重、无外部网络；
- 只执行 manifest 内固定的 exact nodeid 或仓库相对脚本；
- 不接受任意 pytest 参数或 shell 片段。

入口：

```bash
python tools/l15_family_matrix.py validate
python tools/l15_family_matrix.py run \
  --up-tree "$HCU_UP_TREE" \
  --families dense,mla,moe,quantization,mamba \
  --out results/l15-family-matrix/run.json
```

## 代表行为

| 家族 | 核心行为 | 当前代表用例 |
|---|---|---|
| Dense | backend 构造、prefill、decode、GQA/MQA、页边界、碎片 KV、graph metadata/replay contract | 上游 Dense Triton 三个 exact class method |
| MLA | Tiny DeepSeek MLA 构造、compressed KV prefill/decode、碎片布局、graph contract、sparse virtual sink | 上游 MLA Triton 三项 + adapter sparse FlashMLA |
| MoE | Top-K 权重、grouped gate、block alignment、EP `-1` sentinel、capture-safe metadata | adapter MoE gate/align 五项 |
| Quantization | FP8 row/group/fused/scaled quant、scale 语义、异形 fallback | adapter Aiter quant 五项 |
| Mamba | Mamba2 prefill/decode、碎片布局、真实变输入 graph、状态槽位隔离、hybrid dispatch | 上游 Mamba2 四项 + adapter Mamba2/GDN 计算链与 Conv1D |

## 裁决口径

- 严格运行现强制单张真实 HIP 设备、绑定文件指纹和源码导入来源；
  逐 nodeid 的 call 阶段必须通过，skip/xfail/xpass/缺失报告均失败。
  一键入口与本轮执行计划见 [l15-readiness-plan.md](l15-readiness-plan.md)。
- sparse emu 数学项标为 `cpu-contract`，不计 HCU 内核覆盖。

- 上游 class method 内的 `subTest` 会作为一个 exact nodeid 整体裁决；这是上游粒度限制。
- 名称带 `cuda_graph` 的上游 Dense/MLA/Mamba runner 主要验证 capture/replay metadata 生命周期，不自动等价于真实 `torch.cuda.CUDAGraph` launch 证明。
- MoE/Quantization 先使用 adapter-owned GPU contract，因为 v0.5.22 不同源码快照中的上游文件路径已有漂移；应在目标 HCU 树实际 collect 后再把稳定 nodeid 纳入主 classified baseline。
- 用例通过只形成 family-matrix green；晋级 `core-green` 仍需 route evidence，证明未意外走 CUDA stub 或错误 fallback。

## 历史 HCU 实测结果

拆分前的兼容快照在与 manifest 两个 binding hash 完全一致的社区 v0.5.22
源码树上，以单张 gfx938 HCU 完整执行：**22/22 PASS**。该快照同时包含下述
生产兼容修复，现移至 [PR #27](https://github.com/HYGON-AI/adapter-das/pull/27)；
不能据此声称仅合入工具链的分支也已达到同样的 GPU 结果。

| 家族 | 通过 |
|---|---:|
| Dense | 3/3 |
| MLA | 4/4 |
| MoE | 5/5 |
| Quantization | 5/5 |
| Mamba | 5/5 |

该历史快照为取得上述结果修复了四类 adapter 兼容缺口，生产实现不在工具链 PR 内：

- 新版 paged allocator 无条件导入的 HCU kvcache ABI，回落到上游 Triton/torch correctness 路径；
- speculative 与 sparse attention 的可选 public import surface，未选择的能力允许收集，实际调用仍 loud fail；
- Mamba causal Conv1D 所需 CUB block headers 转发到 hipCUB；
- column-major FP8 scales 从无法在 gfx938 编译的 CUDA JIT 路径改为 torch correctness fallback。

该结果证明 family-matrix 行为通过；仍不自动等价于全部 case 已具备独立 route evidence 或性能达标。

## 当前明确缺口

1. Mamba2 组件级真实 CUDAGraph fresh-input replay 已补齐（见下）；纯
   Mamba2 真实 checkpoint、更多形状及 speculative/graph padding 仍未覆盖；
2. FP8 grouped expert GEMM 已增加小形状数值合同（见下），但真实 DeepEP
   多卡路由及模型精度仍未由此验证；
3. INT8 MoE 两段 GEMM 的端到端合成合同已增加，完整真实 checkpoint 精度仍缺；
4. INT4/W4A8 在 HCU 上主要是 correctness-first W4A16 降级，不能宣称原生 CUTLASS 行为；
5. v0.5.22 部分候选路径在本地后续快照已重命名，必须以 manifest 指纹绑定的目标树 collect 为准；
6. 该组件矩阵仍不覆盖权重加载、scheduler、HTTP serving 和真实模型生成；
   独立 Engine 冒烟的覆盖与实测见 [l15-readiness-plan.md](l15-readiness-plan.md)。

2026-11-03 扩展：新增 BF16 Triton MoE 的 up/SiLU/down/weighted-combine、
社区动态 INT8/FP8 Linear，以及 GDN 非方形状态池、envelope stride、
禁写和 ReplaySSM ring 验证。以上链路与 FP8/INT8 GEMM 都捕获真实 graph，
连续三轮变输入对拍。FP8 Linear 保留实际平台默认 dispatch；不伪报
CUTLASS 支持。独立 GEMM 用例验证 adapter 的 Triton 实现。
Conv1D emu 项与 sparse emu 一样明确标为 CPU 契约，不计 GPU 覆盖。

2026-11-03 真实 Channel-FP8 模型 trace 确认了社区 CT W8A8-FP8 →
dynamic token quant → fallback scaled-mm 路径。`quant-linear-chain`
补充 `ct-fp8` 参数：使用真实 scheme、channel scale、加载后处理和默认
padding，在 1/17/33 token 下独立 FP32 对拍并做三轮变输入 graph replay，
没有强制 CUTLASS 或替换平台路由。该 case 下现在收集 9 个参数化节点。
nmz7 GPU5 的量化家族严格门 **8/8 case PASS**，包括这 9 个节点，
各执行后社区源码完整性通过；报告
`results/l15-readiness/iteration21-quantization-family.json`。
这不是重跑全部五家族，也不替代真实量化模型的独立数值验收。

新增 `moe-int8-expert-chain`：真实官方
`W8A8Int8MoEMethod` 的权重创建、加载后处理、Triton runner、
top1/top2 路由及两段 GEMM/SiLU/combine，在 1/7/33 token 下验证。
量化整数先以独立 FP64 契约校验，仅在 FP32 reciprocal 的舍入误差界内
允许半整数两侧取整；尺度误差也单独断言，再以已验证的整数/尺度执行
独立 FP32 GEMM 参考。最终输出沿用原数值阈值和 cosine ≥ 0.9999，
并进行三轮变输入 graph replay。nmz21 GPU3 的组件回归 **3 passed**，
见 `results/l15-readiness/iteration30-int8-moe-chain.log`；
这不是全部 29 case 的整轮证据。

严格 MoE 家族增量验证 **7/7 case PASS**（零 skip/xfail，执行后社区源码
完整性通过），见 `iteration30-moe-family.json`。显式 W8A8 Linear 另增加
`quant-w8a8-linear`，矩阵现在为 30 case；公开权重创建、加载后转置、
per-token 量化与 scaled GEMM 覆盖 1/17/33 token、bias 两态及三轮变输入
graph，共 6 个参数化节点。输入设为可精确量化的二进制格点，隔离前述
半整数不确定性；独立 FP32 GEMM/尺度/bias 对拍保持原输出阈值。
nmz21 GPU4 的原始 INT8 GEMM 与该新链路合计 **20 passed**，
见 `iteration31-int8-linear-gpu.log`；完整 30 case 尚待整轮验证。

2026-11-04 nmz7 GPU4 的量化家族严格门 **9/9 case PASS**，包含新增
W8A8 Linear 的 6 个节点及 CT-FP8 等 9 个 quant-linear 节点，零 skip/xfail，
执行后社区源码完整性通过；报告 `iteration34-quantization-family.json`。
这是量化家族整轮证据，与 MoE 7/7 的独立报告分别保留。

新目录组合在 nmz28 单卡 GPU 0 的严格矩阵实测为 **27/27 PASS**，
原始报告 `results/l15-readiness/phase2-all.json`。其中 25 个 GPU case，
另 2 个是显式 CPU 契约。社区路由入口接入后，计算链与 GDN 桥回归
另跑 **13 passed**。这次补链实际抓到并修复 GDN bridge 的多余状态转置；
内核实际按 `[slot, head, V, K]` 访问，保留原 pool view 才能同时保持数值
和 envelope slot stride。仍保留上述 FP8/INT8 专家量化等缺口。

第三步再新增非零 RoPE MLA case，当前清单共 28 case。新增 case 的
`256+32`、`512+64` KV 布局各测 prefill/decode/真实 graph，共 6 个
参数化节点在同节点全部通过；graph 连续三轮变输入独立对拍。
原 27 case 的整轮报告与新增 case 的增量报告分别保留，没有将两次执行
描述成一次完整 28 case 运行。模型级 Mamba graph 也已通过，但不替代
上述组件级 Mamba 独立变输入数值对拍缺口。

2026-11-04 新增 `mamba-fresh-input-graph`，固定清单增至 31 case。
用社区 fixture 构造真实 `MambaMixer2`，执行官方投影、causal Conv1D、
SSM、gated RMSNorm 和输出投影。CPU FP32 参考独立计算上述数学，
仅在公开的激活/状态存储边界转 BF16，不调用生产算子或其参考 forward。
batch1/page1/group1 与 batch3/page16/group2 各捕获一次真实 CUDAGraph，
三轮改变输入及非连续状态槽位；独立参考状态连续递推，
检查输出、卷积与 SSM 状态、eager/graph 一致性及未使用槽位不被改写。
这不替代 scheduler 图回放、纯 Mamba2 模型或 speculative 验收。

nmz7 GPU4 的整个计算链文件 **25 passed**；严格 Mamba 家族
**7/7 case PASS**（包含 2 个新参数节点，零 skip/xfail），矩阵工具
回归 **32 passed**，执行后社区源码完整性通过。证据：
`results/l15-readiness/iteration39-chains.log`、
`iteration39-mamba-family.json`、`iteration39-matrix-cpu.log`、
`iteration39-mamba-runtime.json`。该报告仅覆盖 Mamba 家族，
不是完整 31 case 的一次整轮运行。

2026-11-04 新增 `fp8-grouped-expert-contract`，清单增至 32 case。
masked 与 contiguous 两种 DeepEP 布局各覆盖 K/N=128 和 256，
128×128 权重 block scale、per-token-group 激活 scale、有符号 FP8，
以及非连续 scale view。masked 覆盖空专家、1/17/32 个有效 token
及 padding 禁写；contiguous 按 64-row expert alignment，覆盖 129 行
的尾 tile。每项独立 FP32 反量化对拍，保持 cosine ≥0.9999、
maxrel ≤0.05，真实 graph 三轮改变输入、权重、尺度和设备端
counts/expert IDs。这些是 adapter Triton 实现的直接合同，
不宣称已经覆盖分布式 dispatch/combine 或默认 DeepEP 选型。

nmz7 GPU5 整个 FP8 GEMM 文件 **17 passed**（含新增 4 个参数节点），
执行后社区源码完整性通过，见
`results/l15-readiness/iteration40-fp8-grouped.log` 和
`iteration40-fp8-grouped-runtime.json`。
接入后的严格量化家族 **10/10 case PASS**，零 skip/xfail，矩阵工具
**32 passed**，执行后社区源码完整性通过；分别见
`iteration41-quantization-family.json`、`iteration41-matrix-cpu.log` 和
`iteration41-quantization-runtime.json`。L0 **GREEN（31 warn）**、纯
CPU 算子 **38 passed**，见 `iteration41-l0.log`、`iteration41-ops-cpu.log`。
Mamba 7/7 与量化 10/10 是分别执行的家族报告，不拼接为整轮 32 case。

## DoD

矩阵完成的必要条件：

- 五个 family 均至少有一个构造/路由用例和一个数值或状态语义用例；
- 结果 artifact 逐 case 记录 argv、returncode、timeout、stdout/stderr；
- 任一固定 case 失败，runner 返回非零；
- 缺失上游文件 loud fail，不静默跳到其他源码树；
- 后续 HCU 实测结果按 exact case 更新，不以总通过数掩盖替换失败。

输入校验要求非空 `manifest_id`、正整数 `defaults.timeout`，以及恰好五个
不重复的家族；`families`/`cases` 每项必须为对象。布尔值不能冒充整数，
坏输入通过 `MatrixError` 返回可读错误。artifact 原子写入失败时清理临时文件，
并保留原有结果文件，避免磁盘错误产生残留或覆盖上一次有效证据。
