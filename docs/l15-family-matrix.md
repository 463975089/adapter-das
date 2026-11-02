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
| Mamba | Mamba2 prefill/decode、碎片布局、replay metadata、hybrid dispatch、Conv1D varlen state | 上游 Mamba2 四项 + adapter Conv1D |

## 裁决口径

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

1. Mamba 真实 CUDAGraph fresh-input replay；
2. FP8 grouped expert GEMM 的小形状数值合同；
3. INT8 MoE 两段 GEMM 的端到端合成合同；
4. INT4/W4A8 在 HCU 上主要是 correctness-first W4A16 降级，不能宣称原生 CUTLASS 行为；
5. v0.5.22 部分候选路径在本地后续快照已重命名，必须以 manifest 指纹绑定的目标树 collect 为准；
6. 该矩阵仍不覆盖权重加载、scheduler、HTTP serving 和真实模型生成。

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
