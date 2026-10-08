# hcu-sglang-adapter

不修改 sglang 官方代码、让社区版 sglang 直接跑在海光 HCU（DCU/DTK）上的适配层。

核心是本仓库的 **`sgl_kernel` 包**（triton 实现）：sglang 的统一算子层
`sglang.kernels`（RFC #29630 的 `BaseFusedOp` 多后端模型）在 HIP 平台上把
**AOT 后端**（`forward_aot` → `import sgl_kernel`）作为默认内核路径
（JIT triton 后端只注册给 CUDA）。因此只要提供一个同名的 `sgl_kernel`
包，上游 sglang 零修改即可在 HCU 上解析到我们的内核。

## 安装

```bash
pip install -e .
# 之后正常安装/运行 sglang（不要安装官方 sgl-kernel wheel，避免同名冲突）
```

## 覆盖范围（v1）

| 模块 | API | 实现 |
|---|---|---|
| elementwise | `silu_and_mul` / `gelu_and_mul` / `gelu_tanh_and_mul` / `gelu_quick` | triton（CPU/不支持 dtype 走 torch） |
| layernorm | `rmsnorm` / `fused_add_rmsnorm` / `gemma_rmsnorm` / `gemma_fused_add_rmsnorm` | triton（fp32 累加；CPU 走 torch） |
| rope | `rotary_embedding`（NeoX 布局） | triton；GPT-J 交错布局暂走 torch 参考 |
| moe | `topk_softmax` / `topk_sigmoid` / `moe_fused_gate`（新旧两种签名）/ `moe_align_block_size` / `moe_sum` / `moe_sum_reduce` | 批量 fp32 torch（严格对齐 CUDA 核的排序/占位契约），triton 化在后续迭代 |

设计原则：
- API 签名与官方 `sgl-kernel/python/sgl_kernel` 逐字对齐（含 `out=`、
  `enable_pdl` 占位参数、destination-passing 语义）。
- 所有 triton kernel 数学均在 fp32 累加；bf16/fp16/fp32 输入。
- import 永不失败：未覆盖的算子在使用时抛带提示的 `NotImplementedError`。

## 测试

```bash
pytest tests/ -v          # HCU/CUDA 节点上跑（纯 CPU 也可验证 torch 兜底路径）
```

单测以官方 CUDA 核的语义为基准实现参考对照（softmax/sigmoid gate 的排序、
tie-break 与 moe_align_block_size 的 padding 契约）。
