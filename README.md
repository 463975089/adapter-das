# hcu-sglang-adapter

不修改 sglang 官方代码、让社区版 sglang 直接跑在海光 HCU（DCU/DTK，HIP 路径）上的适配层。

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

## 覆盖范围

| 模块 | API | 实现 |
|---|---|---|
| elementwise | `silu_and_mul` / `gelu_and_mul` / `gelu_tanh_and_mul` / `gelu_quick` | triton（CPU/不支持 dtype 走 torch） |
| layernorm | `rmsnorm` / `fused_add_rmsnorm` / `gemma_rmsnorm` / `gemma_fused_add_rmsnorm` | triton（fp32 累加；CPU 走 torch） |
| rope | `rotary_embedding`（NeoX 布局） | triton；GPT-J 交错布局暂走 torch 参考 |
| moe gate | `topk_softmax` / `topk_sigmoid` / `moe_fused_gate`（新旧两种签名） | 批量 fp32 torch（严格对齐 CUDA 核的排序/tie-break/共享专家契约） |
| moe align/sum | `moe_align_block_size` / `moe_sum` / `moe_sum_reduce` | `moe_align_block_size` 为纯 device triton 四 kernel（CUDA-graph capture 安全）；`moe_sum`/`moe_sum_reduce` 走 torch |
| kv cache io | `transfer_kv_*` 共 16 个 | torch 移植（索引 gather/scatter 契约对齐 CUDA 版） |

设计原则：
- API 签名与官方 `sgl-kernel/python/sgl_kernel` 逐字对齐（含 `out=`、
  `enable_pdl` 占位参数、destination-passing 语义）。
- 所有 triton kernel 数学均在 fp32 累加；bf16/fp16/fp32 输入。
- import 永不失败：未覆盖的算子在使用时抛带提示的 `NotImplementedError`。
- GPU 路径禁用一切主机同步（`.item()` / `nonzero` / 布尔掩码索引），
  保证 decode CUDA-graph capture 链路可用（首个 MoE 模型实测暴露的坑）。

## HIP 适配桥（sglang/vllm 零改动）

| 组件 | 作用 |
|---|---|
| `sitecustomize.py` | PYTHONPATH 自动加载：桥接 vllm `_custom_ops` 的 `rms_norm` / `fused_add_rms_norm`（HIP 6 参形态）到本包；`apply_token_bitmask_inplace_cuda` → sglang 仓内 triton 实现 |
| `sgl_kernel/_adapters/` | 元路径桩：`aiter` / `flashinfer` / `flash_attn` import 成功、未实现调用响亮报错；其中 `aiter.silu_and_mul` 用本包 triton 核真实现（官方 parity 测试通过） |
| `sgl_kernel/kvcacheio.py` | PD 分离/分层 KV transfer 的完整 torch 移植 |

## 运行要点（HCU 实测）

```bash
export PYTHONPATH=/path/to/hcu-sglang-adapter:/path/to/sglang/python
python -m sglang.launch_server --model-path <model> --port 30000 \
    --attention-backend triton ...   # aiter attention 后端在 HCU 不可用，须显式指定 triton
```

- MoE 模型（已验证）：Qwen3.6-35B-A3B（GDN 线性注意力 + MoE topk8）单卡 144G
  TP1 拉起，decode CUDA-graph capture 通过；dense 小模型 Qwen3-0.6B 亦通。
- thinking 模型建议给足 `max_tokens` 或用
  `chat_template_kwargs: {"enable_thinking": false}`，否则思考文本会占满
  输出预算（评测误判的常见来源）。

## 测试

```bash
pytest tests/test_ops_cpu.py -v          # 纯 CPU，验证 torch 兜底路径（任何机器）
pytest tests/kernels -v                  # 自有算子（HCU 节点，含 CUDA-graph capture 回归）
pytest tests/vendor -v                   # vendor 高危：FA 场景矩阵 + aiter AR 压测/快关 gate
python tests/tools/run_triage.py         # 精度问题四级快速定位（容器内）
```

- 单测以官方 CUDA 核的语义为基准对照（gate 排序/tie-break、
  moe_align_block_size 的 padding/cumsum 契约）；CPU 36 例 + GPU 29 例全绿
  （GPU 含 moe_align capture/replay 回归）。
- 官方 sgl-kernel parity：fused-op GPU 对照 12/12（`tests/kernels/`）。
- 高危 vendor 算子（custom all-reduce / flash attention）有专门压测：
  尺寸×dtype×CUDA-graph-replay 变输入，黄金参考为 fp32 精确和 / fp32 SDPA。
- 高危算子均配**快关**：`HCU_DISABLE_OPS=aiter_ar[,aiter_fa]` 一翻开关即
  绕开疑似算子复跑（gfx93* 上 aiter AR 默认自动拒绝，`HCU_AITER_AR=1` 夺回）。
- 端到端精度：Qwen3.6-35B-A3B 对话验证 + 64 并发 soak 0 错误 + GSM8K 前 100
  题（thinking 原生模式）69%，错例为思考截断与模型自身水平，算子链路无损；
  GLM-5-W4A8_V2_6 TP8 + CUDA graph ON 零旗标验证连贯（短/长 prompt）。

## 开发规约（skill）

改本仓库前先读 [`skills/hcu-adapter-dev/SKILL.md`](skills/hcu-adapter-dev/SKILL.md)：
影子库矩阵与三档实现策略（triton/vendor/torch）、测试四质量维度、
快关注册流程、精度定位流程、cookbook（NVIDIA 参数）降级矩阵。

## 已知未覆盖 / 路线图

- 量化路径：compressed-tensors（w4a8/int8 GEMM）、fp8 blockwise 等未做；
  跑共享盘 GLM-5.3-Flash INT4 量化版（160G）前需补齐。
- `sgl_kernel.allreduce`（TP 多卡 custom allreduce）：TP1 不需要，多卡待补。
- aiter attention 的 AMD 原生符号（`paged_attention_ragged` / `mla_*`）：
  按需 triton 化（混合适配方案，见任务 #240 结论）。
- MoE gate（topk 系列）triton 化提性能：当前 torch 版正确性已验证，性能优化
  排期中。
