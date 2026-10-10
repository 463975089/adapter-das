# 自动化测试套件要求

目标：**adapter API 全绿 ⇒ 官方 sglang 在 HCU 上无大问题**。为此每个出口
API 必须覆盖四个质量维度，缺一不可（每个维度都有历史 bug 背书）：

## 四质量维度

| 维度 | 断言 | 不测会漏掉的 bug（真实案例） |
|---|---|---|
| 1 数值契约 | vs torch fp32 参考：cos ≥ 0.9999 且 maxrel ≤ tol | rope cache 半分布局（fbc540c）："能跑"但语义错 |
| 2 graph-safety | capture→replay 换新输入，输出必须跟随 | AiterCustomAllreduce 图内重放损坏（1eb038d）：eager 全对 |
| 3 场景变体 | 短/长 prompt、奇数长度、变长打包、因果/非因果 | FA/AR 类 bug 只在特定 size 出现 |
| 4 集成金标 | 层级隐藏态对齐 + e2e golden | 单算子都对、组装点错（路由/量化衔接） |

**断言规范**：bf16 链路 top-k 平局重排是常态 → e2e 用分布级断言
（top-k 集合带内 / 文本金标 / 语义锚点），不逐 token 比相等。
算子级用 `cos + maxrel` 双指标：布局类 bug 会毁 cos，噪声不会。

tol 参考：bf16 maxrel ≤ 5e-2（GEMM 类，excess ≤ 1.05× 输入舍入下限）、
fp16 ≤ 1e-2、fp32 ≤ 1e-4；AR 类用 `tol × max|x|` 的自适应 atol。

## 文件清单

| 文件 | 内容 | 运行 |
|---|---|---|
| `test_ops_cpu.py` | 无 GPU 也能跑的契约面 | 任何机器 |
| `test_gpu_kernels.py` | sgl_kernel triton/torch 档全量契约 | 1 GPU |
| `test_vendor_allreduce.py` | AR：eager/graph-replay × 尺寸×dtype + gate 三态 | 1 GPU（gate）/ 2 GPU（stress，自驱动 torchrun） |
| `test_flash_attn_gpu.py` | FA 场景矩阵（探针式后端注册）+ graph-replay | 1 GPU |
| `run_triage.py` | 四级定位流程（见 precision-triage.md） | 容器内 |

## 高危 vendor 压测矩阵

vendor 同事自测通过的 ≠ 我们场景能过。接入 vendor 档（或怀疑 vendor
kernel）时，以下矩阵是最低要求：

**custom all-reduce**（`_ar_worker.py`）：
- 尺寸 8K/64K/256K/1M/8M 元素 × dtype bf16/fp16/fp32；
- eager vs **fp32 精确和**（all_gather 后 fp32 累加，不用 NCCL 当参考——
  它也在被测链路里，只做 harness sanity）；
- graph capture 后 replay 4 次、每次换新输入——检测"烘焙/损坏"两类。

**flash attention**（`test_flash_attn_gpu.py`）：
- (qo,kv) ∈ {(1,1),(1,7),(1,127),(1,1024),(3,129),(17,1023),(127,127),
  (128,128),(129,257),(1023,1023),(1024,1),(2051,2051)}，
  causal 双向 × head_dim {64,128} × dtype {bf16,fp16}——重点奇数长度、
  qo≠kv、2 的幂边界 ±1；
- graph-replay 变输入（decode 形状）。

新后端接入：在 `test_flash_attn_gpu.py` 的 `load_backends()` 注册一个
探针类（包一层把统一入参翻译成该后端 ABI），矩阵自动全量跑一遍。

## 新增算子的最低测试配置

契约测试（维度 1）必带；进 decode 链路的加维度 2（参照
`_graph_safety_probe.py` 模式）；vendor 档再加维度 3 压测 + 快关。
跑法写在文件 docstring，容器外开发时用 `HCU_ADAPTER_ROOT`/`HCU_CKPT`
覆盖探针路径。
