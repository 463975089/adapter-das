# 自动化测试套件要求

共享机制的 framework-free 契约覆盖 import 和调用两个阶段。模型 lifecycle
测试须在新解释器阻断 vLLM、SGLang、Torch 导入后，执行真实注册、原子预检、
post-load/guard dispatch；另以真实 vLLM Worker/config 验证桥上的原覆盖查询、
空注册表短路和调用顺序。双宿主 bootstrap/import 回归不等于模型或 GPU 验收。
公开门面迁移须检查 source/wheel/install 来源及实际消费者的 callable ABI，
不能用安装成功代替宿主语义验证。

guard 的非 None 返回属于契约违规，不能被回调异常包装成主动 refused；分别断言
诊断分类、owner/operation/role、原异常 cause、后续 guard 不执行和 dispatch 状态
恢复。误返回的协程须关闭而不执行。修复共享 dispatch 时同步保留真实宿主入口回归。

共享 lifecycle 的 CPU 门禁包含 `test_runtime_lifecycle.py`、实际双宿主
`test_bootstrap.py`、portable sequence/isolation 子集，以及原 vLLM 健康、
selector 和真实 metadata/spawn 门禁。双宿主 import 探针拒绝框架/Torch/
kernel/provider import；named consumer 回归不得冒称 SGLang engine 试验。

vLLM 扩展生命周期：source 与独立 wheel installed 分开跑。check_health 的
必需参数须在激活前拒绝；报告替换/重复、sink 失败、caught error 后其他
selector、实际 fork 格式化窗口均需断言。public registry clear/foreign 必须
经真实官方 selector 验证，不以直接 checker 测试替代。见
`docs/vllm-flash-registry.md`；不把 CPU 协议验证当作数值验收。

目标：**adapter API 全绿 ⇒ 官方 sglang 在 HCU 上无大问题**。为此每个出口
API 必须覆盖四个质量维度，缺一不可（每个维度都有历史 bug 背书）：

## 四质量维度

vLLM mHC 安装态生命周期探针为 `tests/tools/_probe_vllm_mhc_lifecycle.py`，
其 CPU 身份 double 与真实 native/GPU 门禁分开。官方 Torch 复用仍须数值、
同地址变输入 Graph、长/短/空批次和 layer/e2e golden。capture 不执行 Graph；
先 replay 核对原输入基线，空 Graph 单列。原门槛下 vendor 大 batch 舍入
失败保留，不得用另一参考或另一实现追认原 fused 通过。

vLLM 0.33 的 draft-weight-update 真实路径会遍历 draft model 来恢复 source
LM heads。CPU runner 夹具须返回空 `torch.nn.Module`，namespace 不满足
官方契约；保留真实遍历，不把它 mock 掉。多次 spawn 的整文件预算与
每个子进程 timeout 分开计算，截断后须保留原日志，以独立完整运行补证。

vLLM dense FP8官方fallback接线探针：`tests/tools/_probe_vllm_fp8_linear.py`
直接验证既有官方provider；`_probe_vllm_fp8_family.py` 验证runtime正式注册、
官方选型、QuantFP8及apply_weights完整链路。非单位tensor/token/channel尺度、
bias和三波变输入Graph必须对拍FP32；只验证的gfx936不为gfx938背书。
桥接契约由 `tests/adapter/test_vllm_fp8_bridge.py` 离线覆盖。命令与退出条件
见 `docs/frameworks/vllm032.md`，禁止通过伪CDNA或放宽native Torch支持谓词接线。

vLLM 启动故障回归须覆盖异常 `__str__` 自身失败（普通 Exception 与
BaseException）、格式化期间实际 fork、父子 health/同进程 retry 持续拒绝。
公开 ExtensionCommitError 经坏字符串 wrapper 包装也须保留真实 FAILED
报告并停止后续 owner；干净 optional 的诊断失败仍保持原有降级语义。
使用实际 metadata/官方 general-plugin loader 与 spawn/unpickle 验证新安装
产物；源码计数和旧 wheel 证据不可转授。该 seam 不引入算子质量晋级。

CPU required-policy 用例须显式隔离 native ABI 与 FP8 注册前置条件，避免
在被测 required 检查前退出。runtime ABI 用例保留真实官方 loader 和无效库
负控，只隔离后续 FP8 provider；扩展替身须接收并检查公开
`health_check_sink` 参数。前置隔离不能用于设备或 native ABI 验收。

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

## 文件清单与分类规划（73-296：单测越多越要先分好类）

**分类原则**：一级按运行环境/对象（cpu / gpu 自有 kernel / vendor 高危 /
bench / tools），二级按算子域。命名即分类：`test_<域>_<维度>.py`。

```
tests/
  test_ops_cpu.py            # CPU 契约面（torch 参考档+包装面），任何机器跑
  kernels/                   # sgl_kernel 自有算子（GPU）
    test_elementwise.py      #   silu/gelu 系列：契约+graph-safety
    test_layernorm.py        #   rmsnorm/fused-add/gemma：契约+graph-safety
    test_rope.py             #   neox/interleave：契约+cache 布局回归(fbc540c)
    test_moe_gate.py         #   topk 系列：排序/tie-break/共享专家契约
    test_moe_align.py        #   align/sum：capture 回归（527ad97）
    test_kvtransfer.py       #   transfer_kv_*：索引契约
  vendor/                    # 高危 vendor 压测（一份矩阵多后端）
    test_allreduce.py        #   aiter AR：eager+graph-replay+gate 三态
    test_flash_attn.py       #   FA 场景矩阵+graph-replay（探针式后端）
  bench/                     # 性能/并发基准（非门禁，长跑）
    gsm8k_eval.py            #   GSM8K-N 并发精度快评
  tools/                     # 排查工具（手动，_ 前缀约束同旧）
    probes（rope/rms/gemm/graph-safety 探针，HCU_ADAPTER_ROOT 可覆路径）
    run_triage.py            #   四级定位入口
    _ref_forward_h/_bisect_cmp   # 层对齐参考与二分
```

**门槛连接**：门禁 = `test_ops_cpu.py` + `kernels/` + `vendor/` 里的 gate 测试
（fast 集合）；`vendor/` 的 stress 与 `bench/` 为 nightly/按需。
单测新增一律先进对应二级域文件，不为每个 bug 新开文件（回归用例打注释标
bug 号，如 fbc540c/1eb038d）。

## 高危 vendor 压测矩阵

vendor 同事自测通过的 ≠ 我们场景能过。接入 vendor 档（或怀疑 vendor
kernel）时，以下矩阵是最低要求：

**custom all-reduce**（`_ar_worker.py`）：
- 尺寸 8K/64K/256K/1M/8M 元素 × dtype bf16/fp16/fp32；
- eager vs **fp32 精确和**（all_gather 后 fp32 累加，不用 NCCL 当参考——
  它也在被测链路里，只做 harness sanity）；
- graph capture 后 replay 4 次、每次换新输入——检测"烘焙/损坏"两类。

**flash attention**（`vendor/test_flash_attn.py`）：
- (qo,kv) ∈ {(1,1),(1,7),(1,127),(1,1024),(3,129),(17,1023),(127,127),
  (128,128),(129,257),(1023,1023),(1024,1),(2051,2051)}，
  causal 双向 × head_dim {64,128} × dtype {bf16,fp16}——重点奇数长度、
  qo≠kv、2 的幂边界 ±1；
- graph-replay 变输入（decode 形状）。

新后端接入：在 `vendor/test_flash_attn.py` 的 `load_backends()` 注册一个
探针类（包一层把统一入参翻译成该后端 ABI），矩阵自动全量跑一遍。

## 新增算子的最低测试配置

契约测试（维度 1）必带；进 decode 链路的加维度 2（参照
`_graph_safety_probe.py` 模式）；vendor 档再加维度 3 压测 + 快关。
跑法写在文件 docstring，容器外开发时用 `HCU_ADAPTER_ROOT`/`HCU_CKPT`
覆盖探针路径。

## 单测脚手架三段式（新算子测试的起步模板）

照 `tests/kernels/test_gemm_int8.py` 的骨架抄，三段缺一不可：

```python
"""<算子名>（<语义一句话>），GPU contracts.          ← 段0: docstring 带跑法
Reference math: <参照口径，如 fp32 dequant 对照>
Run: pytest tests/kernels/<file>.py -v (HCU node)
"""
import pytest, torch
import sgl_kernel

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires HCU/CUDA device")

# 段1: 形状矩阵 —— 覆盖 decode 最小 M / graph-pad M / 奇数维 / TP 切片
SHAPES = [(1, 512, 512), (17, 1024, 512), (64, 2048, 1024), (1, 128, 128)]
DTYPES = [torch.bfloat16, torch.float16]

# 段2: torch/fp32 参照 —— 数学只写一遍，kernel 调用一次
def _ref(x, w, ...):
    return <torch 或 fp32 解晰实现>

# 段3: 断言 —— 容差按精度类型选型（速查见下方容差选型行）
@pytest.mark.parametrize(... )
def test_xxx(...):
    out = sgl_kernel.<op>(...)
    ref = _ref(...)
    torch.testing.assert_close(out, ref, rtol=..., atol=...)
```

- **顺序纪律**（governance §7.3 维护闭环 / SKILL 法则 9）：算子类单测先于 e2e；seam 类 "one
  pit, one test"（e2e 探明的坑修复时必带离线复现）；
- 容差选型速查：fp32 数学 rel≤1e-3~5e-3；bf16 kernel cos≥0.999 或
  1-ULP 单舍入差；量化路径按既有同类（int8 rel~0.02 / fp8 逐位或 8e-3）；
- 形状矩阵必含：最小 M(=1)、graph pad 桶、奇数(2 的幂 ±1)、TP 切片维。
## 宿主与源码来源

混合门禁的 SGLang 桥探针使用完整 SGLang 依赖及显式
`HCU_ADAPTER_HOST=sglang`；vLLM 的真实 general-plugin 回归在独立进程
选择对应宿主。不得把 SGLang 六参数 RMS 桥装到 vLLM 进程。
从 `ci/`、`tools/` 或 `tests/tools/` 启动源码检查时，启动前把目标 checkout
放入 `PYTHONPATH`，并核对 `hcu_adapter.__file__` 与 catalog 来源。
`HOST=none` 只关闭桥安装，不阻止 sitecustomize 提前导入已安装的 Adapter；
脚本运行后再改 sys.path 可能混用已缓存包。缺桥的 skip 单列，不能算设备通过。
