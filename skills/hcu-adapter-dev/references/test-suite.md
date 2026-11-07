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

CT-FP8 CPU 参考回归见 `test_checkpoint_quantization_reference.py`：
真实小型 Qwen3 safetensors、官方加载器、独立 Linear QDQ 数学对拍、
逐个 tensor 核对及实际逐层 QDQ 观察；缺层 QDQ、错误反量化、
加载缺失、未绑定权重和格式不符必须拒绝。与
`test_l15_checkpoint_reference.py` 一起在 HOSTnone/隐藏 GPU 的干净进程运行，
不得用取消量化或放宽原模型精度阈值来使参考通过。

CT channel-INT4/token-INT8 的设备组件专项为
`tests/kernels/test_w4a8_int8.py`：有效行的整数码/尺度须与独立 CPU
参考精确一致，INT8×INT4 GEMM 的 FP32 输出要求 relative L2 < 2e-5、
cosine > 0.99999。覆盖空专家、NaN padding、capacity1024、半整数和
变输入/counts/尺度 graph 重放。组件尚未接入服务，不代表整模型或
社区 tensor-FP8 路径验收通过。

CT channel-INT4/token-INT8 的层级 oracle 为 `tools.ct_w4a8_reference`，
回归见 `test_ct_w4a8_reference.py`，在 HOSTnone/隐藏 GPU 的独立进程与
官方 compressed-tensors 对拍。CT 0.17.0 的 FP32 对称 INT8 用
`absmax/127.5`，不能套用 SGLang INT8 的 `/127`；参考要求 3D 输入，
避免该库 token reducer 在 2D 退成 tensor 级尺度。这个 oracle 仅覆盖
声明的层级数学，不代表社区 W4AFP8、HCU 实现或整模型精度通过。

该工具保留两种显式参考：`linear` / `expert` 使用反量化后的 FP32
Linear；`int32_linear` / `int32_expert` 先用独立 INT64 点积模拟无溢出的
INT32 累加，再将完整整数和转 FP32，依次乘 token/channel 尺度。
后者拒绝 K > 2097151，测试以官方整数码、FP64 精确整数点积和 Python
整数和交叉验证，并覆盖超过 FP32 精确整数范围的累加结果。两种路径
的微小差异可被后续量化边界放大；整数参考通过不能追认 FP32 参考失败，
也不能代替 BF16 中间存储、routing/combine 或整模型精度门禁。

CT 原 dtype 量化另见 `token_int8_native` 与设备
`w4a8_int8_native_triton.token_int8_native_masked`：尺度、除法结果及 QDQ
按 FP32/BF16/FP16 输入 dtype 舍入，零行使用对应 dtype 的 epsilon。
这不同于始终提升到 FP32 的原接口，不能自动互换。
CPU `test_ct_w4a8_native_reference.py` 与官方 CT 逐码/尺度/QDQ 对拍；
设备 `test_w4a8_int8_native.py` 覆盖非连续输入、奇数维度、空容量、
padding 与变输入/counts graph。显式将已舍入尺度转 FP32 后接整数 GEMM
还须对拍独立整数点积；这不证明 BF16 专家中间存储或服务接线通过。

原 dtype 整数专家 CPU 参考 `int32_native_expert` 的整数点积、缩放、
投影存储边界用官方 CT 整数码和独立 FP64 整数点积交叉检查。
融合激活按 FP32 SiLU×up 计算后一次转回输入 dtype；不能用分别存储
SiLU 和乘积的两次舍入替代。FP32/BF16/FP16 × 两种 clamp 的设备
graph 回归还须改变输入/counts/尺度，并保持原数值门槛。CPU 参考通过
不代表设备专家组件、routing/combine 或模型分布门禁通过。
对应设备组件 `w4a8_int8_native_masked_experts` 的 iteration86 已验证
三种输入 dtype × 两种 clamp 的 eager/变输入 graph 合同；仍须另验
真实 checkpoint、正式 loader 和 DeepEP combine，不能据此转为服务默认。

显式 `HCU_W4A8_MOE=int8` 的接线合同见
`tests/adapter/test_ct_w4a8_int8_moe.py`：保留 regex 匹配到的激活配置，
只复用官方 raw INT4 loader，不走 W4A16 或 stock tensor-FP8。
官方 `QuantizationArgs` 会将未指定的对称零点 dtype 规范化为
`torch.int8`，须接受此元数据默认值，但仍拒绝非对称组和实际零点权重。
多符号注册须先全部成功才发布绑定，失败恢复注册记录并允许重试。
CPU doubles 不代替全新进程的真实 bootstrap、官方加载和 DeepEP LL ABI
设备回归；手工构造 dispatch output 也不代替真实通信或整模型精度。

## 四质量维度

MoE softmax 并列回归见 `tests/kernels/test_moe_gate.py`：CPU 与 HCU
分别覆盖 FP32/BF16、两个公开入口、是否 renormalize、非连续输入、
top-k 边界及内部并列。独立 Python 标量 oracle 按分数降序、专家 ID
升序选取，不能继续以任意 `torch.topk` 结果作金标；设备端另覆盖
同地址变输入 Graph。该规则由社区 power-of-two AOT CUDA 及
Triton router 源码约定，真实 logits 重放还须核对实际调用路径和
完整专家 ID/对应权重，组件稳定不能替代整模型 FP32 分布验收。

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

既有开关也须验证 `hcu_env.resolve` 的值和来源，再经真实 installer
验证路由。DeepGEMM 的 masked BF16/FP8 曾漏登 REGISTRY，
`enabled` 将未知键的 `None` 当作开启，导致显式 0 被忽略；
回归见 `tests/adapter/test_deepgemm_env.py`，默认、配置、显式覆盖分开验证。
该 CPU provider 替身测试只证明接线，不为 vendor 数值背书。

FP8 grouped GEMM 小形状门见 `test_gemm_fp8.py` 的
`test_fp8_grouped_expert_contract`：反量化参考分别展开 per-token-group
与 128×128 block scale；覆盖有符号输入、非单位尺度、strided scale、
空专家/padding 禁写及 64-row aligned contiguous 尾 tile。graph 必须
同时更新输入/权重/尺度与 counts/expert IDs。该直接内核合同不能
代替真实 DeepEP 分布式 dispatch/combine 或模型精度证据。

Mamba2 graph 数值门见 `test_l15_execution_chains.py` 的
`test_mamba2_decode_fresh_input_graph`：独立 CPU FP32 计算投影、卷积、
SSM 与门控归一化，保留激活存储 dtype。捕获后连续三轮改变输入与
映射槽位，参考状态独立递推；同时检查输出、卷积/SSM 状态和其他槽位
不被改写。仅 replay metadata、同输入 replay 或仅检查输出不足以覆盖
状态链路；组件通过也不能外推为 scheduler/speculative/真实权重模型通过。

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

## DeepEP 调用方缩放契约

`tools/ct_w4a8_caller_contract.py --up-tree <community-checkout> --out <external-report>`
在全新 `HCU_ADAPTER_HOST=none` 且 HIP/CUDA/ROCR 可见性均为空的进程中执行。
工具校验上游方法的完整文件字节与 Git commit 一致，提取原方法体，
以显式 CPU gate、TopK 和 expert doubles 检查 routed factor 恰好应用一次。
96 个组合覆盖三种 factor、共享专家、TopK 预缩放、aiter、空输入和 NEXTN。
报告只记录仓库相对源码路径、commit 和内容摘要，不记录部署目录。

这是普通 LL 调用方的源码契约检查，不代表真实宿主、设备内核、
dispatch/combine、checkpoint 或模型精度通过；overlap 路径也不在范围内。
缺失方法、装饰器、可见设备及漏乘/重复缩放具有拒绝回归。

验证基线：社区 commit `baa9befa381511cd4d56f83980951e5a45fffc6d`，
96 个源码组合通过；6 项工具回归、15 项诊断汇报回归及 38 项 CPU
算子共 59 passed，L0 GREEN（34 条既有警告）。原始调试记录保留在仓库外。

CT token-INT8 候选的现代 `CompressedTensorsFusedMoEMethod.apply` 经
`scheme.apply_weights` 调用与旧 `DeepEPMoE.run_moe_core` 相同的 LL core，
执行前要求官方加载后处理已完成。标准 dispatch 及未验证 runner 参数仍拒绝。
真实 scheme/loader 和 LL 数据类型下，新旧入口的 20 项设备检查通过：
两种 clamp × eager/graph 基线、空专家、变 counts/尺度、重新激活。
BF16 输出与独立 CPU 整数参考逐元素一致，topk 元数据对象不变；
67 项 CPU 接缝回归通过，上游源码前后无变化。这里构造 dispatch output，
尚未验证真实通信、模型生成或相对原 FP32 fake-quant 的精度。

LL core 接受社区默认的 `inplace=True` 建议，但仍返回独立输出，保留
dispatch 输入字节。有限 Python routed factor（含 1.5）也可传入；
core 不对原始专家 buffer 再乘系数，原 TopK 权重交给 combine，
未融合的系数由社区调用方应用。非有限值和 tensor/string 系数明确拒绝，
避免设备标量转换引入同步。该契约不扩展到标准 dispatcher 或 overlap。

对应验证：121 项 CPU 回归、96 个 CPU 源码组合、L0 GREEN（34 条既有
警告）；160 项设备检查覆盖两种 inplace、四种 factor、两个入口及
两种 clamp 的 eager/变输入 graph。设备输出再经合成加权归约和原始
社区调用方方法，1,280 个共享专家/TopK 预缩放组合与 CPU 参考逐元素一致。
加权归约和 TopK 仍为显式 doubles，未执行真实通信。padding 含 NaN，
输入未改动需比较底层位模式，不能用普通浮点相等断言。前后上游源码无变化。

候选只复用 WNA16 的加载器，不继承其执行语义：
`get_triton_quant_info` 和 `get_marlin_quant_info` 均明确拒绝标准
Triton/Marlin/LoRA metadata 请求。这些父类描述符没有 token-INT8 激活
QDQ 契约，不能因为字段形状可用就退回 W4A16。两条入口各有继承陷阱回归。
这表示对应执行链尚未接入，不表示硬件无法支持 LoRA。该拒绝修复通过
123 项 CPU 回归及 L0（34 条既有警告），未据此新增设备或模型验收结论。

真实 checkpoint 的专家加载暴露了合成直接拷贝遗漏的回调契约：
社区 `FusedMoE._weight_loader_impl` 按类名白名单对 CT packed 权重和
channel scale 转置。独立候选类未命中时，`[2304, 1]` scale 被写入
`[1, 1]` 切片，模型在加载阶段失败。现在只扩展该类名测试的常量 tuple，
并验证原函数签名及字节码使用位置；保留函数对象、已有别名和原指令，
继续使用原 EP 映射、TP 分片和异步加载，不改写社区文件。
三项注册全部成功后才发布代码及绑定，注册冲突不得留下部分适配。

验证包含：126 项 CPU 回归、L0（34 条既有警告）、真实社区回调的
18 次并行加载、非本地专家忽略，以及旧/新入口 20 项设备 eager/graph 检查。
一个真实专家的 9 个张量先复现原错误，再经原回调和官方转换逐字节核对；
组件前后上游源码无变化。这证明加载接缝修复，尚未证明整模型启动或精度通过。

可重复的真实社区回调回归位于
`tests/adapter/test_ct_w4a8_real_loader.py`。将 Adapter checkout 和
`<community-tree>/python` 放入 `PYTHONPATH`，在完整宿主依赖环境中，
分别以 `HCU_ADAPTER_HOST=sglang HCU_W4A8_MOE=w4a16` 和
`HCU_ADAPTER_HOST=sglang HCU_W4A8_MOE=int8` 启动全新进程，设置
`HCU_TEST_SGLANG_TREE=<community-tree>`，执行
`python -m pytest -q -ra tests/adapter/test_ct_w4a8_real_loader.py`。

每个宿主模式必须实际通过 5 个用例；未选择源码或宿主产生的 skip
表示验证未完成。测试核对实际导入文件与所选 Git HEAD 的字节，使用
真实参数分配、公开加载回调及官方加载后转换，覆盖 TP 两个分片、
EP 非零 rank、非本地专家忽略和并行加载，并先撤销类名适配复现失败。
张量 fixture 位于 CPU，层对象只初始化回调所需字段；不代表真实分布式
模型构造、通信或数值验收。两种模式共 10 项通过，另有 38 项 CPU 算子
回归及 L0 GREEN（34 条既有警告）；前后上游源码无变化。

真实模型在完成权重加载后，曾因一次分配全部专家的 FP32 中间张量而在
decode graph 捕获中 OOM。原 dtype INT8 专家链现在按张量形状将独立
专家分批计算，共享最终 expert-major 输出，保持整数累加、尺度乘法、
投影及融合激活的存储边界；分批决策不读取设备 counts。
临时计算以 64 MiB 为分组目标，单个专家可能超过该目标，
因此它不是模型显存上限或所有配置可启动的保证。

`tests/kernels/test_w4a8_int8_native.py` 的 25 项设备检查通过，包含
三种 dtype、两种 clamp 在完整组/强制单专家组的变输入 graph 回归。
独立 CPU 整数参考和既有精度阈值保持不变；显存回归还要求分批与完整组
输出逐元素一致，且所测样本的增量峰值显存低于完整组的 60%。
另有 63 项 CPU 回归及 L0 GREEN（34 条既有警告），上游源码前后无变化。
这些是组件证据，实际模型仍需重新验证捕获、生成和独立精度。

## CT INT8 的真实双卡 DeepEP 通信回归

在全新 SGLang 宿主进程设置 `HCU_ADAPTER_HOST=sglang`、
`HCU_W4A8_MOE=int8`，明确选择两张空闲设备，将 Adapter checkout 与
`<community-tree>/python` 放入 `PYTHONPATH`，运行：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m tools.ct_w4a8_deepep_probe \
  --up-tree <community-tree> --out <external-report-directory>
```

再以独立报告目录添加 `--swiglu-limit 0.75` 运行一次。
工具使用真实 BF16 DeepEP LL dispatch、已安装的旧版专家 core 和
真实 weighted combine，覆盖 eager、graph 初始、变输入/路由/权重、
全无效路由及重新激活。每个 rank 的输出须与独立 CPU 整数专家加权参考
逐元素一致。报告绑定实际导入社区文件的 Git commit/摘要，检查前后
文件字节，拒绝覆盖既有报告；设备原始日志保留在仓库外。

这是合成权重下的通信及专家执行回归，不包含完整社区 dispatcher
封装、真实 checkpoint 加载、模型调用方缩放或整模型精度。
两种 clamp 配置的双 rank 共 20 个执行相位均逐元素匹配，
127 项 CPU 回归与 L0 GREEN（34 条既有警告）通过；
全部社区源码在专项前后检查无变化或新增。

添加 `--community-dispatcher` 可进一步执行官方
`DeepEPDispatcher.dispatch/combine` 及其状态机，添加 `--recv-hook`
覆盖接收回调，省略时使用 event 等待。此模式另要求
`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8`，避免测试容量与
社区配置不一致。工具通过官方 `publish(ServerArgs(...))` 初始化上下文，
显式选择 BF16；`model_path="dummy"` 仅避免配置解析读取 checkpoint，
不构造模型，专家仍用显式合成权重，不能计为真实模型通过。
社区 facade 的通信 buffer 按其自动析构约定释放，不将默认 buffer
冒充为要求显式 `destroy()` 的独立 buffer。

扩展后的工具已在六个独立双 rank 进程中完成直接通信、社区 event、
社区 recv-hook 各两种 clamp 配置，合计 60 个 rank/执行相位检查，
全部与独立 CPU 整数参考逐元素一致，包含全无效路由后的重新激活。
L0 GREEN（34 条既有警告），全量社区源码前后字节检查无变化或新增；
该证据仍限于合成权重的通信与专家链，不替代模型精度验收。

四种社区 dispatcher 配置已登记到 `tests/manifests/l15_comm_v1.json`：
`ct-int8-deepep-event`、`ct-int8-deepep-event-clamp`、
`ct-int8-deepep-hook`、`ct-int8-deepep-hook-clamp`。
明确暴露两张空闲设备，在仓库根目录执行：

```bash
python -m tools.l15_comm run \
  --manifest tests/manifests/l15_comm_v1.json --root . \
  --up-tree <community-tree> --out-dir <fresh-external-report-directory> \
  --mode blocking \
  --cases ct-int8-deepep-event,ct-int8-deepep-event-clamp,ct-int8-deepep-hook,ct-int8-deepep-hook-clamp
```

runner 为每项启动独立双 rank 进程，固定选择 INT8/BF16 社区 dispatcher，
设置容量 8，并在前后检查全量社区源码；超时按进程组清理本项 worker。
退出码 0 之外，还必须存在两个 rank 各五个不同相位的精确对拍报告，
且 commit、clamp、event/hook 模式匹配。已有 rank 报告目录拒绝复用。
`observe` 沿用观察模式的非阻断语义，准入检查应明确使用 `blocking`；
新增项目仍为候选组件，不计为真实 checkpoint 或模型精度验收。
正式清单入口的四项 blocking 回归已全部通过，共 40 个精确 rank/相位
检查；每项前后及整轮前后源码检查通过。另有 86 项 CPU 回归通过
（通信工具契约与 CPU 算子），L0 GREEN（34 条既有警告）。
这次没有重跑清单中原有的三个 collective 项目。

真实 INT4/token-INT8 checkpoint 的一次四 rank 诊断已完成全部 48 个
权重分片加载、decode graph 的 batch 4/8/12/16 捕获及三次短提示生成。
首都/算术/重复首都请求分别返回 Paris/2/Paris，重复请求的输出 IDs
和所选 logprob 一致；运行前后社区源码及完整 checkpoint 字节核验通过。
这只建立该次配方的短提示启动/生成闭环，长输入、并发、eager/graph
模型级对照和完整独立量化精度仍需单独验证，不追认原 FP32 参考失败。

## 相同 token 历史下的解码概率诊断

`tools.l15_decode_consistency` 读取两个 JSON record，每个包含实际发送的
`input_ids` 和 `/generate` 原始返回 `result`。返回须含 `output_ids`，
以及 `meta_info` 内的 prompt/completion/cached token 计数、所选和
top token logprobs。在隐藏设备的 HOSTnone 进程运行：

```bash
python -m tools.l15_decode_consistency \
  --left <incremental-record.json> --right <prefill-record.json> \
  --out <fresh-external-report.json>
```

只有完整 token 历史相同的位置才对齐；冻结前缀应使用实际 prompt 加
实际已生成 token，单 token prefill 对应增量输出的具体位置。
两个完整 eager/graph 返回也可直接比较，分叉后的不同历史不会继续对拍。
报告保留所选 ID 是否相同、共同 top token 数量及各 token 概率差，
缺失共同 token 时误差为 null，不能解释为零；输入字节绑定 SHA256，
已有输出文件拒绝覆盖。应另行留存实际 seed、模式、缓存策略及完整
源码/checkpoint 前后审计。诊断报告没有 passed，不代替独立精度门禁。

23 项冻结前缀/异常证据回归与 38 项 CPU 算子共 61 passed，L0 GREEN
（34 条既有警告），执行进程未初始化 GPU。工具还分析了一轮真实量化
模型的六对 eager/graph 串行返回：三个短输入的首个预测所报告概率一致，
首个 decode 位置开始偏离；两个长输入在首个预测已有概率偏差。
其中一对长输入的全部输出 ID 相同，仍有共同 top token 概率差。
该轮还完成各模式的四项并发请求，所选样本的并发 ID 与串行一致，
最终全量社区源码及完整 checkpoint 前后字节检查通过。
两种服务使用不同随机种子，且输入为裸前缀；这些观察不能判定根因，
也不是聊天语义或独立量化精度通过。固定种子、原生聊天编码和冻结
前缀的后续对照需以自己的完整审计及结果另行验收。

后续固定 seed=42 的真实量化模型诊断已完成，两种实际模式各执行
八次请求，每次先清缓存并确认 cached_tokens=0。社区原生聊天编码下，
三个短问答在两种模式均返回 Paris、北京、2；三个冻结前缀的单 token
prefill 所报告 top-20 和所选概率在两种模式完全相同。增量 decode
仍有概率偏差；eager 自身在相同历史下也与重新 prefill 不同，并在
一个位置出现继续生成与 EOS 的选择差异。两种模式各自的重复请求
输出一致，最终全量社区源码及完整 checkpoint 前后字节检查通过。
这已排除本轮随机种子不同这一混淆因素，尚未定位具体计算环节，
也未建立完整独立量化精度参考；短问答文本一致不能代替概率验收。

随后通过社区公开 eager hooks 取得前两层的 144 份输入、路由和输出样本。
按实际 TP rank、尾部 padding 和完整 token 历史，对齐三个预测位置，
共六个层/位置对照；gate 输入、router logits、top-k ID/权重及 MLP 输出
均完全相同，gate 输入与 MLP 返回时的输入也一致。八次请求的全部输出 ID
及报告概率与未加观察器的固定种子基线一致，最终全量社区源码和完整
checkpoint 字节审计通过。因此后续定位应扩展到尚未观察的计算环节。

这轮仍保留采样覆盖失败：末层固定模块路径未匹配，不能将整轮报告记为
采样通过，也不能把前两层 MLP 的局部一致外推为全部算子或模型精度正确。
后续采样需覆盖 `_next_layer` 引用导致的规范名称变化，并以实际
`layer_id`、rank 和有效 token 行检查覆盖；完整 prefill 仍不是独立精度金标。

后续通配路径采样已覆盖全部 40 层、四个 rank，共 2,880 份样本；
实际模块身份和有效行对齐后，形成三个预测位置的 120 个层/位置对照。
初始 prefill 位置的全部 40 层采样点一致。首个增量 decode 位置的
gate 输入、logits、权重和 MLP 输出最早在第 2 层不同，top-k ID
最早在第 4 层不同；下一个位置分别最早在第 3 层和第 6 层不同，
层编号从 0 开始。首个 decode 的第 2 层 gate 输入最大绝对差
0.001953125、相对 L2 差约 0.0007412，MLP 输出相对 L2 差约 0.01047。

所有 120 个对照中，gate 输入与 MLP 返回时保留的输入一致；八次请求
全部输出 ID 和所报告概率也与未加观察器基线一致。最终全量社区源码
及完整 checkpoint 前后字节审计通过，原始记录与环境保存在仓库外。
该结果将定位推进到 MLP 之前的边界，没有证明 attention、压缩缓存、
Engram 或 mHC 中哪一项有错，也不能从前层混合向量一致推出原始残差
一致。prefill/decode 可有算法和舍入差异；这仍是归因证据，不是独立
量化精度通过，更不能将其直接写成新内核修复的期望值。

继续采样原始 mHC 状态、混合操作及 attention 调用边界后，前四层、
四个 rank 的 1,152 份样本形成 1,128 个有效行对照，无未覆盖项。
首个增量 decode 位置最早在第 2 层 attention 返回时不同，输入和此前
观察到的边界一致；下一个位置最早在第 3 层 attention 返回时不同。
三个逐步增长的完整前缀间另有 6,016 个历史 token 行对照全部一致。
同节点独立运行的无 hook 对照与观察组，八次请求全部输出 ID 和报告
概率一致。最终完整源码和 checkpoint 字节审计通过后，重新分析的
样本摘要、对照及首差位置与审计前完全相同。该覆盖仍只包含所选边界，
不能据此断言 KV 缓存内容或 attention 内部各步骤一致。

真实第 2 层输入与 checkpoint 压缩权重的独立组件实验中，默认构造
选择 unfused ratio-2 compressor；当前 token 在 decode/prefill 批量
形状下的 KV、score 投影最大绝对差分别约为 2.98e-7、5.07e-7。
将已观测的同一历史 partner 与当前行放入相同的单对 pooling launch，
最终 BF16 latent 完全一致，也与独立 CPU FP64 点积、pooling、
BF16 存储后归一化参考一致。实验前后源码及所加载权重分片字节审计
通过。这仅说明本次组件输入下的投影舍入未改变最终 latent；
默认组件构造不证明实际服务路由，单对 pooling 不覆盖后端批量
pooling、ring 读写、索引器、RoPE 或量化缓存，不能排除它们的问题，
也不能把该局部一致当作模型精度验收。

后续实际服务的第 2 层内部采样确认使用未融合 ratio-2 compressor。
840 份原始帧样本形成 828 个有效行对照；函数样本按所属 attention
进入/返回区间归属，不能用各函数自己的调用序号冒充 forward 序号。
另保留 16 个仅 decode 有的调用、8 个分支特有局部字段及 8 个超出
父调用采样区间的记录，不将它们猜测对齐或算作完整内部覆盖。

四个 rank、三个预测位置的 Q、压缩归一化 BF16 输出，以及写入方法
返回时保留的 latent 均一致。首个 decode 位置，rank 0 的后端输出在
实际使用的 16 个 head 中仅 1/8192 个 BF16 元素不同，最大绝对差
0.000244140625；rank 2 另有两个极小差异，其余两个 rank 后端输出
一致。rank 1 的 `wo_a` 在相同输入下也出现一个极小输出差异，尚未有
独立 GEMM 参考判定。rank 0 后端差异经输出投影后扩展为 294/5120
个元素不同，最大绝对差 0.0078125。比较时须排除其余 48 个 padding
head；下一个位置的大部分后端差异就在未消费的 head 中，不能据此
归因真实模型的后续分叉。

同节点独立无 hook 对照的八次请求，完整输出 ID、所选概率及完整
top-token 集合和报告概率均与观察组一致。最终完整源码及 checkpoint
字节审计通过，审计后重新分析的全部样本摘要、对齐结果和对照与
原报告完全相同。该证据把范围缩到后端 attention 的读取或计算，
尚未直接对照实际缓存 key、有效 mask 与独立 attention 数学参考；
不能据此将浮点舍入认定为实现缺陷，也不能用强制跨模式逐位相同
代替模型精度验收。原始证据继续保存在仓库外。

进一步采样低层 sparse attention 的实际选中 key、有效 mask、Q、sink、
scale、打分与概率后，取得 200 份样本。三个预测位置、四个 rank 的
12 组有效 head 对照中，Q、sink、scale 及保留重复项的有效 key 多重集合
全部一致；后两个位置的 key 排列不同。独立 CPU 80 位十进制点积、
虚拟 sink softmax、P@V 与直接 BF16 舍入，在两侧给出相同参考结果。
这不表示实际输出全部等于参考：两条路径仍各有少量偏差，完整 prefill
也不能作为精度金标。

首个 decode 位置 rank 0 的关键差异元素，精确 attention 结果约为
0.04650879667319083，仅高于 BF16 中点 0.0465087890625 约 7.61e-9。
该 head 按实际 key 对齐后，采样的全部九个注意力概率完全相同。对这些 FP32
概率与 value 做精确乘积累加，结果为
0.046508800121955573558807373046875，应舍入到 0.046630859375。
实际 decode 的 FP32 P@V 累加器为 0.0465087853372097，跨到中点另一侧，
因而存为 0.04638671875。

原形状 HCU P@V 子表达式重放连续三次一致，并复现两侧有效行的全部
16 个 head 的 BF16 输出，以及 decode 保留的 FP32 输出。保持各自形状
不变，只交换有效概率与 value 的对应排列，decode 的关键 FP32 元素变为
0.046508803963661194、舍入到较高 BF16 值；反向交换 prefill 的排列，
则得到原 decode 的较低值。由此将这个关键元素的跨模式差异归因到
P@V 累加顺序引起的 FP32 舍入；未据此修改生产排列或精度阈值。
prefill 原始 FP32 累加器未被采样，其重放的 FP32 值不能冒充原服务观测。
此实验不外推为其他元素、输出投影或整模型的数值验收。

同节点两个独立 eager 服务的八次请求，完整 ID、所选概率及全部报告的
top-token 集合和概率一致；三个原生聊天短请求分别正确回答 Paris、
北京和 2，并正常停止。最终完整社区源码及 checkpoint 前后字节审计
通过，审计后重新计算的全部样本摘要、对照及 12 组高精度结果与初次分析
一致。仓库外保存了可独立重放的选中张量和原形状 P@V 输入，重新从该
归档计算的 24 侧高精度结果也一致。以上只补齐所选位置的归因和三个
最小语义锚点，不代表长输入、graph、完整量化分布或多模型精度通过。

对前述 rank 1 的 `wo_a` 差异另做了 checkpoint 分片参考：按源码的
TP 输出轴分片推导两组 `1024×4096` BF16 权重，相关源码与实际读取
权重分片的前后字节校验通过。同一输入下，decode 的 2048 个元素均与
CPU FP64 点积舍入结果一致，prefill 仅一个元素不同，差为
1.1920928955078125e-7。该元素的独立 100 位十进制乘积和为
0.000016122939996421337127685546875，最近 BF16 值为
1.609325408935547e-5；实际 prefill 为 1.621246337890625e-5。

以推导权重做普通分组 BF16 GEMM 子表达式重放，两侧有效行的全部
2048 个输出均与服务采样一致，连续三次可重复。再将同一输入行重复成
不同批量，batch 8 得到上述较高值，batch 1、4、16 得到较低值，说明
矩阵批量形状足以复现这处舍入差异。参考还未直接采样服务权重对象或
原始输入 stride，因此不将重放一致写成加载后权重逐字节相同的证明。
最初整张量重复性检查因未消费 padding 行的 NaN 失败，修正为有效行
检查后通过，保留该失败记录；不得把未消费输出的 NaN 当作模型故障。
没有为此改变默认 dtype、GEMM 路由或整模型精度阈值。

### Qwen3 BF16 完整精度配置复验

在包含上述输出投影回归的 adapter 提交 `a6540c6` 上，真实
Qwen3-0.6B 使用原样社区 v0.5.22，默认 BF16、单卡 Triton attention，
分别完成 eager 和 graph 的双语短输入、512/2048 token、
缓存复用与变输入三轮请求。两种模式的启动和生成均通过，96 个输出位置
全部与独立 CPU FP32 参考的 top-1 相同；完整社区源码和 checkpoint
前后字节检查通过。这仍不是精度通过。

原 `bf16-cpu-fp32-top20-v1` 策略保持不变，完整精度结果为 **FAIL**：
所选 token 最大 logprob 误差为 0.041772，greedy regret 为 0，
但 top-20 最大 logprob 误差为 0.732379，超过 0.15；
其余五项指标均在原门限内。12 个请求的完整输入、输出及全部报告概率
与此前绑定的失败记录相同，因此该失败仍能在当前代码复现。
配方来源及实际参数差异另存仓库外；这是 Triton attention 精度配置，
不外推为 cookbook 的 FA3、page-size 或服务解析器配置已验收。

另在隐藏 GPU 的独立 CPU 进程中，仅改变 HF BF16 RMSNorm
乘权重与存回 BF16 的先后顺序，未消除短双语概率偏差。
该实验没有模拟 SGLang 的 fused residual 链或其他内核，不能称为
SGLang 等价参考，也不能代替修复或豁免原精度门。
最坏点由概率误差自动 argmax 定位：英文输出索引 5、报告排名 11、
token 16209；中文输出索引 2、报告排名 14、token 46618。
输出索引从零开始、排名从一开始；早期诊断中硬编码的英文索引 6
不是最坏点，原始记录保留，由自动定位结果更正。

### Fused-add RMSNorm 回退的方差存储边界

默认 Triton 路径与已核对的社区 fused-add RMSNorm 默认 JIT 公式，
均以 FP32 输入加法和计算方差，以激活 dtype 存回的残差作归一化分子。
旧 Torch 回退先原地存回残差，再从该舍入值计算方差，两条路径不等价。
修复为先保留 FP32 加法和，分别计算方差与存回残差，保持既有分子和
权重相乘顺序；普通和 Gemma 入口共用修复，不改变默认后端。

`test_fused_add_native_preserves_unrounded_variance` 以相邻 FP16/BF16
存储值之间的加法和配合独立 CPU FP64 参考，修复前四个参数节点均在
输出数值断言失败，存回残差本身正确。该反例能区分方差的两种计算顺序，
不依靠宽松相对误差掩盖差异。`test_fused_add_variance_fresh_input_graph`
另覆盖 Torch/Triton、普通/Gemma、FP16/BF16 八个参数节点，
每次捕获后连续三轮改变输入、残差与权重，同时检查输出和原地残差。
该问题尚未归因为 Qwen 默认配置的整模型精度失败；修复回退契约不能
据此将先前的 Qwen 精度 FAIL 改为 PASS。

修复后上述四个 CPU 反例通过；组合 CPU 回归 76 项通过且同进程未初始化
GPU，归一化文件另有 25 项通过（4 个 opt-in vendor parity 节点未执行）。
严格 Dense 家族 4/4 通过，包含八个新 graph 参数节点，零 skip/xfail、
收集错误或超时；前后完整社区源码字节检查通过，L0 GREEN
（34 条既有警告）。此处不将 opt-in vendor 数值行为记为已验收。

### Qwen2 BF16 完整精度配置复验

同一 `a6540c6` 运行组合下，真实 Qwen2-0.5B-Instruct 的 eager、graph
均完成双语、512/2048 token、缓存复用和变输入三轮请求；源码及
checkpoint 前后字节审计通过。96 个位置中 94 个匹配独立 CPU FP32
参考 top-1，最大 greedy regret 为 0.008249，所选 token 最大
logprob 误差为 0.070095。但所有固定输入类别的 top-20 最大 logprob
误差均超过原 0.15 门限，整轮最大值为 0.546772；
其余五项策略指标在门限内，原 V1 精度裁决仍为 **FAIL**。
该小模型是固定精度探针，所参考的 cookbook 固定快照没有对应的
Qwen2-0.5B-Instruct 文本启动配方，不宣称完成了 cookbook 等价配置验收。

另对 Qwen3 做了 CPU BF16 残差流诊断：保留独立加载的 HF attention、
Linear、RoPE，逐层传递贡献与 BF16 残差，在归一化时先计算 FP32 加法和。
分别使用未舍入或存回 BF16 的和作分子，乘权重后再存回 BF16。
小型 FP32 模型的流改写对照逐位一致，真实 BF16 模型的原始及恢复后
对照均复现此前缓存；源码、权重和实例方法恢复检查通过。但两种约定
仍未消除短双语概率偏差。这只是全前缀 CPU 流诊断，不能称为覆盖
增量 KV、所有 GPU 内核或 SGLang 实际调度的等价参考，也不替代原 V1 门。

### 修复后完整回归与真实归一化输入对拍

提交 `a929a1a` 的五家族 L1.5 矩阵一次完整运行 **48/48 PASS**，
包含 86 个参数节点和 58 个 subTest，所有 setup/call/teardown 通过，
零 skip/xfail、收集错误或超时；前后全量社区源码字节检查通过。
计数与 manifest 绑定见 `docs/l15-family-matrix.md`，不外推为完整
adapter 测试套件通过。

同一修复提交上的 Qwen3-0.6B、Qwen2-0.5B-Instruct 均完成原精度配置
的 eager/graph 复验。两模型每个请求的输出 token、所选及全部报告的
top-20 概率与修复前逐位一致，top-20 最大 logprob 误差仍分别为
0.732379 和 0.546772，原 V1 精度裁决仍为 **FAIL**。

对真实 Qwen3 的 28 层 input/post-attention RMSNorm 与最终 RMSNorm，
通过社区公开 forward hooks 采集 57 个模块、每模块 8 次调用，共
456 份 eager 样本（prefill `[47,1024]`、decode `[2,1024]`）。
必须在原地写回前克隆输入；仅在 post-hook 读取 args 会丢失原始残差。
同节点未观察对照的六个请求，其输出 token、所选及 top-20 概率
与观察后逐位一致；前后社区源码和权重字节检查通过。

隐藏 GPU 的独立 CPU 进程以 FP64 计算平方归约、逆平方根和乘权重，
fused 方差使用未舍入 FP32 加法和，分子使用存回 BF16 的残差，
最后转 BF16。3560448 个输出元素中 38 个与参考不同，均为相邻
BF16 值；其未舍入参考距两值中点至多为间距的 `2.775e-5`。
448 份 fused 样本的存回残差全部逐位一致，所有输出有限，
最差相对 L2 为 `1.826e-4`，最小 cosine 为 `0.999999983`。
这些样本未发现超出相邻值舍入差异的异常，不据此归因整模型概率偏差，
也不替代 graph、长输入或其他模型的精度门。

本次只覆盖普通及残差 RMSNorm。社区 `apply_qk_norm` 可直接调用
`fused_inplace_qknorm`，绕过 q_norm/k_norm 模块，因此上述 hooks
不能证明融合 QK Norm 已被观察；后续须在实际函数边界或独立算子合同
另行验证。原始样本清单、摘要与关键操作数均保存在仓库外。

融合 QK Norm 的组件回归见 `tests/kernels/test_qknorm.py`：
直接调用社区 JIT 入口，不以两次普通 RMSNorm 代替。FP16/BF16
分别覆盖 128 维 head 的 prefill/decode 形状及 512 维 head 分支；
Q/K 使用带 token stride、非零 storage offset 的 packed QKV 视图。
独立 CPU FP64 参考逐 head 计算，输出限制在参考的相邻存储值内，
并限制相对 L2；真实 graph 一次捕获后连续三轮更换输入及 Q/K 权重。
要求 eager/graph 逐位一致，V、前后保护区及权重不得改写。

六个参数节点通过，接入后的正式 Dense 家族 5/5、17 个参数节点
全部 setup/call/teardown 通过，无 skip/xfail、收集错误或超时；
另有 70 项 CPU 回归、L0 GREEN（34 条既有警告）及前后源码检查通过。
当时 manifest 为 49 个逻辑 case，该轮只验 Dense 家族，不宣称
完整 49 项整轮通过，也不据此更改任何整模型精度裁决。

随后对真实 Qwen3 的 28 层 attention，以公开模块 hook 在未启用采样
的 warmup 中安装当前线程 Python 函数边界观察器，按原始
`fused_inplace_qknorm` 的 code object 保存写回前的 Q/K 和返回后的值。
不替换函数或改写社区文件；28 层各 8 次调用，共 224 份样本，
实际调用栈包含 `apply_qk_norm`，Q/K 的 token stride 均为 4096。
采样前后六个请求的输出 token、所选及全部报告的 top-20 概率逐位一致，
前后源码与权重检查通过。

隐藏 GPU 的 CPU FP64 逐 head 参考对拍共 5246976 个 Q/K 输出元素，
26 个与最后存回 BF16 的参考不同，均为相邻 BF16 值；
未舍入参考距中点至多为间距的 `2.563e-5`。全部输出有限，
最差相对 L2 为 `1.446e-4`，最小 cosine 为 `0.999999989`。
本次短双语 prefill/decode 样本未发现超出相邻值舍入差异的异常；
不外推为 graph、长输入、RoPE、attention 或整个模型精度通过。
全部 224 份操作数、摘要、控制输出和审计报告归档在仓库外。
此前一次观察器导入路径错误发生在采样前，保留失败证据，
不计为模型数值失败，也不计入这份有效样本。

### 真实 RoPE 的相消误差与原始 JIT graph 回归

真实 Qwen3 的 28 层 RoPE 共采集 224 份 eager 样本，同时保存旋转前后
Q/K、位置和实际使用的 FP32 cos/sin 行；共享 rotary 模块按实际
attention 调用帧归属层，不能按 rotary 对象个数推测层覆盖。
旋转前的全部 Q/K 与前一轮融合 QK Norm 输出逐位一致；六个请求的
输出 token、所选及全部报告的 top-20 概率与未观察对照逐位一致，
源码及权重前后审计通过。

独立 CPU FP64 对拍 5246976 个旋转输出元素，29 个与最终存回 BF16
的参考不同，其中一个相消结果超出相邻 BF16 值；最差相对 L2
`2.349e-5`、最小 cosine `0.999999999724`。
按原公式分别计算独立 FP32 乘积与两种 FMA 收缩顺序，每个实际值
都可由至少一种顺序复现，也均落在按乘积绝对值和构造的 FP32
舍入区间内。这不证明某一种编译指令顺序适用于全部元素。
本次位置 0～30 的实际缓存与独立 FP64 三角函数最大差为 `1.490e-6`，
不能外推其他位置或长输入。没有据此修改内核、dtype 或原模型精度门。

`tests/kernels/test_original_rope.py` 将上述相消边界纳入原始社区 JIT
合同，覆盖 NeoX/GPT-J、FP16/BF16、2/47 token 的 packed QKV 视图。
逐元素使用 FP64 数学与 FP32 乘加舍入区间，最终存储后再比较；
单次真实 graph 后三轮改变输入、位置和缓存，检查 V、保护区、
缓存与位置不被改写。新增八个参数节点通过，正式 Dense 家族
6/6、25 个参数节点全部 setup/call/teardown 通过，零 skip/xfail、
收集错误或超时；70 项 CPU 回归、L0 GREEN（34 条既有警告）、
manifest 校验和前后全量源码检查通过。清单增至 50 项，但本轮
不宣称五家族完整 50 项整轮通过，也不豁免 Qwen 原 V1 精度失败。

### 真实 Dense attention 的 KV 与独立参考诊断

真实 Qwen3 默认 BF16、单卡 Triton attention 的 28 层各采集一次
prefill 和七次 decode，共 224 份样本、448 组请求/层输出。
在 `RadixAttention` 写入前克隆 Q/K/V，同时绑定实际请求长度、位置、
token、KV 索引和写入位置，返回后读回实际 KV 池的对应行。
全部 Q/K 与前一轮 RoPE 输出逐位一致，六个请求的输出 token、
所选及全部报告的 top-20 概率与未观察对照逐位一致；
社区源码及权重前后审计通过。

独立 CPU 缓存按捕获的写入位置递推；1708 行写入和 10780 行按实际
decode 索引读取的 K/V 全部逐位一致。两个请求的共同前缀发生
504 次物理槽位替换，均逐项验证完整 token 前缀和 K/V 值相同。
最初要求物理槽位恒等的分析守卫因此拒绝，保留该失败记录，
由显式验证前缀与数值等价的分析取代，不能归因为模型缓存串用。

以捕获的 BF16 Q/K/V 和验证过的语义历史独立计算 CPU FP64 causal
GQA attention，共比较 3497984 个输出元素，全部有限。
prefill/decode 的请求级最大相对 L2 分别为 `0.001936`/`0.002496`，
单 head 最大相对 L2 为 `0.004461`/`0.004059`，
最小 cosine 为 `0.999992411`/`0.999995207`。
该参考未模拟 Triton 的概率中间存储及归约顺序，结果用于继续定位，
不输出 attention 与社区逐位等价或整模型精度通过。
原 V1 的概率分布失败、默认 dtype 和阈值保持不变，
本次结果也不外推 graph、长输入或其他模型。
全部样本、实际缓存读回、索引、控制输出与审计保存在仓库外。

继续使用相同 224 份输入和缓存验证，在隐藏 GPU 的 CPU 进程比较
FP32 logits、exp/exp2 及概率存储边界。社区默认 prefill 和 grouped
decode 实现包含 `tl.dot(p.to(v.dtype), v)`：未归一化概率先存为
V 的 dtype，分母仍来自未舍入概率的求和。本次短输入适用单块候选；
这不是对长输入 online softmax 或跨 split 归约的等价实现。

共 3497984 个最终 BF16 输出，FP64 数学参考存回 BF16 后，
prefill/decode 分别有 631778/193894 个值不同。使用 CPU FP32 logits
和 exp，但 P 保持 FP32，分别仍有 631774/193889 个不同；
仅将未归一化 P 存为 BF16 后再做 PV，降为 746/225 个不同，
请求级最大相对 L2 分别为 `0.000403`/`0.001561`。
exp2 的同类候选仍有 1068/252 个不同。全部候选使用同一实际输入，
没有修改生产代码、默认 dtype 或精度阈值。

该结果支持继续检查社区中间舍入约定，但没有直接捕获实际 QK dot、
P 或机器指令；剩余 971 个值及最终模型概率偏差尚未闭环。
不能据此宣称 attention 精确等价、修改生产 P 精度或豁免原 V1 失败。
完整分析脚本、结果及对应社区源码保存在仓库外，并用哈希绑定前一份
含全部操作数的归档；CPU 诊断未初始化 CUDA。

进一步对同一输入逐项替换 CPU 参考的计算精度：QK 用 FP64 计算后
存回 FP32、exp 用 FP64 后存回 FP32、PV 用 FP64、分母用 FP64。
QK 候选分别匹配原 prefill/decode 差异中的 511/125 个，
同时新增 405/79 个差异，总数为 640/179；不能只报告减少的数量。
exp 候选未改变任何最终值，PV 和分母候选也未消除全部差异。
这将后续检查重点收敛到 QK 累加及其对概率舍入边界的影响，
但不证明实际 GPU 的点积顺序，也不能拼合候选为验收参考。
所有候选仍是诊断，原模型精度门及失败结论保持不变。

### 真实权重 CPU attention 存储边界诊断

在隐藏 GPU 的独立进程加载同一绑定字节的 Qwen3 BF16 checkpoint，
固定短中英文 prompt 与 HCU 已生成的 token 序列，CPU 全前缀 eager
重放概率。HCU 报告中的 eager/graph 请求各两份，但只有两条不同
token 历史；这里没有执行 CPU graph，也没有运行新的 HCU 请求。

先通过实例私有 forward globals 路由到原 HF eager attention，验证
小型 FP32 模型、真实 BF16 模型和恢复原方法后的 logits/概率逐位一致。
实验不改写 HF 或社区文件，checkpoint 与相关源码前后哈希相同。
随后用共同的 FP32 QK/exp 比较两种概率存储边界：归一化 P 存 BF16
后 PV，或未归一化 P 存 BF16 后 FP32 PV、再除未舍入分母并存 BF16。
另加入此前验证过的融合残差/Norm 约定作组合诊断。

相对 HCU 短中英文输出，原 HF BF16 的最大 top-20 logprob 差分别为
`0.915293`/`0.625000`；归一化 P 候选为 `0.547446`/`0.812500`，
未归一化 P 候选为 `0.906373`/`0.374788`，组合候选为
`0.549255`/`0.500000`。效果随请求和配置改变，没有消除分布差异；
不能将单一中间存储约定认定为整个模型失败的充分解释。
这些是 CPU BF16 诊断候选与 HCU 的差异，不能混同原 CPU FP32 V1
验收指标；其他 CPU Linear、RoPE 及全前缀执行仍不等价于 HCU 增量执行。
全部候选概率、实际调用覆盖、恢复对照和源码/权重审计归档在仓库外。

继续在同一真实 checkpoint、相同短请求和恢复对照下，分别检查
RoPE 的 FP32 乘加、CPU 生成的 FP32 cos/sin 缓存以及融合 SiLU×up
仅在最终输出存 BF16 的约定。HF 原实现会先将缓存及部分中间值
存回输入 dtype，不能将其 BF16 执行直接视为社区融合路径的参考。
实验只替换实例私有 forward，未改写社区或 HF 文件。

七个候选各覆盖 28 层、两条独特历史；原样转发候选与原概率逐位一致，
既有融合残差/attention 候选与上一轮缓存逐位一致，恢复后也逐位一致。
RoPE 仅改 FP32 乘加时，中英文最大 top-20 logprob 差为
`0.500000`/`0.557202`；加上 CPU FP32 缓存为
`0.977117`/`0.504205`；仅融合 SiLU 为 `0.573725`/`0.991879`。
组合所测 Norm、attention、RoPE 和 SiLU 存储约定后仍为
`0.487216`/`0.553995`，未消除最终分布差异。
CPU 的三角函数、Linear 及全前缀执行仍不等价于实际 HCU 增量执行，
不能凭这些候选修改生产默认路径或将原 V1 精度失败改为通过。
36 组请求比较、全部候选概率、调用覆盖及源码/权重审计保存在仓库外；
GPU 在此诊断进程中保持隐藏且未初始化。

### 原始融合 SiLU 的存储与变输入 graph 回归

`tests/kernels/test_original_silu.py` 直接调用社区原始 JIT
`silu_and_mul`，覆盖 FP16/BF16、2/47 token、3072 维 gate/up、
非零 storage offset 和前后保护区。独立 CPU FP64 参考在最终
dtype 存储前应用 FP32 运算误差区间，同时保留相对 L2 检查；
每轮输入还验证提前将 SiLU 中间值存回输入 dtype 的候选会被拒绝。
单次真实 CUDAGraph 捕获后三轮改变输入，eager/replay 逐位一致，
输入及全部保护区未被修改。

四个新增参数节点通过，正式 Dense 家族 7/7、29 个参数节点及
18 个 subtest 全部通过，setup/call/teardown 完整，零
skip/xfail、收集错误或超时；70 项隐藏 GPU 的 CPU 回归及
L0 GREEN（34 条既有警告）通过，前后全量社区源码检查通过。
清单增至 51 项；本节仅证明 Dense 家族及原始融合激活契约，
不代替其他家族、完整 MLP 或真实模型精度验收。

### 新增 QK Norm、RoPE 后的五家族完整回归

固定已提交版本 `f4e403d` 和当时的 50 项清单，执行一次完整
Dense/MLA/MoE/Quantization/Mamba 矩阵：分别为 6/17/10/10/7，
合计 50/50 通过。实际收集 100 个参数节点及 58 个 subtest，
每个节点 setup/call/teardown 均通过，零 skip/xfail、收集错误或超时。
70 项隐藏 GPU 的 CPU 回归、L0 GREEN（34 条既有警告）以及
社区 10594 个跟踪文件的前后完整字节检查均通过。

本轮执行时尚未纳入随后新增的原始 SiLU 项；当前 51 项清单中的
该项已另经 Dense 7/7 验证，不能把两个报告拼成一次完整 51 项运行。
完整报告、逐项相位及 subtest、控制器、资源与源码审计均归档在仓库外。
家族矩阵仍不替代真实 checkpoint 的功能和原 V1 精度验收。

### 真实模型 TP 参数与逐设备预检

`tools.l15_model_smoke --tp-size` 将 TP 数量绑定到控制器、worker、
Engine、实际 server info 和 scheduler 状态。默认 TP1，预检严格要求
可见设备数匹配，并在每张设备分配、计算及同步，记录独立设备条目。
参考入口检查报告、worker 与实际配置一致，以 `execution_tp_size`
记录受测配置；旧缺失字段只按 TP1 解释。使用方式见
`docs/l15-model-smoke.md`。

261 项隐藏 GPU 的 CPU 回归通过，包含 TP 配置丢失、被忽略、布尔值、
scheduler 不一致、预检缺少设备及分配失败拒绝；L0 GREEN
（34 条既有警告）。真实 Qwen3-0.6B 默认 BF16 的 TP2 accuracy
profile 完成 eager/graph 启动及三批双请求生成，包含短中英文、
512/2048 token、缓存复用与变输入。六组请求的 eager/graph 输出
token、所选及报告的全部 top-20 概率逐位一致；逐设备预检、
社区源码和 checkpoint 的前后完整字节检查通过。

独立 CPU FP32 参考比较 96 个位置，top1 96/96，所选 logprob
最大差 `0.027206`，top-20 最大差 `0.604497`，
top20+tail JSD 最大 `0.000502`。原 V1 仍为 FAIL，
不能用启动、模式间一致或 top1 相同豁免分布失败。
另外以新入口省略 TP 参数执行默认 TP1 的完整 eager/graph，
12 组请求的输入/输出 ID、所选及全部报告的 top-20 概率与原单卡
未观察对照逐位一致，源码及权重检查通过。

配方参考 cookbook commit `40adc4d30ec2640e3391cfd78759bd3598e95f71`
的 `docs/model-deployment/sglang/qwen3.md` 中同模型单卡条目。
TP2 是本轮显式扩展；同时记录社区 0.5.22、Engine API、Triton
attention、page1、mem fraction 0.25、accuracy context4096/
max total tokens16384、默认 warmup/overlap 和 eager/graph 等差异。
未执行原配方的 HTTP、工具/推理解析器和系统设置。
完整日志、各 rank 初始化记录、模式比较、默认配置对照、参数与
源码/权重审计保存在仓库外。此轮不外推 TP4/TP8、EP 或其他模型。

### Conv1D decode 测试必须注册为 pytest 用例

`test_conv1d_update_emu.py` 原先在模块顶层分配 GPU、修改环境和
`sys.path`、执行断言，却没有 pytest 测试函数。现改为显式 CPU/CUDA
两项参数化用例；导入和收集不得初始化 CUDA，或修改 Torch RNG、
环境和导入路径。独立 CPU FP64 标量卷积参考检查 BF16 输出，
保持 cosine ≥ 0.9999，并增加最大绝对误差 ≤ 0.02 的实际断言。
使用非连续、重排的缓存槽位，逐元素核对更新后的完整缓存，覆盖
未选中槽位禁写，并检查输入、权重、bias 和槽位索引未被改写。

远端隐藏 GPU 的收集检查通过，两项均被发现；CPU 单项通过且 CUDA
未初始化。独立空闲 HCU 上运行整个文件 **2/2 PASS**，无 skip、
xfail 或错误；L0 GREEN（31 条已有警告）。此项只验证 Torch
emulator 的 eager 输出与状态，不包含 graph、fork/vendor 档或真实模型。
L1.5 manifest 未变，仍为 51 个逻辑 case。原始证据保存在仓库外。

### FA3 paged 路由须恢复 packed 输出布局及 out 缓冲区

三维 packed Q 仅为调用 Triton 临时转成 `[B,L,H,D]`；decode 和
uniform wide-window 返回时须恢复 `[B*L,H,D]` 输出及 `[B*L,H]` FP32
LSE。显式四维调用保留原 Triton 布局，底层 kernel 接口不变。
decode 的 `out=` 必须写入并返回原对象；不能忽略缓冲区或以广播
掩盖形状错误。数值对拍须先断言形状，再计算误差。

新增 10 项契约在修复前复现 6 FAIL / 4 PASS，覆盖单 token、
三 token、packed/四维、带/不带 LSE 及非连续 out。修复后整个
FA3 文件 **25 PASS**，包含既有四项失败和变 Q/页表/长度的三次
Graph replay；输出与 LSE 均检查独立 dense CPU 参考。CP 探针
三个上下文拆分场景通过；隐藏 GPU 的 CPU 算子 38 PASS、L0 GREEN
（31 条既有警告）。社区 10594 个跟踪文件及额外源码的前后完整性
检查通过，候选源码字节不变。证据为仓库外
`external-evidence:matrix257-full261-fa264-complete-raw.json.gz`，
SHA256 `47707911280677faf2fae4ce97f8ef64d39038f1078e4c8530a0eedd2f87622a`。
本项不代替修复后的全量 adapter 或真实模型精度验收。

### Prefill BCG 的注册与函数发布必须一致

`prefill_bcg_gate` 原先遗漏 `cuda_graph` catalog 条目，并在登记前
替换两个宿主命名空间，导致登记报错时留下未报告的有效绑定。
现补 catalog/UI/档位矩阵，先登记再发布函数。离线回归覆盖
draft/force 两个开关、其他 backend 原样委托、非 HIP 无操作、
重复安装和登记失败时两个命名空间均不变。新增 8 项在修复前
7 FAIL / 1 PASS；修复后连同 registry/bootstrap 和 CPU 算子共
82 PASS，隐藏 GPU 且 CUDA 未初始化；L0 GREEN（31 条既有警告）。
真实社区宿主启动另确认两处别名和 registry 一致、重复安装不变，
社区源码前后完整性检查通过。该启动探针未执行模型或 capture，
不作为硬件 BCG 支持或模型资格证据。
原始证据 `external-evidence:bcg265-complete-raw.json.gz`，
SHA256 `0c02ce30d66b4cc28c257b7ae31f3d906fa65f96a7ab70009127b4730a1ca3b3`。

### L1.5 增补 FA3 路由覆盖

清单增加 `dense-fa3-paged-output-layout`、
`dense-fa3-packed-output-buffer`、`dense-fa3-packed-replay` 三项，
引用上述已独立通过的真实路由测试，而不是仅调用底层 Triton kernel。
清单当前共 54 个逻辑 case，Dense 为 10 个；新增三项展开为
8+2+1 个参数节点。新增契约已包含在 FA3 25 PASS 中，清单与 CPU
算子组合 70 PASS、L0 GREEN。合并 FA3、Conv1D 和 prefill 登记修复
的不可变快照正式 Dense **10/10 PASS**，实际 40 个参数节点、
18 个 subtest，所有相位通过，无 skip/xfail、收集错误或超时。
社区 10594 个跟踪文件及额外源码前后审计通过，603 个候选文件
摘要全部保持不变。首次不带 Git 元数据的快照在执行用例前被
编排器拒绝，原失败记录完整保留；只在另一份带 Git 元数据的
快照补跑，没有修改并行全量回归使用的目录。
证据 `external-evidence:dense266-268-complete-raw.json.gz`，
SHA256 `71cbc0c5e9e521a595988a3c4bd8463e29fd4574dd7362cd0a871bb52a8d9d8c`。
全量 adapter 回归另行执行；旧完整 51 项与本次 Dense 10 项不拼为
新完整 54 项通过。
