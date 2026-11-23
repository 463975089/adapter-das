# Small-copy 动态批次兼容

## 用途与范围

恢复 SGLang 上游 batched buffer copy 在 HCU 上的逐元素拷贝语义。
当前验证对象为 SGLang `baa9befa381511cd4d56f83980951e5a45fffc6d`，
DTK Triton 的 `triton.backends.hcu.jit.FastJITFunction`。
该问题涉及 eager 和 decode graph 的输入缓冲区，不是模型专属 NEXTN 算法。

## 首次分叉与实现

HYV4 NEXTN 两条真实请求在 padding 后 `seq_lens=[104,104]`，
上游 eager registry 执行 grouped copy 后变为 `[104,0]`；
input IDs、positions、request indices 和 cache locations 同时只剩首元素。
逐字段拷贝前后记录证实 copy 本身已失真，尚未进入第 0 层 attention。
独立五字段拷贝在 batch 1→2→16→1 时复现漏写、旧指针/错误值，
不需要模型、CP 或 FlashMLA。

HCU FastJIT 的 saved-kernel key 在此路径只保留 BIT_WIDTHS/BLOCK，
不能正确区分这些 tuple 参数的实际 launch ABI/specialization。
`hcu_adapter.features.jit.small_copy_jit` 仅给这一上游 kernel 创建标准
`triton.runtime.jit.JITFunction`，保留原始函数与 decorator 参数，
把原 kernel 对象的 run 绑定到标准 JIT；已有对象别名和公共
try_small_copy 入口保持有效，其他 JIT 对象不变。无源码写回、无全局
FastJIT 关闭、无 CPU 同步、无模型采样流程改写。

## 检查与状态

- `HCU_ADAPTER_HOST=none python -m pytest tests/adapter/test_small_copy_jit.py -q`
  验证对象别名、原参数、幂等及非 HCU FastJIT 不受影响。
- `TRITON_FAST_JIT=1 HCU_ADAPTER_HOST=sglang python -m pytest tests/kernels/test_small_copy_jit.py -q`
  在真实 HCU bootstrap 下验证变 batch、变地址、空输入、INT32/INT64
  转换、二维非连续源/目标、目标哨兵与变输入 graph replay，逐元素零容差。
  先核对实际对象仍为 HCU `FastJITFunction`、其 `run` 已绑定标准 JIT，
  且 registry 登记可见。标准 JIT 默认路径通过不能作为 FastJIT 修复通过。
- 真实 HYV4 原配置 graph 生成、并发与 GSM8K 结果另行记录；组件通过
  不等于整模型精度通过。

当前 catalog 为 candidate。临时诊断注入不属于生产实现。

## DTK JIT 构造选项兼容

不同 DTK 构建不一定提供 `_init_kwargs`。存在时保留其完整选项；不存在时，
从原 JIT 对象读取 `version`、两类 `do_not_specialize`、`debug`、
`noinline`、`_repr` 与 `launch_metadata`，传给官方标准 `JITFunction`。
不能遗漏 alignment 选项或用默认值覆盖显式 False/回调；字段缺失时在登记和
绑定发布之前拒绝。保留原函数、kernel 对象和已有别名，不改官方数学实现。
CPU 故障回归与真实安装态 FastJIT/Graph 验证分别记录，原模型结果不转授新构建。


### 2026-11-19 原配置整机复验结果

修复后原TP8/EP8/CP8、NEXTN steps3/topk1/draft4、graph开启：
单/双请求各两轮、n=2与六条不同内容并发均正常。
GSM8K前200、并发16、temperature0、no_think、max_tokens4096：
195/200（97.5%），219.75秒；200/200 stop与boxed，0截断/缺失预测。
5道错题完整输出已人工检查，为连贯解题/题意解释错误；本轮未再观察到原乱码。
原始响应、finish reason和运行日志保留在部署任务的
nextn-small-copy-fixed-graph-20261119-105758证据目录。
首次启动HIPBLAS分配失败不算通过，以上结果来自成功重试；
修复后未单独重启全模型eager配方，组件覆盖eager与变输入graph。
不是同工作区无NEXTN单变量A/B，不能据此证明全部模型数值等价。
图文导览此次未做浏览器视觉复验。代码保持未提交，原始脚本与official不改。
