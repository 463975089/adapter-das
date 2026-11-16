# SGLang Mamba state scatter grid

`hcu_adapter/mamba_state_scatter.py` 在 SGLang 启动阶段绑定原 wrapper 使用的私有
JIT 符号。wrapper 函数对象与已缓存别名保持不变；官方文件不写回。计算实现在
`hcu_adapter/kernels/mamba_state_scatter_triton.py`。

当前只对 gfx936 的 FP32 src/dst、每条状态不少于1024元素、非空且可展开的
三维 grid 使用二维 request/block 展开；其他输入保留原 JIT。架构从输入 tensor
的设备查询，不在安装阶段初始化默认 GPU。上游 wrapper 签名、JIT ABI 和内核
源码摘要必须匹配 v0.5.22 基线，否则保留上游。runtime dump 的条目名为
`mamba_state_scatter`。

source-row 映射、状态槽位、step、所有 bounds checks、层/请求 stride、
输出 entry-contiguous 检查和输入连续性检查均保留。新增路径只改 CTA 到地址块
的映射，不改变元素读写值或计算精度。候选执行后的异常不重试，避免部分写回后
再次执行。

显式回退：启动前设置 `HCU_DISABLE_OPS=mamba_state_scatter` 后重启；
不能在已经捕获 Graph 的运行中撤销。未新增专用环境变量。

验证覆盖：独立 Torch 状态复制参考、完整 dst 与未触及槽位、非 identity
source-row、无效 source/destination/step、尾 tile、外层 stride、变输入 Graph、
实际 SGLang wrapper 别名与模型 MTP 路径。性能结论须以同条件无 profiler
serving A/B 为准，不把单 kernel 时长直接换算为端到端吞吐。
