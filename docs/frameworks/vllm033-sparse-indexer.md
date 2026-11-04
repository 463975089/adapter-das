# vLLM 0.33 C4 sparse-indexer compatibility candidate

官方 `SparseAttnIndexer.forward_hip` 在真实 gfx936 拒绝入口，原始 Torch
decode fallback 又存在 capture 时长度读回与 C4 block-flat cache 布局不符。
Adapter 通过公开 `CustomOp.register_oot` 补齐这个既有功能的 HCU 接缝。
平台身份保持真实；不写回 Core，不修改 helper globals，不复制计算内核。

计算沿用官方 public gather、prefill logits、native paged logits 和 C++ top-k。
仅 C4、H64/D128、FP8 query、FP32 weights、top512、page64/132-byte cache，
由官方原始 compressor 写 cache。PCP/DCP、candidate masking 与 cache insertion
未覆盖时在写输出前拒绝；其他压缩比/FP4 交给原始 dispatch。
TP 不拆分这64个 indexer heads；不能把测试头数除以 TP。

`requires_padding` 使用公开 `PackSeqTritonKernel`/`UnpackSeqTritonKernel`
并传入 host metadata 的 token count 与 write_max_decode_len。
query 按原始 FP8 字节搬运，weights 同步打包，top-k 再解包；不使用便利
wrapper 内的 GPU `.max().item()`/`.sum().item()`。保留官方 builder 对
不等长有效请求的 decode/prefill 划分，以及较长 MTP 的 flattening。

默认 stock，不发布 OOT；配置 `ops.vllm_sparse_indexer: native` 启动时 opt-in。
调试覆盖 `HCU_VLLM_SPARSE_INDEXER_IMPL=native`，快关
`HCU_DISABLE_OPS=vllm_sparse_indexer` 优先恢复 stock。
快关是新进程诊断回退，不证明官方已知失败路径通过。注册后改变策略、
public selector 或编译 factor 所有权漂移均锁存失败；不在健康查询中修复。
子进程须先激活；fork 不重复注册，失败状态不因 fork 或坏异常格式化清除。
桥与官方 Python provider/metadata/workspace 内容、Torch/HIP 和 vLLM 版本
进入官方 `VLLM_` 编译因子，供普通/AOT 身份计算使用；两层实际缓存复用仍待验。

## 验证状态

最终离线桥契约32项通过；宿主/共享 lifecycle/bootstrap/CPU 加真实 metadata
discovery 回归合计430个去重用例通过，无跳过。完整 discovery17项需顺序
启动多个独立进程；外层180秒不足，保留原超时后按实际工作量给预算，最终
17项在432秒通过。不能把总预算超时直接定性为某个测试死锁。

最终独立 wheel 的真实 general-plugin 注册、fullgraph/dynamic compile 与
Graph 检查覆盖12个非空场景、36状态/249行参考；包括普通、prefill/mixed、
padding holes、MTP1、官方 MTP5 flattening，以及固定地址变 padding 位置和
因果长度。另有1个空场景3状态，无数值工作且不写输出。所选集合在非空
状态随输入改变，并与未编译执行/Graph 外执行一致；不要求 top-k 顺序一致。
预绑定前使用官方类真实的空 cache placeholder 通过 profiling 检查。
custom op/fake 原位写输出后返回 None，调用方返回输出 buffer，避免新 op
返回输入别名触发 Torch2.11 弃用警告。

安装态 CPU spawn/fork、普通/AOT hash factor、env-cache 与 selector drift
拒绝通过；这个 CPU 探针显式使用硬件准入替身，不作为设备计算或真实引擎
证据。全部205 Adapter/2991 Core Python 文件匹配 source/wheel/install 字节。
节点完整命令出现两个失败标记后600秒超时，没有完成 JUnit，不算通过；
triage 单独保留结果。两层实际缓存复用、完整节点、四质量维度与
cache compressor/write、V4 engine/HumanEval32 尚未闭环。
catalog 保持 candidate；不提升 last-good，不把组件结果当作完整模型验收。

设备探针：`tests/tools/_probe_vllm_sparse_indexer.py`，运行前提见文件头。
它使用真实官方 ModelConfig/metadata builder；独立 CPU FP64 检查 decode，
prefill 保留官方 BF16 dot 舍入边界，另存 pure FP64 诊断。top-k 顺序可能
非确定，检查集合/所选值，不能据顺序变化单独判定 Graph 错误。
保持原始 .002/.002 selection gate，不使用小模型分数政策豁免算子。

本桥与 mHC 候选独立开发；完整 V4 需组合验证，两者组件通过不证明组合可用。
官方真实 gfx936 路径通过同门禁后删除本桥并归还原始 dispatch。

## 官方页对齐与交错层视图（0.33 后续修订）

完整 V4 Flash TP4 已读完46片原始权重，随后在 Graph 内存预估阶段被
旧的整张 cache `is_contiguous()` 检查拒绝。官方 C4 payload 为64×132
=8448字节，但 `fp8_ds_mla` 的576字节对齐产生8640字节页；BLHNC/
BLNHC 还可在同一物理块中交错其他层。官方 gather 和 paged-logits 已
接收外层 page stride，不能以全局 contiguous 限制其原有布局。

检查改为页内 stride=(132,1)、page stride≥8448且可被4整除、storage
起点可被4整除，保留原 shape/dtype/device 限制。非连续外层视图直接
交给官方 helper，不复制 cache，也不修改 compressor、metadata 或数学。
离线负控覆盖重叠页、页/起点未对齐及页内空洞，拒绝发生在输出写入前。
GPU 探针的 `--cache-storage aligned|packed` 用官方
`create_kv_cache_views` 构造，核对 Graph replay 数值和未用填充/其他层字节。
原 `dense` 对照仍保留。填充 metadata 的 `--dynamic-lens` 仅适用于探针
原有 MTP1 配方，MTP5 使用独立 padding 配方。

该修订仍为显式 opt-in 的 Adapter ABI 兼容，不新增 Plugin 算法。
组件通过不代表完整 V4、cache compressor/write 或 HumanEval32 通过；
完整模型记录与 Quark 原生产物/JIT 校验单独保存。
