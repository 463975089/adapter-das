# MiniMax-H3 专用 partial RoPE

用途：为官方 SGLang 0.5.22 与 HCU plugin 提供 MiniMax-H3 的专用旋转接口，
保留旧 DAS provider 的部分旋转、原位更新与 dtype 舍入语义。实现独立于通用 RoPE。

## 接口与运行组合

`sgl_kernel.minimax_h3_rotary_embedding(positions, query, key, head_size,
cos_sin_cache, is_neox)` 原位更新 Q/K，返回 None。

| 项目 | 支持范围 |
|---|---|
| 几何 | DiT head128/rotary96；视频 VAE head64/rotary48 |
| 布局 | NeoX；Q/K 为二维连续 head 行，支持行间 stride |
| dtype | BF16、FP16、FP32；cache 与 Q/K dtype 一致 |
| positions | 一维 int32/int64，与 Q/K/cache 同设备 |
| 其他 | 可选 key、不同 Q/K head 数、空输入、相同 Q/K 视图去重 |
| 非旋转区 | head128 保留后32维，head64 保留后16维 |

plugin 优先调用已有 `torch.ops.sgl_kernel.minimax_h3_rotary_embedding`；
无旧 native 符号时转本公开接口。adapter 提供算子，plugin 负责 DiT/VAE 消费者接线。
其他 head/cache 几何与非 NeoX 输入明确拒绝，不静默走另一种旋转算法。

## 实现与回退

Q/K 分别发射，lane tile 使用 head_size/2 并对实际旋转区遮罩；热路径不增加
位置范围的主机读取。BF16 保留乘积先舍入再加减；FP16/FP32 按旧 native 的
第二乘积舍入和 FP32 FMA 次序计算。通用 RoPE 的实现与分发保持原样。

`HCU_DISABLE_OPS=minimax_h3_rotary_embedding` 切到 Torch 诊断分支；CPU 也使用
该数学参考。GPU 路径没有 Triton 时明确报错。诊断分支不计为性能优化。

## 固定候选的设备与模型证据

设备为单卡 gfx938，用户口径 BW1100、工具标识 BW1102。官方 SGLang 固定于
`baa9befa381511cd4d56f83980951e5a45fffc6d`；已测 adapter/plugin 候选分别基于
release `87f91522113f35e0f361747c644b29389e3d0d75` /
`b45972b1e10c85a92de09a6f6e5d277161cd041e`。

- 两种几何乘以 BF16/FP16/FP32，共六组完整 Q/K 与旧 native 逐位一致。
  输入与输出共同固定，避免跨环境 RNG 生成差异。独立归档保留参考输入、输出与哈希，当前 PR 不携带大体积 JSON。
- 旧候选设备测试覆盖原位和尾部保护、Q/K 视图、变输入/位置/cache 的 Graph replay。
  首版 BF16 舍入差异与 JIT 编译错误已保留原始失败，未用同源参考替代 native 对拍。
- 与配套 plugin 组合，T2VA、FL2VA 三种关键帧输入及 Ref2VA 五种输入组合，
  覆盖新旧、完整/压缩 AdaLN，共36条短程生成与完整音视频解码通过。
- T2VA 同卡三环境正式采样18次、完整预热6次。旧 DTK 环境换新代码，压缩/完整
  AdaLN 的 Server Total 中位数从791.856/762.342秒变为778.919/749.331秒。
  相应六个正式 MP4 与旧代码样本 SHA256 均相同，仅覆盖固定 prompt/seed 的 T2VA。
- 所有生成均设置 `FLASH_ATTENTION_FAST_SOFTMAX=1`，默认 LT，统一 medium 编码。
  新 DTK 配套环境下输出仍有数值差异，单样本 VBench 六维已评估；本算子不修改
  blaslt 调优表，不以全局关闭 LT 作为修复。

本 PR 已迁到官方 main 来源分支；专用实现与上述固定候选保持一致。上游并入部分
另做 CPU/结构检查，不能据此把历史 GPU 结果改写为合并后完整镜像重测。
测试包含 vendor Torch2.11/Triton3.6；release 元数据的 Torch2.13/Triton3.7 组合
未在本任务实测，安装时保留所选 DTK 配套依赖。catalog 仍标 candidate，不新增泛化承诺。

## 检查与证据

CPU：`python3 -m unittest discover -s tests -p 'test_minimax_h3_rotary_cpu.py' -v`。
设备检查需显式设置 `MINIMAX_H3_RUN_GPU_TESTS=1`，再运行对应 GPU 测试，并用
`HIP_VISIBLE_DEVICES` 选定已授权的卡。无空闲卡时不自动执行设备测试。

模型、trace 和 MP4 不进入源码仓库。任务证据独立保存，性能归档 SHA256：

- `performance-t2va-20261113-review.tar.gz`：
  `9177f9372a94801c07fc13c9e3c46b6e1561c9cc7dfc78f19e85711af04ac869`。
- `performance-t2va-old-dtk-20261113-review.tar.gz`：
  `ae773c6d183a461b7f8dc91e9bf36be6d22274dbebdfc5d189eb4bc2258cefa2`。

安装接入与任务选择见 [阅读指南](reading-guide.md)，开关见 [配置](config.md)。
当前提交的检查日志随评审说明提供；未测条件和完整质量容差保留验证边界。

PR 精简（2026-11-15）：旧 native 数值 fixtures 及其读取用例移出 PR，保留独立原始证据；
仓内保留轻量 CPU 参数/尾部/别名契约、GPU 数值与 DiT/VAE Graph 测试。
运行时源码与 main 来源 T2VA 两步实测候选逐文件一致，两种 AdaLN 实际请求及完整音视频解码通过。
短程结果不替代标准步数质量或性能验证。
精简后检查：L0 与 diff 检查通过；CPU pytest 42 项通过。原生 fixture 对拍仍属于归档历史证据，未在本次重跑。
