# sgl_kernel HCU wheel 构建配方（参考线 / 可选 cpp 档）

从官方 v0.5.22 aot 树（`python/sglang/kernels/aot`，工作区拷贝，官方树
零改动）在 dtk2610 / py3.12 容器上构建 `sglang-kernel` wheel。用法见
`build.sh` 头部注释（`HCU_GPU_TARGET` 缺省 gfx938）。

## 状态（2026-11-03）：暂停，未出过完整 wheel

主线已按 governance §0.5 走 vendor 档（lightop / aiter）闭环 L1 算子；
本目录是 vendor 覆盖不到的算子补 **cpp 档** 时的参考配方。恢复构建时
从"头文件已处理、进入编译"这一步继续。

round 7 卡点（`math_hip.cuh` FileNotFoundError）已定性并修复：锁定
flashinfer commit bc29697b 里只有 `math.cuh`（`math_hip.cuh` 是后续版本
的文件名），patch 目标已改正，并对锁定 commit 实测：**10/10 替换命中、
残余 asm 语句 0**。下一轮尝试从编译错误开始计。

## 七轮已修问题（每轮一个修复，落在对应文件）

| 轮 | 症状 | 修复落点 |
|---|---|---|
| 1 | 缺 flashinfer 头 / cuda_fp8.h | `build.sh` 拉锁定 commit（sha256 校验）+ `shims/` |
| 2 | thrust / vector_types 与 hipcc 冲突 | 弃 DTK 全兼容目录，自写最小 shim |
| 3 | cuda_fp8.h 转发壳闭包无底洞 | `shims/cuda_fp8.h` 映射 hip 原生 fp8 API |
| 4 | PTX asm `'=f'` 约束编不过 | `patch_math_hip.py` 可移植化 |
| 5 | dtk2610 wave64 要求 u64 掩码 | `build.sh` sed `moe_align_kernel.cu`（.cu 而非 hipify 现生成的 .hip） |
| 6 | eagle.cuh 缺失 / merge_attn_states bf16 编不过 | `setup_hcu.py` 源列表剔除（理由在注释） |
| 7 | `math_hip.cuh` 不存在 | 文件名改正 → `math.cuh`（见上节） |

另修镜像 wheel 的根因：官方 `load_utils.py` 只 glob `sm90//sm100/`
子目录，而本构建把 .so 放包顶层——`patch_load_utils.py` 指 glob 到
顶层。这正是 dtk2604 镜像 `torch.ops.sgl_kernel == []` 空壳 wheel 的
成因（2026-11-03 复核）。

## 文件

- `build.sh` — 编排：工作区拷贝 → 拉 flashinfer 锁定 commit → 打三处
  patch → `setup_hcu.py bdist_wheel`
- `setup_hcu.py` — das `setup_hip.py` 配方 + HCU 目标白名单 +
  L1 三算子源；版本后缀 `+hcu.dtk2610`
- `patch_math_hip.py` — `math.cuh` 十个 PTX 函数体可移植化
- `patch_load_utils.py` — 装载 glob 指向包顶层（修空 op wheel 根因）
- `shims/cuda_fp8.h` — CUDA→HIP fp8 拼写映射（零新代码路径）
- `shims/cuda_device_runtime_api.h` — 故意空的 stub（缺失符号会响亮编译失败）

## MR 卫生（governance §5.0）

按"MR 只提交有用的部分，无用代码提前检查清理"：本目录六个文件各承载
一个已验证的修复（无装饰性步骤、无调试残留），整目录自洽入库。构建
产物（工作区、日志、tarball）不入库，留在构建容器 `/home/build_sgl_kernel`。
