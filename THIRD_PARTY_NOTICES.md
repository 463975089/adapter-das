# 第三方与外部来源清单（THIRD_PARTY_NOTICES）

> 本文件登记仓库内所有非本仓原创的源码来源（HYGON 开源合规规则
> `hygon-open-source-v1.7` §4 要求）。新增移植/借源代码时同步登记本表；
> 纯 HYGON 原创文件只需文件头 HYGON Copyright + SPDX，不进本表。

## 1. 外部开源项目

### flash-linear-attention（MIT）

- 项目：flash-linear-attention（fla-org），Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
- 仓库：<https://github.com/fla-org/flash-linear-attention>（MIT）
- 引入版本：经 sglang-das fork 固定 commit `364fabf`（见 §2）整体引入；
  文件内 `Adapted from` 链接为上游 main 快照，以其所在 fork commit 为固定锚点
- 本地路径：`hcu_adapter/fla_hcu/{chunk_delta_h,chunk_intra,chunk_intra_token_parallel,cumsum,index,kda,l2norm,op,utils}.py`
- HYGON 修改：包内 import 重写到本适配层包；携带 fork 的 gfx938 修正
  （清单见 `hcu_adapter/fla_hcu/__init__.py`）。原文件内版权与许可声明原样保留。

### vLLM（Apache-2.0）

- 项目：vLLM，Copyright contributors to the vLLM project
- 仓库：<https://github.com/vllm-project/vllm>（Apache-2.0）
- 引入版本：`kda.py` 锚定 vLLM commit `0384aa7150c4c9778efca041ffd1beb3ad2bd694`；
  `conv1d_triton.py` 经 sglang-das fork（`364fabf`）引入
- 本地路径：`hcu_adapter/fla_hcu/{kda,conv1d_triton}.py`
- HYGON 修改：随 fla_hcu 包的 import 重写与 gfx938 修正；原 SPDX/版权声明原样保留。

### causal-conv1d（BSD-3-Clause）

- 项目：causal-conv1d，Copyright (c) 2024, Tri Dao
- 仓库：<https://github.com/Dao-AILab/causal-conv1d>（BSD-3-Clause）
- 引入版本：经 vLLM `causal_conv1d.py` / sglang-das fork（`364fabf`）引入
- 本地路径：`hcu_adapter/fla_hcu/conv1d_triton.py`
- HYGON 修改：同上（import 重写、fork 的 HCU 修复）。

## 2. HYGON 内部仓库移植（同集团血缘，非外部第三方）

> 这些来源属 HYGON 团队自有仓库，无外部许可证义务；为满足来源可追溯
> 要求登记固定 commit。

### sglang-das fork（Apache-2.0，sglang 衍生）

- 仓库一：<https://github.com/maxiaobetter/sglang-das>，分支
  `main_glm53_flash_feature_cleanup_tx`，commit `364fabf`
- 仓库二：<https://github.com/HYGON-AI/sglang-das>，commit
  `7b91060ae3ef5e112048988da6f4f673cf4b6a89`
- 本地路径与修改：
  - `hcu_adapter/fla_hcu/`（除 `__init__.py`/`_shim.py` 外全部文件）——
    内核逐字移植，仅包内 import 重写；`_shim.py` 垫片与 `__init__.py`
    为本仓原创（已加 HYGON 头）。
  - `hcu_adapter/fla_hcu/fused_sigmoid_gating_recurrent.py`——KDA decode
    recurrent kernel，同上逐字移植。
  - `hcu_adapter/kernels/engram_gate_triton.py`——算术/归约顺序/launch
    geometry 与源一致，接口扩展（见该文件 docstring 与文件头 Modified by）。
  - `hcu_adapter/aiter_moe.py`——按 fork `fused_experts_impl_aiter` 配方
    重实现的桥接层（非逐字拷贝）。

### lmslim（HYGON 内部量化算子库，DTK wheel 分发，非公开开源项目）

- 来源：内部 wheel `lmslim 0.4.0+das.opt1.dtk2604.torch290` 系列
- 本地路径：`hcu_adapter/slimquant.py` 中 `_w4a8_moe_kernel`——自
  `fused_moe_kernel_int4_w4a8` 移植；该文件其余部分（checkpoint 解析、
  safetensors shim）为 HYGON 原创。

## 3. 登记规则速记

- 外部开源（MIT/BSD/Apache 等）：保留原声明 + 本表登记，不加 HYGON 头。
- HYGON 内部跨仓移植：本表登记固定 commit；有实质修改的文件加
  HYGON Copyright + `Modified by`（H2 样式，如 `engram_gate_triton.py`）。
- 纯原创：文件头两行（Copyright + SPDX-Apache-2.0），脚本
  `tools/compliance/add_hygon_headers.py` 可批量补齐并幂等跳过已有头。
