#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# trace-mode FP8 server launcher (container-side)
export PYTHONPATH=/adapter:/sglang-v0521/python
export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:/usr/local/lib/python3.10/dist-packages/amdsmi:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/ucx/lib:/opt/mpi/lib:/opt/hwloc/lib
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export CUDA_VISIBLE_DEVICES=1
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export SGLANG_RUST_BUILD_MODE=never
export HCU_FP8_TRACE=1
cd /adapter
pkill -f "port 301" 2>/dev/null
sleep 2
exec python -m sglang.launch_server \
  --model-path /models/qwen3/Qwen3-4B-Channel-FP8 \
  --port 30113 --tp-size 1 --attention-backend triton \
  --mem-fraction-static 0.80 --disable-cuda-graph
