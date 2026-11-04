#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# EP auto mode (prefill normal contiguous + decode low_latency) - Qwen3-30B-A3B
# TP2+EP2 on GPU4+5, triton contiguous grouped GEMM via HCU_TRITON_CONTIG=1 (default)
export PYTHONPATH=/root/hcu_t2:/root/sglang-v0521-src/python
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export HCU_AITER_AR=0
export HCU_DEEPGEMM_BRIDGE=1
export SGLANG_DEEPEP_BF16_DISPATCH=1
export CUDA_VISIBLE_DEVICES=4,5
mkdir -p /root/logs
exec python -m sglang.launch_server \
  --model-path /public/opendas/DL_DATA/llm-models/qwen3/Qwen3-30B-A3B \
  --trust-remote-code \
  --attention-backend triton \
  --tp-size 2 --ep-size 2 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --moe-runner-backend deep_gemm \
  --disable-radix-cache \
  --disable-cuda-graph \
  --mem-fraction-static 0.60 \
  --nccl-port 30393 \
  --host 0.0.0.0 --port 30193 \
  > /root/logs/ep_auto_30193.log 2>&1
