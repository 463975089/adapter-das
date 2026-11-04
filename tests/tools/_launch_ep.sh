#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# EP (expert parallel) validation - DeepEP backend, Qwen3-30B-A3B (128 experts), EP2 on GPU6+7
# Usage: _launch_ep.sh <normal|low_latency|auto> [port]
MODE=$1
PORT=${2:-30192}
LOG=/root/logs/ep_${MODE}_${PORT}.log
export PYTHONPATH=/root/hcu_t2:/root/sglang-v0521-src/python
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export HCU_AITER_AR=0
export HCU_DEEPGEMM_BRIDGE=1
export SGLANG_DEEPEP_BF16_DISPATCH=1
export CUDA_VISIBLE_DEVICES=6,7
mkdir -p /root/logs
EXTRA=""
if [ "$MODE" = "low_latency" ] || [ "$MODE" = "auto" ]; then EXTRA="--disable-radix-cache"; fi
exec python -m sglang.launch_server \
  --model-path /public/opendas/DL_DATA/llm-models/qwen3/Qwen3-30B-A3B \
  --trust-remote-code \
  --attention-backend triton \
  --tp-size 2 --ep-size 2 \
  --moe-a2a-backend deepep \
  --deepep-mode $MODE \
  --moe-runner-backend deep_gemm \
  $EXTRA \
  --disable-cuda-graph \
  --mem-fraction-static 0.60 \
  --nccl-port 30392 \
  --host 0.0.0.0 --port $PORT \
  > $LOG 2>&1
