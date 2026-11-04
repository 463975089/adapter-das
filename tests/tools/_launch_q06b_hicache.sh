#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Task73: dense-model hicache isolation test - Qwen3-0.6B has no GDN layers,
# so a PASS here means the FULL-component host write-back/read-back is correct
# and any Qwen3.5-4B (GDN hybrid) drift is pinned to the MAMBA state fallback.
export PYTHONPATH=/root/hcu_t2:/root/sglang-v0521-src/python
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export HCU_AITER_AR=0
export CUDA_VISIBLE_DEVICES=6
mkdir -p /root/logs
exec python -m sglang.launch_server \
  --model-path /root/models/Qwen3-0.6B \
  --attention-backend fa3 \
  --enable-hierarchical-cache \
  --max-total-tokens 2048 \
  --chunked-prefill-size 1024 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 --port 30087 \
  > /root/logs/q06b_hicache_30087.log 2>&1
