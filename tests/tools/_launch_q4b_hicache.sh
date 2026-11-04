#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Task73: Qwen3.5-4B MTP + hicache (hierarchical KV cache) on HCU.
# Small max-total-tokens forces LRU eviction so re-sent long prompts must be
# fetched back from the host-memory pool (get_device_accessible_ptr path).
export PYTHONPATH=/root/hcu_t2:/root/sglang-v0521-src/python
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export HCU_AITER_AR=0
export CUDA_VISIBLE_DEVICES=6
M=/public/opendas/DL_DATA/llm-models/qwen3.5/Qwen3.5-4B
mkdir -p /root/logs
exec python -m sglang.launch_server \
  --model-path $M \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path $M \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --attention-backend fa3 \
  --enable-hierarchical-cache \
  --max-total-tokens 2048 \
  --chunked-prefill-size 1024 \
  --mem-fraction-static 0.80 \
  --host 0.0.0.0 --port 30085 \
  > /root/logs/q4b_hicache_30085.log 2>&1
