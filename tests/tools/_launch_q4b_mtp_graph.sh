#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Task73: Qwen3.5-4B MTP (EAGLE) + cuda graph ON variant of _launch_q4b_mtp.sh
# (chain spec keeps per-phase shapes static: 3x1-token draft, 4-token verify).
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
  --mem-fraction-static 0.80 \
  --host 0.0.0.0 --port 30084 \
  > /root/logs/q4b_mtp_graph_30084.log 2>&1
