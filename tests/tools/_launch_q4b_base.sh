#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Task73: Qwen3.5-4B no-MTP baseline - identical to _launch_q4b_mtp.sh minus
# the speculative flags, for the MTP throughput delta on the same GPU.
export PYTHONPATH=/root/hcu_t2:/root/sglang-v0521-src/python
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export HCU_AITER_AR=0
export CUDA_VISIBLE_DEVICES=6
M=/public/opendas/DL_DATA/llm-models/qwen3.5/Qwen3.5-4B
mkdir -p /root/logs
exec python -m sglang.launch_server \
  --model-path $M \
  --attention-backend fa3 \
  --disable-cuda-graph \
  --mem-fraction-static 0.80 \
  --host 0.0.0.0 --port 30083 \
  > /root/logs/q4b_base_30083.log 2>&1
