#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# PD disaggregation (node-internal) - PREFILL server on GPU6, Qwen3-0.6B
export PYTHONPATH=/root/hcu_t2:/root/sglang-v0521-src/python
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export HCU_AITER_AR=0
export MC_TCP_ENABLE_CONNECTION_POOL=true
export CUDA_VISIBLE_DEVICES=6
mkdir -p /root/logs
exec python -m sglang.launch_server \
  --model-path /root/models/Qwen3-0.6B \
  --trust-remote-code \
  --attention-backend triton \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake_tcp \
  --disaggregation-bootstrap-port 30590 \
  --nccl-port 30390 \
  --tp 1 \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 --port 30190 \
  > /root/logs/pd_prefill_30190.log 2>&1
