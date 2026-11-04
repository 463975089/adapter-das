#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# DP attention test: Qwen3-0.6B, TP2 + DP2 on GPU4/5
export PYTHONPATH=/root/hcu_t2:/root/sglang-v0521-src/python
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export CUDA_VISIBLE_DEVICES=4,5
mkdir -p /root/logs
exec python -m sglang.launch_server   --model-path /root/models/Qwen3-0.6B   --trust-remote-code   --attention-backend triton   --tp-size 2 --dp-size 2 --enable-dp-attention   --mem-fraction-static 0.80   --nccl-port 30398   --host 0.0.0.0 --port 30198   > /root/logs/dp_attn_30198.log 2>&1
