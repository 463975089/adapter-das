#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Qwen3.5-122B-A10B-AWQ TP4 on nmz26 GPU0-3
# key: --quantization moe_wna16 (upstream triton W4A16 MoE runner + plain AWQ linear
# via HIP triton dequant). Default 'awq' auto-converts to awq_marlin on cap=(9,3)
# and crashes: awq_marlin_repack is None on the HIP branch.
export HIP_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/root/hcu_t2:/root/sglang-v0521-src/python
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export HCU_AITER_AR=0
# fp16 checkpoint: conv-state pool defaults to bf16 upstream (env-only knob,
# ignores model dtype) -> dtype mismatch at first GDN forward. Align to fp16.
export SGLANG_MAMBA_CONV_DTYPE=float16
mkdir -p /root/logs
cd /root
python -m sglang.launch_server \
  --model-path /public/opendas/DL_DATA/llm-models/qwen3.5/Qwen3.5-122B-A10B-AWQ \
  --tp-size 4 \
  --attention-backend triton \
  --quantization moe_wna16 \
  --trust-remote-code \
  --disable-cuda-graph \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 --port 30078 \
  > /root/logs/122b_awq.log 2>&1
