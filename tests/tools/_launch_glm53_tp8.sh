#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# GLM-5.3-Flash Phase A baseline: TP8, sglang v0.5.21 + adapter 67176fc
# Model: native fp8 (dynamic), hybrid mamba/NSA-MQA + MoE, page 256
export PYTHONPATH=/work/hcu_t2:/work/sglang-v0521/python
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /work/glm53_logs
exec python -m sglang.launch_server \
  --model-path /work/models/GLM-5.3-Flash \
  --trust-remote-code \
  --tp-size 8 --page-size 256 \
  --max-running-requests 64 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 --port 30053 \
  --log-level info \
  > /work/glm53_logs/glm53_tp8_30053.log 2>&1
