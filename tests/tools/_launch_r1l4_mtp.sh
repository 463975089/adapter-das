#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# R1-L4-MTP NEXTN speculative decoding on nmz26 GPU6
# target: prefill=fa3 emu (attempt), decode/verify=flashmla emu; draft=nextn layer4
# fallback if fa3-MLA prefill fails: --prefill-attention-backend triton
export HIP_VISIBLE_DEVICES=6
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=/root/hcu_t2:/root/sglang-v0521-src/python
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
mkdir -p /root/logs
cd /root
python -m sglang.launch_server \
  --model-path /root/models/R1-L4-MTP \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path /root/models/R1-L4-MTP \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --prefill-attention-backend fa3 \
  --decode-attention-backend flashmla \
  --page-size 64 \
  --disable-cuda-graph \
  --mem-fraction-static 0.50 \
  --host 0.0.0.0 --port 30079 \
  > /root/logs/r1l4_mtp.log 2>&1
