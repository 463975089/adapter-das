#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# bisection: R1-L4 WITHOUT speculative decoding, same mixed attention pair
# (prefill=fa3 emu incl. MLA qv path, decode=flashmla emu). If output is
# garbage -> the qv prefill math is wrong; if clean -> spec plumbing.
export HIP_VISIBLE_DEVICES=6
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=/root/hcu_t2:/root/sglang-v0521-src/python
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
mkdir -p /root/logs
cd /root
python -m sglang.launch_server \
  --model-path /root/models/R1-L4 \
  --prefill-attention-backend fa3 \
  --decode-attention-backend flashmla \
  --page-size 64 \
  --disable-cuda-graph \
  --mem-fraction-static 0.55 \
  --host 0.0.0.0 --port 30080 \
  > /root/logs/r1l4_fa3pf.log 2>&1
