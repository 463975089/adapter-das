#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# GLM-5.3-Flash dp8ep8 + dp-attention, NO MTP — deliver the dp/EP perf arm.
# MTP red on stock v0.5.21 (accept 0.01, hidden drift; das env-gated spec
# cache-loc patches absent here) — filed separately, do not mix variables.
cd /root
export PYTHONPATH=/root/hcu_t2:/sglang-v0521/python
export LD_LIBRARY_PATH=/opt/dtk/dcc/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/hip/lib:/opt/dtk/lib
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export SGLANG_DSA_FUSE_TOPK=0
export SGLANG_DISABLE_DSA_INDEXER_FUSION=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCU_DEEPGEMM_BRIDGE=1
unset NCCL_TOPO_FILE
unset ROCR_VISIBLE_DEVICES
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python3 -m sglang.launch_server --model-path /models/glm53f \
  --served-model-name glm53-flash --trust-remote-code \
  --host 0.0.0.0 --port 30188 \
  --tp-size 8 --dp-size 8 --ep-size 8 --moe-dense-tp-size 1 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --dsa-prefill-backend fa3 --dsa-decode-backend flashmla_sparse \
  --dsa-topk-backend torch \
  --page-size 64 \
  --mem-fraction-static 0.75 \
  --chunked-prefill-size 8192 \
  --max-running-requests 16 --cuda-graph-max-bs-decode 16 \
  --disable-radix-cache --disable-prefill-cuda-graph \
  --nccl-port 30388 \
  > /root/logs/srv_g53f_dp8ep8.log 2>&1
