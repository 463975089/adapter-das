#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# GLM-5.3-Flash (/models/glm53f) dp8ep8 + MTP — validation-host-26 container g53-tp8
# 8x GPU0-7, port 30188, log /root/logs/srv_g53f_dp8ep8_mtp.log
# stack: dtk26041 + torch 2.11 + official sglang v0.5.21 + adapter (hcu_t2)
# recipe = TP4 baseline envs + cookbook glm-5.3 dp8ep8+MTP flags adapted:
#   cookbook-only stack envs (LIGHTOP / big-AQL / SPEC_V2 / draft-lm-head-vp)
#   are NOT imported — this stack has its own kernel arms (fa3 prefill,
#   flashmla_sparse decode, dsa-topk torch) and vetoed the AQL envs.
# MTP: EAGLE via nextn layer (num_nextn_predict_layers=1, layers.45 in ckpt)
# deepep removed 2026-10-17: lazy DeepEPBuffer init inside decode-graph capture
# wants a fresh NCCL comm -> TCPStore rendezvous timeout (600s) on this
# single-node stack; EP8 runs via the default dispatcher (stock NCCL).
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
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --speculative-dsa-topk-backend torch \
  --dsa-prefill-backend fa3 --dsa-decode-backend flashmla_sparse \
  --dsa-topk-backend torch \
  --page-size 64 \
  --mem-fraction-static 0.75 \
  --chunked-prefill-size 8192 \
  --max-running-requests 8 --cuda-graph-max-bs-decode 8 \
  --disable-radix-cache --disable-prefill-cuda-graph \
  --nccl-port 30388 \
  > /root/logs/srv_g53f_dp8ep8_mtp.log 2>&1
