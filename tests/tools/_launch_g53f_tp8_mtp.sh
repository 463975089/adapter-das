#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# GLM-5.3-Flash TP8 + MTP only (no dp-attention, no EP) — bisection arm:
# dp8ep8+MTP serving was RED (garbage + accept 0.01-0.03). This run adds
# ONLY MTP over the green TP4 recipe family (tp8 was green in Phase A).
# Green => culprit is dp-attention/EP8; RED => MTP verify path vs emu arms.
# 2026-10-17 specloc arm: sitecustomize #14 (get_last_loc->torch) loaded;
# bf16 KV — fp8 KV pending its own bring-up (fa3 emu D=0 slab + KDA verify
# empty-batch crash sites).
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
  --tp-size 8 \
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
  > /root/logs/srv_g53f_tp8_mtp.log 2>&1
