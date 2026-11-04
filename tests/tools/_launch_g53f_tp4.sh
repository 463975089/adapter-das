#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# GLM-5.3-Flash (/models/glm53f) TP4 production recipe — nmz26 container
# g53-tp8, GPU4-7, port 30187, log /root/logs/srv_g53f_tp4.log
# stack: dtk26041 + torch 2.11 + official sglang v0.5.21 + adapter (hcu_t2)
# 2026-10-16: anchors green (Berlin id=19808 etc.), decode 0.43s/step
# (host hipGraphLaunch ~300ms); cookbook graph-dispatch trio added below,
# pending A/B + anchor re-check at next start.
cd /root
export PYTHONPATH=/root/hcu_t2:/sglang-v0521/python
export LD_LIBRARY_PATH=/opt/dtk/dcc/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/hip/lib:/opt/dtk/lib
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export SGLANG_DSA_FUSE_TOPK=0
export SGLANG_DISABLE_DSA_INDEXER_FUSION=1
export HIP_VISIBLE_DEVICES=4,5,6,7
export HCU_DEEPGEMM_BRIDGE=1
# cookbook graph-dispatch block — A/B REJECTED 2026-10-16 on this stack
# (dtk26041/gfx938): no throughput change (2.23 vs 2.34 tok/s single,
# 14.0 vs 14.2 bs8) AND greedy decode turned NON-deterministic (Berlin
# anchor flipped between runs at temperature=0). Do not re-enable without
# a DTK runtime fix. Details: docs/plugins.md P9, docs/env-vars.md.
# export ROC_AQL_QUEUE_SIZE=131072
# export HIP_GRAPH_ACCUMULATE_DISPATCH=1
# export HIP_GRAPH_USE_CMD_CACHE=1
unset NCCL_TOPO_FILE
unset ROCR_VISIBLE_DEVICES
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python3 -m sglang.launch_server --model-path /models/glm53f --served-model-name glm53-flash --trust-remote-code --host 0.0.0.0 --port 30187 --tp-size 4 --page-size 64 --mem-fraction-static 0.62 --chunked-prefill-size 8192 --max-running-requests 16 --cuda-graph-max-bs-decode 16 --cuda-graph-bs-prefill 8 16 32 --disable-radix-cache --dsa-topk-backend torch --dsa-prefill-backend fa3 --dsa-decode-backend flashmla_sparse --disable-prefill-cuda-graph --nccl-port 30387 > /root/logs/srv_g53f_tp4.log 2>&1
