#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# PD RDMA prefill (GPU6, port 30195) - mooncake RDMA transport (RoCE)
# MUST be started via `docker exec --privileged`: CAP_IPC_LOCK exempts RDMA
# memlock accounting (container default memlock=64MB otherwise ENOMEM on CQ).
python3 -c "import resource; resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1,-1))" 2>/dev/null
export PYTHONPATH=/root/hcu_t2:/root/sglang-v0521-src/python
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export CUDA_VISIBLE_DEVICES=6
exec python -m sglang.launch_server \
  --model-path /root/models/Qwen3-0.6B \
  --trust-remote-code \
  --attention-backend triton \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 30591 \
  --nccl-port 30395 \
  --tp 1 --page-size 64 \
  --host 0.0.0.0 --port 30195 \
  > /root/logs/pd_rdma_prefill_30195.log 2>&1
