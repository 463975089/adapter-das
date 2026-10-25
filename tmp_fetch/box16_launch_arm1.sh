#!/bin/bash
# box16 stall repro — arm 1: tp8/dp8/ep8 dp-attention graph decode, NO speculative
# Purpose: does the ~19.4ms graph-launch tick / A-family stall appear on a
# HEALTHY 3rd box (BW1200, dtk2604) with production-like topology?
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_TOPO_FILE=
cd /root
nohup python3 -m sglang.launch_server \
  --model-path /models/DeepSeek-V4.1-Flash \
  --served-model-name dsv41f \
  --trust-remote-code \
  --host 0.0.0.0 --port 30900 \
  --tp-size 8 --dp-size 8 --enable-dp-attention --enable-dp-lm-head \
  --ep-size 8 --moe-a2a-backend none \
  --page-size 64 --mem-fraction-static 0.80 \
  --chunked-prefill-size 8192 --max-running-requests 16 \
  --cuda-graph-max-bs-decode 16 \
  --disable-radix-cache \
  --nccl-port 30901 --dist-timeout 10000 \
  > /root/logs_srv_arm1.log 2>&1 &
echo "ARM1 FIRED $(date +%H:%M:%S)"
