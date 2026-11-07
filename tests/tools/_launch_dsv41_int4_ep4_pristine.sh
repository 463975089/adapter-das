#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# DeepSeek-V4.1 INT4 W4A8 -> W4A16 DeepEP EP4 launch recipe.
#
# This script performs the pristine-source check before starting SGLang.
# It is an execution recipe, not a model-pass claim; capture logs and run the
# semantic/EP1 comparison probes after the server becomes ready.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SGLANG_TREE="${SGLANG_TREE:?set SGLANG_TREE to the pristine community source directory}"
MODEL_PATH="${MODEL_PATH:-/model/DeepSeek-V4.1-Flash-Channel-INT4-w4a8}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
IFS=',' read -r -a DEVICES <<< "$HIP_VISIBLE_DEVICES"
if [[ ${#DEVICES[@]} -ne 4 ]]; then
  echo "TP4/EP4 requires four explicitly selected devices: $HIP_VISIBLE_DEVICES" >&2
  exit 2
fi

if [[ ! -d "$SGLANG_TREE/python/sglang" ]]; then
  echo "missing pristine SGLang tree: $SGLANG_TREE" >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "missing model checkpoint: $MODEL_PATH" >&2
  exit 2
fi

cd "$ROOT"
export PYTHONPATH="$ROOT:$SGLANG_TREE/python${PYTHONPATH:+:$PYTHONPATH}"
export HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
export HCU_ADAPTER_HOST=sglang
export SGLANG_DSV4_FP4_DEQUANT=1
export SGLANG_HCU_EP_BF16=1
export SGLANG_OPT_FUSE_MHC_POST_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=0
export HCU_W4A8_MOE=w4a16
# Keep the normal community warmup. This EP4 recipe exceeded the community
# 600-second default on validation-host-b; allow the measured startup to finish while
# retaining an explicitly configured timeout.
export SGLANG_WARMUP_TIMEOUT="${SGLANG_WARMUP_TIMEOUT:-1800}"
# Low-latency dispatch has a fixed per-rank token capacity. Keep each
# prefill chunk within it; long prompts are served over multiple chunks.
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
unset NCCL_TOPO_FILE
unset ROCR_VISIBLE_DEVICES

HCU_ADAPTER_HOST=none python -m tools.upstream_source --up-tree "$SGLANG_TREE"

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --trust-remote-code \
  --host "$HOST" \
  --port "$PORT" \
  --tp-size 4 \
  --ep-size 4 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --deepep-dispatcher-output-dtype bf16 \
  --moe-runner-backend deep_gemm \
  --mem-fraction-static 0.90 \
  --max-running-requests 16 \
  --chunked-prefill-size 256 \
  --cuda-graph-max-bs-decode 16 \
  --language-model-only \
  --disable-prefill-cuda-graph \
  --watchdog-timeout 1800 \
  --dist-timeout 1800
