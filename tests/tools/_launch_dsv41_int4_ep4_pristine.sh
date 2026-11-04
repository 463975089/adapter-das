#!/usr/bin/env bash
# DeepSeek-V4.1 INT4 W4A8 -> W4A16 DeepEP EP4 launch recipe.
#
# This script performs the pristine-source check before starting SGLang.
# It is an execution recipe, not a model-pass claim; capture logs and run the
# semantic/EP1 comparison probes after the server becomes ready.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SGLANG_TREE="${SGLANG_TREE:-/home/lipenghui/lph-adapter/sglang-0.5.22-pristine-iter7}"
MODEL_PATH="${MODEL_PATH:-/model/DeepSeek-V4.1-Flash-Channel-INT4-w4a8}"
PORT="${PORT:-30000}"

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
export HCU_ADAPTER_HOST=sglang
export SGLANG_DSV4_FP4_DEQUANT=1
export SGLANG_HCU_EP_BF16=1
export SGLANG_OPT_FUSE_MHC_POST_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=0
export HCU_W4A8_MOE=w4a16
unset NCCL_TOPO_FILE
unset ROCR_VISIBLE_DEVICES

python -m tools.upstream_source --up-tree "$SGLANG_TREE"

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tp-size 4 \
  --ep-size 4 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --moe-runner-backend deep_gemm \
  --page-size 64 \
  --mem-fraction-static 0.86 \
  --max-running-requests 16 \
  --chunked-prefill-size 8192 \
  --cuda-graph-max-bs-decode 16 \
  --skip-server-warmup \
  --watchdog-timeout 1800 \
  --dsa-topk-backend torch \
  --dsa-prefill-backend fa3 \
  --dsa-decode-backend flashmla_sparse
