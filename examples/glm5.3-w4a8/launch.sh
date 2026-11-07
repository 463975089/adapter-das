#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
: "${SGLANG_DIR:?Set the v0.5.22.plugin source directory}"
: "${ADAPTER_DIR:?Set the adapter source directory}"
: "${PLUGIN_DIR:?Set the plugin source directory}"
: "${MODEL_PATH:?Set the compressed-tensors checkpoint directory}"
: "${WORK_DIR:?Set the writable task directory}"
export PYTHONPATH="$ADAPTER_DIR:$SGLANG_DIR/python:$PLUGIN_DIR${PYTHONPATH:+:$PYTHONPATH}"
# Use Ninja's bundled executable so build helpers do not re-enter Python bootstrap.
glm_ninja_bin="$(HCU_ADAPTER_HOST=none python3 -c 'import ninja; print(ninja.BIN_DIR)')"
export PATH="$glm_ninja_bin:$PATH"
export HCU_ADAPTER_HOST=sglang HCU_W4A8_MOE=native HCU_SLIMQUANT=0
export SGLANG_PLUGINS=hcu SGLANG_USE_AITER=0
# Keep the initial packed-INT4 probe on the Triton W4A8 path.
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_OPT_USE_TOPK_V2=0
# Match the native recipe's gfx936 INT8 index-K cache.
export SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
# Compilation caches must stay local to each container across multiple nodes.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/glm53-w4a8/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/glm53-w4a8/torch_extensions}"
unset NCCL_TOPO_FILE
cd "$WORK_DIR"
exec python3 -u -m hcu_sglang \
    --model-path "$MODEL_PATH" --served-model-name glm5.3-w4a8 \
    --host 0.0.0.0 --port "${PORT:-30832}" --trust-remote-code \
    --tp-size 8 --ep-size 8 --quantization compressed-tensors \
    --moe-runner-backend triton --moe-a2a-backend none \
    --disable-custom-all-reduce --disable-cuda-graph --disable-prefill-cuda-graph \
    --disable-shared-experts-fusion --context-length 8192 \
    --chunked-prefill-size 2048 --max-running-requests 4 --mem-fraction-static 0.88 \
    --kv-cache-dtype fp8_e4m3 --page-size 64 \
    --dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_kv \
    --enable-metrics --random-seed 42 "$@"
