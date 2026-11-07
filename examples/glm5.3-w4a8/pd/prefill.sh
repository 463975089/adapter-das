#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# All inference requests use local or private worker addresses.
export NO_PROXY="*" no_proxy="*"
export PYTHONDONTWRITEBYTECODE=1

# DTK26.10 candidate; set node and coordinator variables per topology.json.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NODE_HOST="${NODE_HOST:?Set node IP}"
NODE_IFNAME="${NODE_IFNAME:-bond1}"
HOST="${HOST:-${NODE_HOST}}"
SOCKET_IFNAME="${SOCKET_IFNAME:-${NODE_IFNAME}}"
IB_DEVICES="${IB_DEVICES:-shca_0,shca_1,shca_2,shca_3}"
export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ib
source "${SCRIPT_DIR}/common.sh"

# Only the attention CP path automatically uses an alternate stream.
export SGLANG_ROCM_USE_MULTI_STREAM=0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${SOCKET_IFNAME}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${SOCKET_IFNAME}}"

# Keep deployment-specific IB selection for RCCL, DeepEP, and Mooncake.
export NCCL_IB_HCA="${IB_DEVICES}"
export ROCSHMEM_ALLOWED_IBV_DEVICES="${IB_DEVICES}"
export ROCSHMEM_IB_GID_INDEX=0
export ROCSHMEM_GDR_DISABLE_XDP=0
export MC_TE_FILTERS="${IB_DEVICES}"
export MC_GID_INDEX=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1

# Shared HCU runtime settings from the local GLM-5.3 FP8 P/D cookbook.
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT=120

export SGLANG_USE_LIGHTOP=1
# Match native GLM prefill's strategy CP boundary and per-rank padding.
export SGLANG_ENABLE_CP_V2=1
export SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
# Accuracy must use real speculation and expert routing.
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
unset SGLANG_SIMULATE_UNIFORM_EXPERTS SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS
unset SGLANG_SIMULATED_EXPERT_BALANCE

# PP2 prefill cannot use EAGLE; only the decode service enables MTP.
# Prefill operator settings adapted from the local GLM-5.3 FP8 P/D cookbook.
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_DEQUANT_K_CACHE_BLOCK_T=16
export SGLANG_DSA_MQA_SPLIT_BY_SEQ=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export SGLANG_DSA_HCU_REUSE_SORTED_TOPK=1

# INT4 normal-mode DeepEP requires the HIPC weight layout and GEMM together.
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1

PORT="${PORT:-${PREFILL_PORT:-30000}}"
DIST_PORT="${DIST_PORT:-5001}"
DIST_INIT_HOST="${DIST_INIT_HOST:?Set role coordinator IP}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-${DIST_INIT_HOST}:${DIST_PORT}}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
PP_SIZE="${PP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-8}"
CP_SIZE="${CP_SIZE:-8}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8199}"

exec python3 -m hcu_sglang \
    --model-path "${MODEL_PATH}" \
    --trust-remote-code \
    --host "${HOST}" \
    --port "${PORT}" \
    --dist-init-addr "${DIST_INIT_ADDR}" \
    --nnodes "${NNODES}" \
    --node-rank "${NODE_RANK}" \
    --pp-size "${PP_SIZE}" \
    --disable-overlap-schedule \
    --tp-size "${TP_SIZE}" \
    --dp-size "${DP_SIZE}" \
    --ep-size "${EP_SIZE}" \
    --moe-dense-tp-size 1 \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --attn-cp-size "${CP_SIZE}" \
    --enable-prefill-cp \
    --cp-strategy interleave \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --context-length 1048576 \
    --kv-cache-dtype fp8_e4m3 \
    --dtype bfloat16 \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.8}" \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --max-running-requests 96 \
    --disable-cuda-graph \
    --page-size 64 \
    --dsa-prefill-backend flashmla_sparse \
    --dsa-decode-backend flashmla_kv \
    --quantization compressed-tensors \
    --moe-runner-backend deep_gemm \
    --enable-cache-report \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
    --disaggregation-ib-device "${IB_DEVICES}" \
    --json-model-override-args '{"index_share_for_mtp_iteration": false}' \
    --enable-dsa-cache-layer-split \
    --deepep-config "${SCRIPT_DIR}/ep_config.json"
