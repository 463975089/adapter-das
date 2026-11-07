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
mkdir -p /var/log/shca
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
export ROCSHMEM_BACKEND=gda
export ROCSHMEM_GDA_PROVIDER=shca
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
export SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
# Accuracy must use real speculation and expert routing.
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
unset SGLANG_SIMULATE_UNIFORM_EXPERTS SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS
unset SGLANG_SIMULATED_EXPERT_BALANCE

# Decode uses the checkpoint MTP with EAGLE 5/1/6; CT quantization stays enabled.
# Decode operator settings adapted from the local GLM-5.3 FP8 P/D cookbook.
export GPU_MAX_HW_QUEUES=3
export HIP_GRAPH_USE_CMD_CACHE=1
export SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK="${SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK:-1}"
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1

# Cross-node RCCL is required for this DP/EP group.
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
# Keep the default low-latency dispatch buffer size.
unset SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export LIGHTOP_SPARSE_MQA_GROUP6_ROWS_PER_CTA=3

if [[ -n "${TOPO_FILE:-}" ]]; then
    if [[ ! -r "${TOPO_FILE}" ]]; then
        echo "Topology file not readable: ${TOPO_FILE}" >&2
        exit 1
    fi
    export ROCSHMEM_TOPO_FILE_FORCE="${TOPO_FILE}"
fi

PORT="${PORT:-${DECODE_PORT:-30032}}"
DIST_PORT="${DIST_PORT:-5000}"
DIST_INIT_HOST="${DIST_INIT_HOST:?Set role coordinator IP}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-${DIST_INIT_HOST}:${DIST_PORT}}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
TP_SIZE="${TP_SIZE:-16}"
DP_SIZE="${DP_SIZE:-16}"
EP_SIZE="${EP_SIZE:-16}"

exec python3 -m hcu_sglang \
    --model-path "${MODEL_PATH}" \
    --trust-remote-code \
    --host "${HOST}" \
    --port "${PORT}" \
    --dist-init-addr "${DIST_INIT_ADDR}" \
    --nnodes "${NNODES}" \
    --node-rank "${NODE_RANK}" \
    --tp-size "${TP_SIZE}" \
    --moe-dense-tp-size 1 \
    --dp-size "${DP_SIZE}" \
    --ep-size "${EP_SIZE}" \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --enable-dp-lm-head \
    --deepep-mode low_latency \
    --context-length 1048576 \
    --speculative-draft-lm-head-vp-size 8 \
    --page-size 64 \
    --dsa-prefill-backend flashmla_auto \
    --dsa-decode-backend flashmla_kv \
    --quantization compressed-tensors \
    --moe-runner-backend deep_gemm \
    --dtype bfloat16 \
    --dist-timeout 10000 \
    --watchdog-timeout 3600 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.858}" \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --cuda-graph-max-bs-decode 16 \
    --max-running-requests 128 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --json-model-override-args '{"index_share_for_mtp_iteration": false}' \
    --tokenizer-worker-num 1 \
    --enable-cache-report \
    --enable-request-time-stats-logging \
    --enable-metrics \
    --expert-balancedness-report-mode both \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device "${IB_DEVICES}"
