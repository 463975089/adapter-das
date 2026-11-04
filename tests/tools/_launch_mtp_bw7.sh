#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# MTP rig on bw7: Qwen3.5-27B (qwen3_5, 64L GDN hybrid, mtp_num_hidden_layers=1
# draft embedded in the main shards -- no separate draft repo needed). TP2.
# Validated 2026-10-13 (GPU4/5, :30199, 8/8 anchors, three CUDA graphs).
#
# Recipe hard-won bits (do not drop):
#   - HCU_DEEPGEMM_BRIDGE=1 is REQUIRED for the mtp roles: draft graph capture
#     lazy-loads dsa_backend which does a module-scope `import deep_gemm`.
#     Without the bridge the server dies with the misleading
#     "Unquantized DeepEP MoE requires DeepGEMM BF16" (a tree-kill secondary,
#     first error is the import).
#   - --disable-prefill-cuda-graph: this 64L/hidden5120 model has a ~15GB
#     single-bucket (8192 tok) prefill graph static pool at capture; mem 0.55
#     with 15.6 GB headroom OOMs on the FIRST bucket. Production form = no
#     prefill graph + mem 0.60.
#   - keep k small (steps1/topk1/draft2) while the per-step host cost is
#     un-diagnosed: bigger k adds draft steps only (~130ms each), the
#     ~0.8-1.0s fixed host cost dominates and throughput LOSES to base.
#
# Roles: base = no MTP (throughput reference) ; mtp = EAGLE+graph ;
#        mtpe = EAGLE+eager. Ports/logs keyed by role.
ROLE=$1
WORK=${WORK:-/work}
cd "$WORK"
export SGLANG_RUST_BUILD_MODE=never
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export HCU_LINEAR_ATTN_GDN=triton
export HCU_FA_IMPL=triton
export HCU_DEEPGEMM_BRIDGE=1
export ADAPTER_ROOT=${ADAPTER_ROOT:-$WORK}   # dir with sgl_kernel/ + sitecustomize.py
export SGSRC=${SGSRC:-$WORK/official-sglang/python}
export PYTHONPATH=$ADAPTER_ROOT:$SGSRC
export HIP_VISIBLE_DEVICES=${GPUS:-4,5}
M=${M:-/models/llm-models/qwen3.5/Qwen3.5-27B}
SPEC="--speculative-algorithm EAGLE --speculative-draft-model-path $M --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2"
COMMON="--model-path $M --trust-remote-code --served-model-name q35m --attention-backend fa3 --tp-size 2 --mem-fraction-static 0.60 --max-running-requests 8 --disable-radix-cache --nccl-port 30399 --host 0.0.0.0"
if [ "$ROLE" = "base" ]; then
  exec python3 -m sglang.launch_server $COMMON --disable-prefill-cuda-graph --port 30199 > $WORK/srv_mtp_base.log 2>&1
elif [ "$ROLE" = "mtp" ]; then
  exec python3 -m sglang.launch_server $COMMON $SPEC --disable-prefill-cuda-graph --port 30199 > $WORK/srv_mtp_g.log 2>&1
elif [ "$ROLE" = "mtpe" ]; then
  exec python3 -m sglang.launch_server $COMMON $SPEC --disable-cuda-graph --port 30199 > $WORK/srv_mtp_e.log 2>&1
else
  echo "usage: $0 base|mtp|mtpe" >&2; exit 2
fi
