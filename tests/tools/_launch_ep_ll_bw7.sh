#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# EP low_latency (DeepEP deepep_ll) is_cuda form on bw7 gfx936.
# Validated 2026-10-13 (Qwen3-30B-A3B TP2+EP2 GPU6+7, :30197, 5/5 anchors +
# 6/6 concurrent zero cross-talk).
#
# Recipe deltas vs the nmz26 2026-10-11 form (all three are required):
#   - SGLANG_JIT_DEEPGEMM_PRECOMPILE=false: the bw7 root image runs a DeepGEMM
#     JIT pre-compile session at startup (5.7-7.4 GB per shape), which starves
#     the bf16 masked weight pack -> OOM in _prepare_bf16_masked_bm64... .
#   - --mem-fraction-static 0.50: 0.456 < new KV-floor validator < pack-
#     doubles-the-weights ceiling 0.60. The 0.50 sweet spot sits in between.
#   - eager + triton attention: the known-good nmz26 form. (fa3 + full graph
#     died on deep_ep num_max_dispatch_tokens_per_rank assert -- graph+LL
#     combo is NOT validated, see models.md.)
#   - HCU_DEEPGEMM_BRIDGE=1 aliases the DTK `deepgemm` wheel as deep_gemm.
WORK=${WORK:-/work}
cd "$WORK"
export SGLANG_RUST_BUILD_MODE=never
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export HCU_DEEPGEMM_BRIDGE=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=false
export HCU_FA_IMPL=triton
export ADAPTER_ROOT=${ADAPTER_ROOT:-$WORK}   # dir with sgl_kernel/ + sitecustomize.py
export SGSRC=${SGSRC:-$WORK/official-sglang/python}
export PYTHONPATH=$ADAPTER_ROOT:$SGSRC
export HIP_VISIBLE_DEVICES=${GPUS:-6,7}
M=${M:-/models/llm-models/qwen3/Qwen3-30B-A3B}
PORT=${PORT:-30197}
NCCL=${NCCL:-30397}
exec python3 -m sglang.launch_server \
  --model-path $M \
  --trust-remote-code \
  --served-model-name q30ll \
  --attention-backend triton \
  --tp-size 2 --ep-size 2 \
  --moe-a2a-backend deepep --deepep-mode low_latency \
  --deepep-dispatcher-output-dtype bf16 \
  --moe-runner-backend deep_gemm \
  --disable-radix-cache \
  --disable-cuda-graph \
  --mem-fraction-static 0.50 \
  --nccl-port $NCCL \
  --host 0.0.0.0 --port $PORT \
  > $WORK/srv_ep_ll.log 2>&1
