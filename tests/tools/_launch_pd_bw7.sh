#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# PD disaggregation (mooncake_tcp, node-internal) is_cuda form on bw7 gfx936.
# Validated 2026-10-13 (Qwen3-0.6B, GPU6=P 30190 / GPU7=D 30191 / mini-LB 30090,
# 8/8 distinct-prompt zero cross-talk).
#
# Recipe hard-won bits (do not drop):
#   - decode role IGNORES --disable-radix-cache (create_tree_cache forces the
#     radix factory); SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python is the
#     sanctioned fallback around the rust tree extension (bw7 cargo 1.75
#     cannot build the edition2024 crate, and the never-mode fallback probe
#     does not detect "toolchain too old").
#   - prefill cuda-graph default 58 buckets OOMs at mem 0.45 (bucket 26/58):
#     pass the explicit 9-bucket list (same disease as GDN 137b9a2).
#   - always pin --attention-backend (aiter default resolution breaks dense
#     prefill) ; --page-size 64 must match on both roles.
#
# Usage (each role is a SEPARATE `docker exec -d` — a chained `cmd P; cmd D`
# never starts D because the exec'ed P occupies the shell):
#   docker exec -d <ctr> /path/_launch_pd_bw7.sh P
#   docker exec -d <ctr> /path/_launch_pd_bw7.sh D
#   docker exec -d <ctr> /path/_launch_pd_bw7.sh LB
ROLE=$1
WORK=${WORK:-/work}
cd "$WORK"
export MC_TCP_ENABLE_CONNECTION_POOL=true
export SGLANG_RUST_BUILD_MODE=never
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export HCU_FA_IMPL=triton
export PYTHONPATH=$WORK:$WORK/official-sglang/python
ADAPTER_ROOT=${ADAPTER_ROOT:-$WORK}          # dir containing sgl_kernel/ + sitecustomize.py
SGSRC=${SGSRC:-$WORK/official-sglang/python}
M=${M:-/models/llm-models/qwen3/Qwen3-0.6B}
export PYTHONPATH=$ADAPTER_ROOT:$SGSRC
if [ "$ROLE" = "P" ]; then
  export HIP_VISIBLE_DEVICES=6
  exec python3 -m sglang.launch_server --model-path $M --trust-remote-code --served-model-name q06p --attention-backend fa3 --disable-radix-cache --mem-fraction-static 0.45 --cuda-graph-bs-prefill 4 8 32 128 512 1024 2048 4096 8192 --disaggregation-mode prefill --disaggregation-transfer-backend mooncake_tcp --disaggregation-bootstrap-port 30590 --nccl-port 30390 --tp 1 --page-size 64 --host 0.0.0.0 --port 30190 > $WORK/srv_pd_p.log 2>&1
elif [ "$ROLE" = "D" ]; then
  export HIP_VISIBLE_DEVICES=7
  exec python3 -m sglang.launch_server --model-path $M --trust-remote-code --served-model-name q06d --attention-backend fa3 --disable-radix-cache --mem-fraction-static 0.45 --disaggregation-mode decode --disaggregation-transfer-backend mooncake_tcp --disaggregation-bootstrap-port 30590 --nccl-port 30391 --tp 1 --page-size 64 --host 0.0.0.0 --port 30191 > $WORK/srv_pd_d.log 2>&1
elif [ "$ROLE" = "LB" ]; then
  exec python3 -m sglang_router.launch_router --pd-disaggregation --mini-lb --prefill http://127.0.0.1:30190 --decode http://127.0.0.1:30191 --host 0.0.0.0 --port 30090 > $WORK/srv_pd_lb.log 2>&1
else
  echo "usage: $0 P|D|LB" >&2; exit 2
fi
