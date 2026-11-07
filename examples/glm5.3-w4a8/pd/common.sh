#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${ADAPTER_DIR:?Set adapter source root}"
: "${PLUGIN_DIR:?Set plugin source root}"
: "${SGLANG_DIR:?Set v0.5.22.plugin source root}"
: "${MODEL_PATH:?Set CT W4A8 model root}"
export PYTHONPATH="$ADAPTER_DIR:$SGLANG_DIR/python:$PLUGIN_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HCU_ADAPTER_HOST=sglang HCU_W4A8_MOE=native HCU_SLIMQUANT=0
# The base and plugin already handle DeepEP's vendor quant_type API.
# The legacy use_fp8 wrapper hides that API and cannot represent INT8.
export HCU_DEEPEP_BRIDGE=0
export SGLANG_PLUGINS=hcu SGLANG_USE_AITER=0 SGLANG_ROCM_USE_AITER_MOE=0
# Select the plugin's qualified native RoPE and public W8A8 paths.
export SGLANG_HCU_USE_LIGHTOP_ROPE=1
export SGLANG_HCU_W8A8_USE_LIGHTOP=1
# Match the native BF16 router projection and LightOp expert selection.
export SGLANG_HCU_GLM_CT_BF16_ROUTER=1 SGLANG_LIGHTOP_TOPK=1
# Match the native residual RMSNorm's FP32 sum and final BF16 rounding.
export HCU_FUSED_ADD_RMS_IMPL=lightop
# Preserve native BF16 rounding in plain attention Q/K normalization as well.
export HCU_RMS_NORM_IMPL=lightop
# Override the legacy adapter default; the plugin provides native LightOp Top-K.
export SGLANG_DSA_FUSE_TOPK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/var/cache/glm53-candidate/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/var/cache/glm53-candidate/torch_extensions}"
glm_ninja_bin="$(HCU_ADAPTER_HOST=none python3 -c 'import ninja; print(ninja.BIN_DIR)')"
export PATH="$glm_ninja_bin:$PATH"
unset NCCL_TOPO_FILE
