#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

export PYTHONPATH=/work/sgl_community:/adapter
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCU_SLIMQUANT=1
export HCU_MQA_EMU=1
export HCU_DSA_PAGE_SIZE=1
export HCU_DSA_PRESHUFFLE=0
export SGLANG_DSA_FUSE_TOPK=0
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
exec python -m sglang.launch_server --model-path /public/share/GLM-5-W4A8_V2_6 --host 0.0.0.0 --port 30064 --tp-size 8 --mem-fraction-static 0.60 --chunked-prefill-size 4096 --max-running-requests 32 --json-model-override-args '{"num_hidden_layers": 12}' --enable-return-hidden-states --disable-cuda-graph --trust-remote-code --log-level info
