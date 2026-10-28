#!/bin/bash
# prefill CP (attn-cp-size 2, zigzag) validated recipe -- nmz26 pr418_test
# model: Qwen3-30B-A3B on GPU6,7; baseline = same model TP2 without CP.
# 排查时可加: export HCU_PREFILL_CP_DEBUG=1 HCU_FA_EMU_DEBUG=1
export HIP_VISIBLE_DEVICES=6,7 CUDA_VISIBLE_DEVICES=6,7
export PYTHONPATH=/root/hcu_t2:/root/sglang-v0521-src/python
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export HCU_PREFILL_CP=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /root/logs
exec python -m sglang.launch_server   --model-path /public/opendas/DL_DATA/llm-models/qwen3/Qwen3-30B-A3B   --attention-backend fa3 --trust-remote-code   --tp-size 2 --attn-cp-size 2 --enable-prefill-cp --cp-strategy zigzag   --port 30201 --mem-fraction-static 0.80 --log-level info   > /root/logs/cp2_30b_30201.log 2>&1
