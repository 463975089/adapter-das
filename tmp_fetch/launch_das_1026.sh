#!/bin/bash
export PYTHONPATH=/root/hcu_v0522:/sglang-v0522/python
unset SGLANG_DSV4_FP4_DEQUANT SGLANG_OPT_USE_TOPK_V2 HCU_TOPK_V2_RAGGED SGLANG_DSA_FUSE_TOPK SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND SGLANG_OPT_USE_TILELANG_MHC_PRE SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE ROCM_HOME
exec python3 -m sglang.launch_server --model-path /models/dsv41f \
  --host 0.0.0.0 --port 30296 --tp-size 8 \
  --quantization w8a8_fp8 \
  --trust-remote-code --language-model-only \
  --mem-fraction-static 0.78 --max-running-requests 16 \
  --cuda-graph-max-bs-decode 16 --disable-radix-cache \
  > /root/logs/srv_das_1026.log 2>&1
