#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Parallel single-card suite v2: official v0.5.21 + adapter, version logged.
# Lineup follows user guidance: fp8/int8/int4 first (bf16 only as small anchor).
export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:$LD_LIBRARY_PATH
LOGD=/root/logs/model_suite
mkdir -p $LOGD
export PYTHONPATH=/root/hcu_t2:/root/sglang-v0521-src/python
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export SGLANG_RUST_BUILD_MODE=never
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
M=/public/opendas/DL_DATA/llm-models

# log resolved sglang version once for audit ("we run upstream community")
python - <<'PEOF' > $LOGD/sglang_version.txt 2>&1
import sglang
print("sglang:", getattr(sglang, "__version__", "?"), sglang.__file__)
import sgl_kernel
print("adapter:", sgl_kernel.__version__, sgl_kernel.__file__)
PEOF
echo "[par] stack: $(cat $LOGD/sglang_version.txt | head -2 | tr '\n' ' ')"

launch() {  # $1 card, $2 port, $3 model-id, $4 model-path, $5 extra
  export HIP_VISIBLE_DEVICES=$1 CUDA_VISIBLE_DEVICES=$1
  echo "[par] $3 -> card $1 port $2"
  nohup python -m sglang.launch_server \
    --model-path "$4" --host 0.0.0.0 --port $2 \
    --page-size 64 --mem-fraction-static 0.85 --trust-remote-code \
    --log-level info $5 \
    > $LOGD/launch_$3.log 2>&1 &
}

launch 0 30036 q25-05b  $M/qwen2.5/Qwen2.5-0.5B-Instruct ""
launch 1 30046 q38-fp8  $M/qwen3.8/Qwen3.8-27B-FP8 ""
launch 2 30056 q3-30b   $M/qwen3/Qwen3-30B-A3B ""
launch 3 30066 glm5-i4  $M/GLM-5-Channel-INT4-w4a8 "--page-size 256 --json-model-override-args '{\"index_kpool\": 4}'"
echo "[par] all launched"
