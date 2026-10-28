#!/bin/bash
export PYTHONPATH=/root/hcu_v0522:/sglang-v0522/python
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export NCCL_TOPO_FILE=
export LD_LIBRARY_PATH=/opt/dtk/dcc/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/hip/lib:/opt/dtk/lib
export ROCM_HOME=/tmp/rocm22-mock
exec python3 -m sglang.launch_server --model-path /models_qua/DeepSeek-V4.1-Flash-1017-Channel-INT4-w4a8 \
  --host 0.0.0.0 --port 30297 --tp-size 8 \
  --trust-remote-code \
  --mem-fraction-static 0.85 --max-running-requests 16 \
  --cuda-graph-max-bs-decode 16 --disable-radix-cache --disable-custom-all-reduce --nccl-port 30513 \
  > /root/logs/srv_dsv41_chint4.log 2>&1
