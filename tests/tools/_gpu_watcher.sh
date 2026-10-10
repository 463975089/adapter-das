#!/bin/bash
export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:$LD_LIBRARY_PATH
/opt/dtk/bin/rocm-smi --showmemuse 2>/dev/null | grep -F 'HCU['
docker ps --format '{{.Names}}' | grep -c zhuwenbo_sgl_qwen
ps aux | grep -c 'sglang' 
