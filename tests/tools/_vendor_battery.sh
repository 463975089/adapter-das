#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# vendor test battery, run inside container at /adapter (nmz26, gfx938)
set -o pipefail
export PYTHONPATH=/adapter
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
cd /adapter

echo "=== arch ==="
python -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName)"
echo "=== vendor presence ==="
python - <<'EOF'
try:
    import aiter
    print("aiter:", getattr(aiter, "__file__", "?"))
    print("  flash_attn_varlen_func:", hasattr(aiter, "flash_attn_varlen_func"))
    from aiter.ops.triton import custom_all_reduce as car
    print("  custom_all_reduce:", [n for n in dir(car) if "reduce" in n.lower()][:6])
except Exception as e:
    print("aiter import fail:", repr(e))
try:
    import lightop
    print("lightop:", getattr(lightop, "__version__", "?"))
except Exception as e:
    print("lightop import fail:", repr(e))
EOF
echo "=== [1/5] cpu suite ==="
python -m pytest tests/test_ops_cpu.py -q 2>&1 | tail -2
echo "=== [2/5] gpu kernel suite ==="
python -m pytest tests/kernels -q 2>&1 | tail -3
echo "=== [3/5] fa scenario matrix ==="
python -m pytest tests/vendor/test_flash_attn.py -q --no-header 2>&1 | tail -5
echo "=== [4/5] ar gate (3 states) ==="
python -m pytest tests/vendor/test_allreduce.py -q --no-header -k gate 2>&1 | tail -5
echo "=== [5/5] ar stress (torchrun 2gpu) ==="
python -m pytest tests/vendor/test_allreduce.py -q --no-header -k "stress and not gate" 2>&1 | tail -8
echo "=== triage env stage ==="
python tests/tools/run_triage.py --stage env 2>&1 | tail -12
echo "=== DONE ==="
