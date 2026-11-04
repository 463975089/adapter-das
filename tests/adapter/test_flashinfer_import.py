# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process shadow import orders and norm forwarding, without a GPU.

Run: HCU_ADAPTER_HOST=none HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES=''
     PYTHONPATH=$PWD python -m pytest -q tests/adapter/test_flashinfer_import.py
"""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('order', ['adapter-first', 'kernel-first', 'public-first'])
def test_flashinfer_import_order_and_existing_contract(order):
    root = Path(__file__).resolve().parents[2]
    code = '''
import sys
order = sys.argv[1]
if order == 'adapter-first':
    import hcu_adapter.flashinfer
elif order == 'kernel-first':
    import sgl_kernel
else:
    import sgl_kernel
    import flashinfer

import hcu_adapter.flashinfer as adapter
adapter.register()
adapter.register()
assert sum(isinstance(f, adapter._FlashInferFinder) for f in sys.meta_path) == 1
from flashinfer.norm import rmsnorm, fused_add_rmsnorm
assert rmsnorm is adapter.rmsnorm
assert fused_add_rmsnorm is adapter.fused_add_rmsnorm

from flashinfer import BatchDecodeWithPagedKVCacheWrapper
try:
    BatchDecodeWithPagedKVCacheWrapper()
except NotImplementedError as error:
    assert 'flashinfer' in str(error)
    assert 'BatchDecodeWithPagedKVCacheWrapper' in str(error)
else:
    raise AssertionError('missing backend silently succeeded')

import torch
torch.manual_seed(49)
x = torch.randn(3, 34)[:, ::2]
weight = torch.randn(17)
out = torch.empty_like(x)
eps = .03125
expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight
assert rmsnorm(x, weight, eps=eps, out=out, enable_pdl=True) is out
torch.testing.assert_close(out, expected)
residual = torch.randn_like(x)
total = residual + x
expected = total * torch.rsqrt(total.square().mean(-1, keepdim=True) + eps) * weight
assert fused_add_rmsnorm(x, residual, weight, eps=eps, enable_pdl=True) is None
torch.testing.assert_close(residual, total)
torch.testing.assert_close(x, expected)
assert not any(n == h or n.startswith(h + '.') for n in sys.modules for h in ('vllm','sglang'))
'''
    env = dict(os.environ, HCU_ADAPTER_HOST='none', HIP_VISIBLE_DEVICES='',
               CUDA_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
               PYTHONPATH=str(root), PYTHONNOUSERSITE='1')
    result = subprocess.run([sys.executable, '-B', '-c', code, order],
                            cwd=root, env=env, capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
