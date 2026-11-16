# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Exercise public mHC runtime entries, including competing native capability.

Only GPU eligibility and provider math are doubled. These are dispatch tests,
not a GPU numeric qualification of TileLang.
"""
import os
import subprocess
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from hcu_adapter.kernels import mhc_aiter as mhc


@pytest.mark.parametrize('entry', ['pre', 'fused', 'delayed'])
@pytest.mark.parametrize('tokens', [1, 3, 16])
@pytest.mark.parametrize('capability', ['ready', 'missing_tilelang', 'missing_project'])
def test_selected_tilelang_policy_cannot_be_bypassed(monkeypatch, entry, tokens, capability):
    # Correct production shapes; CPU doubles replace only the device boundary.
    residual = torch.ones(tokens, 4, 1024, dtype=torch.bfloat16)
    fn = torch.ones(24, 4096)
    scale = torch.ones(3)
    base = torch.ones(24)
    monkeypatch.setattr(mhc, '_gpu_tensors', lambda tensors: True)
    native = Mock(side_effect=AssertionError('legacy native pre must not run'))
    legacy = Mock(side_effect=AssertionError('legacy big_fuse must not run'))
    events = []

    def project(gemm, sqrsum, *args):
        events.append('project')
        gemm.fill_(1)
        sqrsum.fill_(1)

    def stage2(gemm, sqrsum, scale, base, residual, post, comb, layer, *args, **kwargs):
        events.append('tilelang')
        post.fill_(2)
        comb.fill_(3)
        layer.fill_(4)

    def reference(*args, **kwargs):
        events.append('torch')
        outputs = (torch.full((tokens, 4, 1), 2.0),
                   torch.full((tokens, 4, 4), 3.0), residual[:, 0].clone())
        return (*outputs, torch.ones(tokens, 4)) if entry == 'delayed' else outputs

    provider = mhc.MhcAiterProvider(
        pre=native, project=None if capability == 'missing_project' else project,
        big_fuse=legacy, tilelang_enabled=True,
        tilelang_pre=None if capability == 'missing_tilelang' else stage2)
    kernels = NS(mhc_pre_torch=reference, mhc_pre_delayed_torch=reference,
                 mhc_post_torch=lambda x, residual, *args: residual.clone())
    runtime = mhc.MhcAiterRuntime(provider, kernels, None,
                                lambda *args: torch.ones(tokens, 4),
                                lambda residual, mix: residual[:, 0].clone())
    args = (fn, scale, base, 1e-6, 1e-6, 1e-6, 1.0, 20)
    if entry == 'fused':
        runtime.fused(residual[:, 0], residual, torch.ones(tokens, 4, 1),
                      torch.ones(tokens, 4, 4), *args)
    else:
        getattr(runtime, entry)(residual, *args)
    assert events == (['project', 'tilelang'] if capability == 'ready' else ['torch'])
    native.assert_not_called()
    legacy.assert_not_called()


def test_explicit_tilelang_opt_out_keeps_legacy_native_pre(monkeypatch):
    monkeypatch.setattr(mhc, '_gpu_tensors', lambda tensors: True)
    residual = torch.ones(3, 4, 1024, dtype=torch.bfloat16)
    native = Mock(return_value=(torch.ones(3, 4, 1), torch.ones(3, 4, 4), residual[:, 0]))
    provider = mhc.MhcAiterProvider(pre=native, tilelang_enabled=False)
    runtime = mhc.MhcAiterRuntime(provider, None, None, None, None)
    runtime.pre(residual, torch.ones(24, 4096), torch.ones(3), torch.ones(24),
                1e-6, 1e-6, 1e-6, 1.0, 20)
    native.assert_called_once()


def test_routing_never_initialises_cuda_in_an_isolated_process():
    """The isolation guarantee cannot live in the shared pytest process.

    L1 collects tests/kernels and tests/adapter together, so a device test that
    ran earlier leaves CUDA initialised and no CPU-only case can restore False.
    The property is checked where it is meaningful: a fresh interpreter with the
    GPUs hidden.
    """
    program = """
import torch
from unittest.mock import Mock
from types import SimpleNamespace as NS
from hcu_adapter.kernels import mhc_aiter as mhc

mhc._gpu_tensors = lambda tensors: True
events = []
def project(gemm, sqrsum, *a):
    events.append('project'); gemm.fill_(1); sqrsum.fill_(1)
def stage2(gemm, sqrsum, scale, base, residual, post, comb, layer, *a, **k):
    events.append('tilelang'); post.fill_(2); comb.fill_(3); layer.fill_(4)
provider = mhc.MhcAiterProvider(project=project, tilelang_pre=stage2,
                                tilelang_enabled=True)
runtime = mhc.MhcAiterRuntime(provider, NS(mhc_pre_torch=Mock(), mhc_post_torch=Mock()),
                              None, None, None)
r = torch.ones(3, 4, 1024, dtype=torch.bfloat16)
runtime.pre(r, torch.ones(24, 4096), torch.ones(3), torch.ones(24),
            1e-6, 1e-6, 1e-6, 1.0, 20)
assert events == ['project', 'tilelang'], events
assert not torch.cuda.is_initialized()
print('isolated-routing-ok')
"""
    environment = dict(os.environ, HCU_ADAPTER_HOST='none')
    for name in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        environment[name] = ''
    result = subprocess.run([sys.executable, '-B', '-c', program], env=environment,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith('isolated-routing-ok')
