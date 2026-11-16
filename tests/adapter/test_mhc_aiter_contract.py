# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AITER mHC provider contracts (CPU/mock).

Run: HCU_ADAPTER_HOST=none python -m pytest tests/adapter/test_mhc_aiter_contract.py
No GPU hardware, framework math, or installation state is exercised.
"""

import pytest
import torch
import torch.nn.functional as F
import inspect
from unittest.mock import MagicMock
from hcu_adapter.kernels.mhc_aiter import MhcAiterProvider, MhcAiterRuntime

@pytest.fixture
def mock_kernels():
    return MagicMock()

@pytest.fixture
def mock_provider():
    p = MagicMock(spec=MhcAiterProvider)
    p.available = True
    return p

def test_aiter_runtime_pre_selection(mock_provider, mock_kernels):
    # Test fallback when provider is partially absent
    mock_provider.pre = None
    mock_provider.project = None

    # Configure mock kernel to return dummy values
    mock_kernels.mhc_pre_torch.return_value = (torch.empty(1), torch.empty(1), torch.empty(1))

    runtime = MhcAiterRuntime(mock_provider, mock_kernels, F.rms_norm, None, None)

    # Mock inputs
    residual = torch.randn(1, 4, 1024, dtype=torch.bfloat16)
    fn = torch.randn(12, 1024, dtype=torch.float32)
    scale = torch.randn(3, dtype=torch.float32)
    base = torch.randn(12, dtype=torch.float32)

    # Should fallback to torch kernels because provider.pre is None
    runtime.pre(residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 1.0, 1)
    assert mock_kernels.mhc_pre_torch.called

def test_aiter_provider_signature_check():
    # Test ABI validation
    mock_mod = MagicMock()
    # Need a function with a different signature
    def mhc_pre_correct_sig(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat): pass
    def mhc_pre_wrong_sig(x): pass

    mock_mod.mhc_pre = mhc_pre_wrong_sig

    # Provider is constructed with the *correct* expected signature
    p = MhcAiterProvider(pre=mhc_pre_correct_sig, module=mock_mod,
                         signatures=(('mhc_pre', str(inspect.signature(mhc_pre_correct_sig))),))
    # require_healthy() checks module.mhc_pre (wrong) against p.pre (correct)
    with pytest.raises(RuntimeError, match="AITER mHC provider changed after publication"):
        p.require_healthy()

def test_split_k_selection():
    from hcu_adapter.kernels.mhc_aiter import select_splits
    assert select_splits(64, 4, 5120, 'auto') == 32
    assert select_splits(100, 4, 5120, 'auto') == 1
    with pytest.raises(ValueError):
        select_splits(100, 4, 5120, '999')


def _stage2_inputs(tokens=2):
    residual = torch.randn(tokens, 4, 1024, dtype=torch.bfloat16)
    fn = torch.randn(24, 4096, dtype=torch.float32)
    scale = torch.randn(3, dtype=torch.float32)
    base = torch.randn(24, dtype=torch.float32)
    return residual, fn, scale, base


def _call_stage2(runtime, inputs):
    runtime._staged_pre(*inputs, 1e-6, 1e-6, 1e-6, 1.0, 20, 1)


def test_tilelang_stage2_replaces_legacy_big_fuse(mock_kernels):
    provider = MhcAiterProvider(
        project=MagicMock(), big_fuse=MagicMock(), tilelang_pre=MagicMock(),
        tilelang_enabled=True)
    runtime = MhcAiterRuntime(provider, mock_kernels, F.rms_norm, MagicMock(), MagicMock())

    _call_stage2(runtime, _stage2_inputs())

    provider.project.assert_called_once()
    provider.tilelang_pre.assert_called_once()
    provider.big_fuse.assert_not_called()
    args = provider.tilelang_pre.call_args.args
    assert args[0].is_contiguous() and args[0].shape == (1, 2, 24)
    assert args[8] == 1024
    assert provider.tilelang_pre.call_args.kwargs == {'n_splits': 1, 'mhc_mult': 4}


def test_explicit_tilelang_opt_out_preserves_legacy_big_fuse(mock_kernels):
    provider = MhcAiterProvider(
        project=MagicMock(), big_fuse=MagicMock(), tilelang_pre=MagicMock(),
        tilelang_enabled=False)
    runtime = MhcAiterRuntime(provider, mock_kernels, F.rms_norm, MagicMock(), MagicMock())

    _call_stage2(runtime, _stage2_inputs())

    provider.big_fuse.assert_called_once()
    provider.tilelang_pre.assert_not_called()


def test_enabled_but_missing_tilelang_uses_torch_not_big_fuse(
        mock_kernels, monkeypatch):
    provider = MhcAiterProvider(
        project=MagicMock(), big_fuse=MagicMock(), tilelang_pre=None,
        tilelang_enabled=True)
    assert not provider.stage2_ready and not provider.available
    runtime = MhcAiterRuntime(provider, mock_kernels, F.rms_norm, MagicMock(), MagicMock())
    monkeypatch.setattr(runtime, '_pre_supported', lambda *args: True)
    residual, fn, scale, base = _stage2_inputs()
    reference = (
        residual.new_empty((2, 4, 1), dtype=torch.float32),
        residual.new_empty((2, 4, 4), dtype=torch.float32),
        residual[:, 0].clone(),
        residual.new_empty((2, 4), dtype=torch.float32),
    )
    mock_kernels.mhc_pre_delayed_torch.return_value = reference

    runtime.delayed(residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 1.0, 20)

    mock_kernels.mhc_pre_delayed_torch.assert_called_once()
    provider.project.assert_not_called()
    provider.big_fuse.assert_not_called()
