# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""LightOp per-token FP8 fast path: contract, fallbacks and idempotency.

CPU only: the LightOp kernel is replaced by a fake module so the rebind and
every fallback branch can be exercised without a device.
"""

import sys
import types

import pytest
import torch

from hcu_adapter.bridges import vllm_input_quant_lightop as bridge


class _GroupShape:
    PER_TOKEN = object()
    PER_TENSOR = object()
    PER_CHANNEL = object()


def _upstream(self, x, scale=None, scale_ub=None, use_triton=False):
    return ("upstream", x, scale, scale_ub, use_triton)


def _make_module(with_marker=False):
    owner = type("QuantFP8", (), {"forward_cuda": _upstream})
    module = types.SimpleNamespace(
        QuantFP8=owner,
        GroupShape=_GroupShape,
        _FP8_DTYPE=torch.float8_e4m3fn,
        _FP8_MIN_SCALING_FACTOR=1.0 / (448.0 * 512.0),
    )
    if with_marker:
        setattr(owner, bridge._MARKER, True)
    return module


def _instance(**overrides):
    values = {
        "is_group_quant": False,
        "group_shape": _GroupShape.PER_TOKEN,
        "num_token_padding": None,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _activate(monkeypatch, scale_shape="2d"):
    calls = []

    def per_token_quant_fp8(x, dtype=None):
        calls.append((x.shape, dtype))
        quantized = x.to(torch.float8_e4m3fn)
        scales = torch.full((x.shape[0],), 0.5, dtype=torch.float32)
        if scale_shape == "2d":
            scales = scales.view(-1, 1)
        return quantized, scales

    package = types.ModuleType("lightop")
    quant = types.ModuleType("lightop.quant")
    quant.per_token_quant_fp8 = per_token_quant_fp8
    package.quant = quant
    monkeypatch.setitem(sys.modules, "lightop", package)
    monkeypatch.setitem(sys.modules, "lightop.quant", quant)
    return calls


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    bridge._COUNTS.clear()
    bridge._LOGGED.clear()
    bridge._DIAG.update(calls=0, rows=0, tiny=0, floor=0.0)
    monkeypatch.delenv(bridge._AMAX_DIAG_ENV, raising=False)
    monkeypatch.delenv(bridge._GRAPH_ENV, raising=False)
    monkeypatch.setenv("HCU_LIGHTOP_PER_TOKEN_FP8", "1")
    # Fallback tests select the public override explicitly; capture is now on
    # by default. The unset/explicit-on dispatch contracts are tested below.
    monkeypatch.setenv(bridge._GRAPH_ENV, "0")
    monkeypatch.setattr(bridge, '_device_ok', lambda x: True)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    yield
    bridge._COUNTS.clear()
    bridge._LOGGED.clear()
    bridge._DIAG.update(calls=0, rows=0, tiny=0, floor=0.0)


def test_install_rebinds_and_hits(monkeypatch):
    calls = _activate(monkeypatch)
    module = _make_module()

    assert bridge.install(module) is True
    assert getattr(module.QuantFP8, bridge._MARKER) is True

    x = torch.randn(4, 8, dtype=torch.bfloat16)
    quantized, scales = module.QuantFP8.forward_cuda(_instance(), x)

    assert len(calls) == 1 and calls[0] == (x.shape, torch.float8_e4m3fn)
    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.shape == (4, 1)  # upstream contract is [M, 1]
    assert bridge._COUNTS == {"hit": 1}


def test_install_is_idempotent(monkeypatch):
    _activate(monkeypatch)
    module = _make_module()

    assert bridge.install(module) is True
    rebinded = module.QuantFP8.forward_cuda
    assert bridge.install(module) is False
    assert module.QuantFP8.forward_cuda is rebinded


def test_existing_marker_skips_install(monkeypatch):
    _activate(monkeypatch)
    module = _make_module(with_marker=True)

    assert bridge.install(module) is False
    assert module.QuantFP8.forward_cuda is _upstream


def test_default_is_on_installs(monkeypatch):
    _activate(monkeypatch)
    monkeypatch.delenv("HCU_LIGHTOP_PER_TOKEN_FP8", raising=False)

    module = _make_module()
    assert bridge.install(module) is True
    assert getattr(module.QuantFP8, bridge._MARKER) is True


def test_kill_switch_skips_install(monkeypatch):
    _activate(monkeypatch)
    monkeypatch.setenv("HCU_LIGHTOP_PER_TOKEN_FP8", "0")

    assert bridge.install(_make_module()) is False


def test_signature_drift_skips_install(monkeypatch):
    _activate(monkeypatch)
    module = _make_module()

    def reordered(self, x, use_triton=False, scale=None, scale_ub=None):
        return _upstream(self, x, scale, scale_ub, use_triton)

    module.QuantFP8.forward_cuda = reordered
    assert bridge.install(module) is False


@pytest.mark.parametrize(
    "kwargs, tensor, reason",
    [
        ({"scale": torch.zeros(1)}, torch.zeros(2, 4, dtype=torch.bfloat16),
         "static_scale"),
        ({"scale_ub": torch.zeros(1)}, torch.zeros(2, 4, dtype=torch.bfloat16),
         "scale_ub"),
        ({"is_group_quant": True}, torch.zeros(2, 4, dtype=torch.bfloat16),
         "group"),
        ({"group_shape": _GroupShape.PER_TENSOR}, torch.zeros(2, 4, dtype=torch.bfloat16),
         "not_per_token"),
        ({"num_token_padding": 128}, torch.zeros(2, 4, dtype=torch.bfloat16),
         "token_padding"),
        ({}, torch.zeros(2, 4, dtype=torch.float32), "dtype"),
        ({}, torch.zeros(2, dtype=torch.bfloat16), "not_2d"),
    ],
)
def test_fallback_reasons(monkeypatch, kwargs, tensor, reason):
    calls = _activate(monkeypatch)
    module = _make_module()
    assert bridge.install(module) is True

    kwargs = dict(kwargs)
    scale = kwargs.pop("scale", None)
    scale_ub = kwargs.pop("scale_ub", None)
    out = module.QuantFP8.forward_cuda(_instance(**kwargs), tensor, scale, scale_ub)

    assert out[0] == "upstream"
    assert calls == []
    assert bridge._COUNTS == {reason: 1}


@pytest.mark.parametrize('capture', [False, True])
def test_non_device_or_disabled_capture_uses_original(monkeypatch, capture):
    calls = _activate(monkeypatch)
    monkeypatch.setenv(bridge._GRAPH_ENV, "0")
    module = _make_module()
    assert bridge.install(module)
    monkeypatch.setattr(bridge, '_device_ok', lambda x: capture)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: capture)
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    assert module.QuantFP8.forward_cuda(_instance(), x)[0] == 'upstream'
    assert not calls
    assert bridge._COUNTS == {'capture' if capture else 'device': 1}


@pytest.mark.parametrize('setting', [None, '1'])
def test_default_and_explicit_graph_enable_use_lightop(monkeypatch, setting):
    calls = _activate(monkeypatch)
    if setting is None:
        monkeypatch.delenv(bridge._GRAPH_ENV)
    else:
        monkeypatch.setenv(bridge._GRAPH_ENV, setting)
    module = _make_module()
    assert bridge.install(module)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    quantized, scales = module.QuantFP8.forward_cuda(_instance(), x)
    assert quantized.dtype == torch.float8_e4m3fn and scales.shape == (2, 1)
    assert len(calls) == 1 and bridge._COUNTS == {'hit': 1}


def test_capture_provider_failure_propagates_without_upstream_fallback(monkeypatch):
    _activate(monkeypatch)
    monkeypatch.delenv(bridge._GRAPH_ENV)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    error = RuntimeError("capture provider failure")

    def fail(x, dtype=None):
        raise error

    monkeypatch.setattr(sys.modules["lightop.quant"], "per_token_quant_fp8", fail)
    module = _make_module()
    assert bridge.install(module)
    with pytest.raises(RuntimeError) as caught:
        module.QuantFP8.forward_cuda(_instance(), torch.ones(2, 8, dtype=torch.bfloat16))
    assert caught.value is error
    assert bridge._COUNTS == {}


@pytest.mark.parametrize("graph_setting", [None, "0", "1"])
def test_capture_contract_metadata_matches_runtime(monkeypatch, graph_setting):
    calls = _activate(monkeypatch)
    if graph_setting is None:
        monkeypatch.delenv(bridge._GRAPH_ENV, raising=False)
    else:
        monkeypatch.setenv(bridge._GRAPH_ENV, graph_setting)
    module = _make_module()
    assert bridge.install(module)
    method = module.QuantFP8.forward_cuda
    assert method._hcu_token_fp8_backend == "lightop"
    enabled = graph_setting != "0"
    assert method._hcu_token_fp8_graph_enabled is enabled
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    output = method(_instance(), torch.ones(2, 8, dtype=torch.bfloat16))
    if enabled:
        assert len(calls) == 1
        assert output[0].dtype == torch.float8_e4m3fn
    else:
        assert output[0] == "upstream"
        assert calls == []


def test_non_contiguous_falls_back(monkeypatch):
    calls = _activate(monkeypatch)
    module = _make_module()
    assert bridge.install(module) is True

    non_contiguous = torch.zeros(4, 16, dtype=torch.bfloat16)[:, ::2]
    assert not non_contiguous.is_contiguous()

    out = module.QuantFP8.forward_cuda(_instance(), non_contiguous)
    assert out[0] == "upstream"
    assert calls == []
    assert bridge._COUNTS == {"non_contiguous": 1}


def test_lightop_failure_falls_back(monkeypatch):
    def exploding(x, dtype=None):
        raise RuntimeError("lightop kernel failure")

    package = types.ModuleType("lightop")
    quant = types.ModuleType("lightop.quant")
    quant.per_token_quant_fp8 = exploding
    package.quant = quant
    monkeypatch.setitem(sys.modules, "lightop", package)
    monkeypatch.setitem(sys.modules, "lightop.quant", quant)

    module = _make_module()
    assert bridge.install(module) is True
    out = module.QuantFP8.forward_cuda(_instance(), torch.zeros(2, 4, dtype=torch.bfloat16))

    assert out[0] == "upstream"
    assert bridge._COUNTS == {"exception": 1}


def test_missing_lightop_skips_install(monkeypatch):
    package = types.ModuleType("lightop")
    quant = types.ModuleType("lightop.quant")  # no per_token_quant_fp8
    package.quant = quant
    monkeypatch.setitem(sys.modules, "lightop", package)
    monkeypatch.setitem(sys.modules, "lightop.quant", quant)

    module = _make_module()
    assert bridge.install(module) is False
    assert module.QuantFP8.forward_cuda is _upstream


def test_scale_1d_is_reshaped(monkeypatch):
    _activate(monkeypatch, scale_shape="1d")
    module = _make_module()
    assert bridge.install(module) is True

    _, scales = module.QuantFP8.forward_cuda(
        _instance(), torch.zeros(3, 4, dtype=torch.bfloat16)
    )
    assert scales.shape == (3, 1)


def test_amax_diagnostic_counts_tiny_rows(monkeypatch):
    _activate(monkeypatch)
    monkeypatch.setenv(bridge._AMAX_DIAG_ENV, "1")
    module = _make_module()
    assert bridge.install(module) is True

    x = torch.randn(8, 4, dtype=torch.bfloat16)
    x[0] = 0.0            # all-zero row
    x[1] = 1e-6           # below the 1/(448*512) clamp floor
    module.QuantFP8.forward_cuda(_instance(), x)

    assert bridge._DIAG["calls"] == 1
    assert bridge._DIAG["rows"] == 8
    assert bridge._DIAG["tiny"] == 2
    assert bridge._DIAG["floor"] == pytest.approx(1.0 / (448.0 * 512.0))


def test_amax_diagnostic_off_by_default(monkeypatch):
    _activate(monkeypatch)
    module = _make_module()
    assert bridge.install(module) is True

    module.QuantFP8.forward_cuda(_instance(), torch.zeros(4, 4, dtype=torch.bfloat16))
    assert bridge._DIAG["calls"] == 0
