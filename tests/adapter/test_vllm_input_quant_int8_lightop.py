# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""LightOp per-token INT8 fast path: contract, fallbacks and idempotency.

CPU only: the LightOp native op is replaced by a fake module so the rebind and
every fallback branch can be exercised without a device.
"""

import logging
import sys
import types

import pytest
import torch

from hcu_adapter.bridges import vllm_input_quant_int8_lightop as bridge

HIDDEN = 1280  # real MLA width, %16 == 0 and <= 8192


def _upstream(input, scale=None, azp=None, symmetric=True):
    return ("upstream", input, scale, azp, symmetric)


def _make_module(with_marker=False):
    module = types.SimpleNamespace(scaled_int8_quant=_upstream)
    if with_marker:
        setattr(module, bridge._MARKER, True)
    return module


def _activate(monkeypatch, exploding=False):
    calls = []

    def per_token_dynamic_quant_int8(out, inp, scales, smooth, round_after_smooth,
                                     limit):
        calls.append(
            {
                "shape": tuple(inp.shape),
                "out_dtype": out.dtype,
                "scales_shape": tuple(scales.shape),
                "smooth": (smooth.numel(), smooth.dtype, bool(smooth.eq(1).all())),
                "round_after_smooth": round_after_smooth,
                "limit": limit,
            }
        )
        if exploding:
            raise RuntimeError("lightop kernel failure")
        out.fill_(1)
        scales.fill_(0.25)

    package = types.ModuleType("lightop")
    op = types.ModuleType("lightop.op")
    op.per_token_dynamic_quant_int8 = per_token_dynamic_quant_int8
    package.op = op
    monkeypatch.setitem(sys.modules, "lightop", package)
    monkeypatch.setitem(sys.modules, "lightop.op", op)
    return calls


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    bridge._COUNTS.clear()
    bridge._LOGGED.clear()
    bridge._SMOOTH.clear()
    monkeypatch.setattr(bridge, '_OP', None)
    monkeypatch.setattr(bridge, '_BINDING', None)
    monkeypatch.setenv(bridge._ENV, "1")
    monkeypatch.setattr(bridge, '_execution_fallback', lambda x: None)
    monkeypatch.setattr(bridge, '_current_stream_id', lambda device: -1)
    # CPU tests cannot allocate accelerator tensors; the device gate is
    # exercised separately in test_cpu_activation_falls_back.
    monkeypatch.setattr(bridge, "_device_ok", lambda x: True)
    yield
    bridge._COUNTS.clear()
    bridge._LOGGED.clear()
    bridge._SMOOTH.clear()
    bridge._OP = None


def _x(m=4, hidden=HIDDEN):
    return torch.randn(m, hidden, dtype=torch.bfloat16)


def test_install_rebinds_and_hits(monkeypatch):
    calls = _activate(monkeypatch)
    module = _make_module()

    assert bridge.install(module) is True
    assert getattr(module, bridge._MARKER) is True

    x = _x()
    quantized, scales, azp = module.scaled_int8_quant(x)

    assert len(calls) == 1
    assert calls[0]["shape"] == (4, HIDDEN)
    assert calls[0]["out_dtype"] == torch.int8
    assert calls[0]["scales_shape"] == (4, 1)  # upstream contract is [M, 1]
    assert calls[0]["smooth"] == (HIDDEN, torch.float32, True)
    assert calls[0]["round_after_smooth"] is False
    assert calls[0]["limit"] == -1.0
    assert quantized.dtype == torch.int8 and quantized.eq(1).all()
    assert scales.shape == (4, 1) and scales.eq(0.25).all()
    assert azp is None
    assert bridge._COUNTS == {"hit": 1}


def test_hit_logs_once_at_info(monkeypatch, caplog):
    _activate(monkeypatch)
    module = _make_module()
    assert bridge.install(module)
    x = _x()

    with caplog.at_level(logging.INFO, logger=bridge.LOG.name):
        module.scaled_int8_quant(x)
        bridge._COUNTS["hit"] = 1023
        module.scaled_int8_quant(x)

    hits = [record for record in caplog.records
            if "per-token INT8 first hit:" in record.message
            or "per-token INT8 progress:" in record.message]
    assert len(hits) == 1
    assert hits[0].levelno == logging.INFO
    assert "hit=1 fallback=0" in hits[0].message
    assert bridge._COUNTS == {"hit": 1024}


def test_smooth_scale_is_cached_per_shape(monkeypatch):
    _activate(monkeypatch)
    module = _make_module()
    assert bridge.install(module) is True

    module.scaled_int8_quant(_x())
    module.scaled_int8_quant(_x())
    assert len(bridge._SMOOTH) == 1

    module.scaled_int8_quant(_x(hidden=4096))
    assert len(bridge._SMOOTH) == 2


def test_install_is_idempotent(monkeypatch):
    _activate(monkeypatch)
    module = _make_module()

    assert bridge.install(module) is True
    rebinded = module.scaled_int8_quant
    assert bridge.install(module) is False
    assert module.scaled_int8_quant is rebinded


def test_existing_marker_skips_install(monkeypatch):
    _activate(monkeypatch)
    module = _make_module(with_marker=True)

    assert bridge.install(module) is False
    assert module.scaled_int8_quant is _upstream


def test_default_is_on(monkeypatch):
    _activate(monkeypatch)
    monkeypatch.delenv(bridge._ENV, raising=False)

    assert bridge.install(_make_module()) is True


def test_graph_capture_uses_original_and_exposes_exact_binding(monkeypatch):
    calls = _activate(monkeypatch)
    module = _make_module()
    assert bridge.install(module)
    assert bridge.native_quant_binding(module) is _upstream
    monkeypatch.setattr(bridge, '_execution_fallback', lambda x: 'capture')
    assert module.scaled_int8_quant(_x())[0] == 'upstream'
    assert not calls and not bridge._SMOOTH

    foreign = bridge._wrap(_upstream)
    module.scaled_int8_quant = foreign
    assert bridge.native_quant_binding(module) is foreign


def test_smooth_cache_is_separate_per_stream(monkeypatch):
    stream = [1]
    monkeypatch.setattr(bridge, '_current_stream_id', lambda device: stream[0])
    device = torch.device('cpu')
    first = bridge._smooth_scale(HIDDEN, device)
    stream[0] = 2
    assert bridge._smooth_scale(HIDDEN, device) is not first
    stream[0] = 1
    assert bridge._smooth_scale(HIDDEN, device) is first


def test_kill_switch_skips_install(monkeypatch):
    _activate(monkeypatch)
    monkeypatch.setenv(bridge._ENV, "0")

    assert bridge.install(_make_module()) is False


def test_signature_drift_skips_install(monkeypatch):
    _activate(monkeypatch)
    module = _make_module()

    def reordered(input, symmetric=True, scale=None, azp=None):
        return _upstream(input, scale, azp, symmetric)

    module.scaled_int8_quant = reordered
    assert bridge.install(module) is False


def test_missing_lightop_skips_install(monkeypatch):
    package = types.ModuleType("lightop")
    op = types.ModuleType("lightop.op")  # no per_token_dynamic_quant_int8
    package.op = op
    monkeypatch.setitem(sys.modules, "lightop", package)
    monkeypatch.setitem(sys.modules, "lightop.op", op)

    module = _make_module()
    assert bridge.install(module) is False
    assert module.scaled_int8_quant is _upstream


@pytest.mark.parametrize(
    "kwargs, tensor, reason",
    [
        ({"scale": torch.zeros(1)}, _x(), "static_scale"),
        ({"azp": torch.zeros(1, dtype=torch.int32)}, _x(), "azp"),
        ({"symmetric": False}, _x(), "asymmetric"),
        ({}, torch.zeros(4, dtype=torch.bfloat16), "not_2d"),
        ({}, torch.zeros(3, 4, 3, dtype=torch.bfloat16), "not_2d"),
        ({}, _x().float(), "dtype"),
        ({}, _x(m=0), "empty"),
        ({}, _x(hidden=40), "hidden"),      # %16 != 0 (40 = 2*16 + 8)
        ({}, _x(hidden=8256), "hidden"),    # > 8192
        ({}, "not a tensor", "not_tensor"),
    ],
)
def test_fallback_reasons(monkeypatch, kwargs, tensor, reason):
    calls = _activate(monkeypatch)
    module = _make_module()
    assert bridge.install(module) is True

    out = module.scaled_int8_quant(tensor, **kwargs)

    assert out[0] == "upstream"
    assert calls == []
    assert bridge._COUNTS == {reason: 1}


def test_non_contiguous_falls_back(monkeypatch):
    calls = _activate(monkeypatch)
    module = _make_module()
    assert bridge.install(module) is True

    non_contiguous = torch.zeros(4, HIDDEN * 2, dtype=torch.bfloat16)[:, ::2]
    assert not non_contiguous.is_contiguous()

    out = module.scaled_int8_quant(non_contiguous)
    assert out[0] == "upstream"
    assert calls == []
    assert bridge._COUNTS == {"non_contiguous": 1}


def test_misaligned_falls_back(monkeypatch):
    calls = _activate(monkeypatch)
    module = _make_module()
    assert bridge.install(module) is True

    storage = torch.zeros(4 * HIDDEN + 1, dtype=torch.bfloat16)
    misaligned = storage[1:].view(4, HIDDEN)  # +2 bytes, still contiguous
    assert misaligned.is_contiguous()
    assert misaligned.data_ptr() % 16 != 0

    out = module.scaled_int8_quant(misaligned)
    assert out[0] == "upstream"
    assert calls == []
    assert bridge._COUNTS == {"misaligned": 1}


def test_lightop_failure_falls_back(monkeypatch):
    _activate(monkeypatch, exploding=True)
    module = _make_module()
    assert bridge.install(module) is True

    out = module.scaled_int8_quant(_x())
    assert out[0] == "upstream"
    assert bridge._COUNTS == {"exception": 1}


def test_fallback_forwards_arguments_verbatim(monkeypatch):
    _activate(monkeypatch)
    module = _make_module()
    assert bridge.install(module) is True

    x = _x()
    scale = torch.zeros(1)
    sentinel = module.scaled_int8_quant(x, scale=scale, symmetric=True)
    assert sentinel[0] == "upstream"
    assert sentinel[2] is scale


def test_counts_line_reports_hit_and_fallback(monkeypatch):
    _activate(monkeypatch)
    module = _make_module()
    assert bridge.install(module) is True

    module.scaled_int8_quant(_x())
    module.scaled_int8_quant(_x().float())
    line = bridge._counts_line()
    assert "hit=1" in line
    assert "fallback=1" in line
    assert "dtype=1" in line

def test_cpu_activation_falls_back(monkeypatch):
    """Without an accelerator the HCU op must not even be attempted."""
    calls = _activate(monkeypatch)
    module = _make_module()
    assert bridge.install(module) is True
    # undo the fixture override: use the real placement gate
    monkeypatch.setattr(
        bridge, "_device_ok", lambda x: bool(getattr(x, "is_cuda", False))
    )

    out = module.scaled_int8_quant(_x())
    assert out[0] == "upstream"
    assert calls == []
    assert bridge._COUNTS == {"device": 1}


def test_smooth_cache_creation_logged_once(monkeypatch):
    _activate(monkeypatch)
    module = _make_module()
    assert bridge.install(module) is True

    module.scaled_int8_quant(_x())
    created = [key for key in bridge._LOGGED if key.startswith("smooth:")]
    assert len(created) == 1

    module.scaled_int8_quant(_x())
    assert [key for key in bridge._LOGGED if key.startswith("smooth:")] == created
