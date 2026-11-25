# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace

import pytest

from hcu_adapter.infra import registry


@pytest.mark.parametrize("legacy_layout", [False, True])
def test_numerics_binding_updates_cached_sglang_imports(monkeypatch, legacy_layout):
    pytest.importorskip("triton")
    from hcu_adapter.bridges import sglang_diffusion_numerics as bridge
    from hcu_adapter.kernels import diffusion_numerics_triton as kernels

    official = "sglang.kernels.numerics"
    legacy = "sglang.kernels.ops.diffusion.common.numerics"
    selected = legacy if legacy_layout else official
    upstream = ModuleType(selected)
    upstream.mul_rn_f32, upstream.div_rn_f32, upstream.rsqrt_approx_f32 = (
        object(), object(), object()
    )
    consumer = ModuleType("sglang.kernels.test_consumer")
    consumer.mul_rn_f32 = upstream.mul_rn_f32
    consumer.div_rn_f32 = object()
    foreign_division = consumer.div_rn_f32
    rsqrt_consumer = ModuleType("sglang.kernels.test_other_consumer")
    rsqrt_consumer.rsqrt_approx_f32 = upstream.rsqrt_approx_f32

    def resolve(path):
        if path != selected:
            raise ModuleNotFoundError(path, name=path)
        return upstream

    monkeypatch.setattr(bridge, "_has_supported_devices", lambda: True)
    monkeypatch.setattr(bridge, "import_module", resolve)
    for module in (upstream, consumer, rsqrt_consumer):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(registry, "_RECORDS", {})
    monkeypatch.setattr(registry, "_BOUND", {})

    assert bridge.install()
    assert upstream.mul_rn_f32 is consumer.mul_rn_f32 is kernels.mul_rn_f32
    assert upstream.div_rn_f32 is kernels.div_rn_f32
    assert consumer.div_rn_f32 is foreign_division
    assert upstream.rsqrt_approx_f32 is rsqrt_consumer.rsqrt_approx_f32 is kernels.rsqrt_approx_f32
    assert registry.get("diffusion_fp32_numerics")["target"] == selected
    assert bridge.install() is False


@pytest.mark.parametrize(
    "hip, architectures, expected",
    [
        ("test-hip", ("gfx936:sramecc+:xnack-", "gfx936"), True),
        ("test-hip", ("gfx938",), False),
        ("test-hip", ("gfx936", "gfx938"), False),
        ("test-hip", ("",), False),
        ("test-hip", (), False),
        (None, ("gfx936",), False),
    ],
)
def test_numerics_install_requires_all_visible_devices_to_be_gfx936(
    monkeypatch, caplog, hip, architectures, expected
):
    pytest.importorskip("triton")
    import torch

    from hcu_adapter.bridges import sglang_diffusion_numerics as bridge
    from hcu_adapter.infra import hip_metadata

    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(architectures))
    monkeypatch.setattr(
        hip_metadata, "device_properties",
        lambda index: SimpleNamespace(gcnArchName=architectures[index]),
    )
    calls = []
    monkeypatch.setattr(bridge, "_upstream_numerics", lambda: calls.append("resolve"))
    assert bridge.install() is False
    assert calls == (["resolve"] if expected else [])
    if hip and architectures and not expected:
        detected = ", ".join(arch or "unknown" for arch in architectures)
        assert f"detected: {detected}." in caplog.text
        assert "requires gfx936" in caplog.text
    else:
        assert not caplog.records
