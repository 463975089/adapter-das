# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""AR+RMSNorm install idempotence for legacy and v0.5.22 seams."""

import sys
import types

import pytest

from hcu_adapter import ar_norm_aiter


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("preinstalled", [False, True])
def test_repeated_install_preserves_dispatch_and_candidate_arm(monkeypatch, legacy, preinstalled):
    def module(name):
        parent, _, child = name.rpartition(".")
        if parent and parent not in modules:
            module(parent)
        value = types.ModuleType(name)
        value.__path__ = []
        modules[name] = value
        monkeypatch.setitem(sys.modules, name, value)
        if parent:
            setattr(modules[parent], child, value)
        return value

    modules = {}
    module("torch").version = types.SimpleNamespace(hip="fixture")
    module("aiter.dist.device_communicators.custom_all_reduce").CustomAllreduce = object
    fcf = module("sglang.srt.layers.flashinfer_comm_fusion")
    add_norm = module("sglang.srt.layers.layer_boundary.residual.add_norm")
    lb_exit = module("sglang.srt.layers.layer_boundary.exit")
    lb_fusion = module("sglang.srt.layers.layer_boundary.fusions.allreduce")
    module("sglang.srt.distributed").get_tp_group = lambda: types.SimpleNamespace(world_size=2, ca_comm=None)
    module("sglang.srt.runtime_context").get_parallel = lambda: types.SimpleNamespace(moe_ep_size=1, moe_tp_size=2)
    add_norm.is_dp_attention_enabled = lambda: False
    stock = lambda *args, **kwargs: ("stock-output", "stock-residual")
    stock._hcu_ar_norm_aiter = preinstalled
    fcf.flashinfer_allreduce_residual_rmsnorm = stock
    if legacy:
        for mod in (add_norm, lb_exit, lb_fusion):
            mod.apply_flashinfer_allreduce_fusion = lambda batch_size: False
    monkeypatch.setattr(ar_norm_aiter.hcu_env, "resolve", lambda name: ("aiter", None))
    registrations = []
    monkeypatch.setattr(ar_norm_aiter, "register", lambda *args, **kwargs: registrations.append((args, kwargs)))

    ar_norm_aiter.install()
    installed = fcf.flashinfer_allreduce_residual_rmsnorm
    ar_norm_aiter.install()
    assert fcf.flashinfer_allreduce_residual_rmsnorm is installed
    assert installed._hcu_ar_norm_aiter
    assert installed(None, None, None) == ("stock-output", "stock-residual")
    assert len(registrations) == 2
    assert registrations[-1][0][1] is installed
    if legacy:
        arm = add_norm.apply_flashinfer_allreduce_fusion
        assert arm is lb_exit.apply_flashinfer_allreduce_fusion
        assert arm is lb_fusion.apply_flashinfer_allreduce_fusion
        assert arm(1) is True
        assert arm(0) is False
        assert "legacy candidate arm retained" in registrations[-1][1]["detail"]
    else:
        assert not hasattr(add_norm, "apply_flashinfer_allreduce_fusion")
        assert "dispatch-only" in registrations[-1][1]["detail"]
