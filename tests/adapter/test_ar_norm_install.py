# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""AR+RMSNorm admission must follow device-bound communicator creation."""

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
    torch = module("torch")
    torch.version = types.SimpleNamespace(hip="fixture")
    torch.cuda = types.SimpleNamespace(is_initialized=lambda: True, current_device=lambda: 3)
    # No AITER module: admission must reuse the host's real communicator.
    class GroupCoordinator:
        def __init__(self):
            self.world_size = 2
            self.device = types.SimpleNamespace(type="cuda", index=3)
            self.ca_comm = types.SimpleNamespace(disabled=False, custom_fused_ar_rms=lambda: None)

        def fused_allreduce_rmsnorm(self, *args):
            return None

    module("sglang.srt.distributed.parallel_state").GroupCoordinator = GroupCoordinator
    hooks = []
    registry = module("sglang.srt.plugins.hook_registry")
    registry.HookType = types.SimpleNamespace(AFTER="after")
    registry.HookRegistry = types.SimpleNamespace(register=lambda *args: hooks.append(args))
    monkeypatch.setattr(ar_norm_aiter, "_installed", False)
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
    assert fcf.flashinfer_allreduce_residual_rmsnorm is stock
    ar_norm_aiter.install()
    assert len(hooks) == 1
    assert hooks[0][2] == "after"
    group = GroupCoordinator()
    modules["sglang.srt.distributed"].get_tp_group = lambda: group
    hooks[0][1](None)
    installed = fcf.flashinfer_allreduce_residual_rmsnorm
    hooks[0][1](None)
    assert fcf.flashinfer_allreduce_residual_rmsnorm is installed
    assert installed._hcu_ar_norm_aiter
    assert installed(None, None, None) == ("stock-output", "stock-residual")
    assert len(registrations) == 3
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


@pytest.mark.parametrize("case", ["ready", "uninitialized", "wrong-device", "cpu",
                                 "missing", "disabled", "missing-method", "stock",
                                 "host-failure", "admission-failure", "register-failure"])
def test_deferred_admission_preserves_host_lifecycle(monkeypatch, case):
    events = []
    ca = types.SimpleNamespace(disabled=case == "disabled", fused_ar_rms=lambda: None)
    if case == "missing-method":
        ca.fused_ar_rms = None

    class GroupCoordinator:
        def __init__(self, device_index, *, sentinel):
            events.append(("host", device_index, sentinel))
            if case == "host-failure":
                raise ValueError("host constructor failed")
            self.device = types.SimpleNamespace(type="cpu" if case == "cpu" else "cuda",
                                                index=device_index)
            self.ca_comm = None if case == "missing" else ca

        def fused_allreduce_rmsnorm(self, *args):
            return None

    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip="fixture")
    torch.cuda = types.SimpleNamespace(
        is_initialized=lambda: case != "uninitialized",
        current_device=lambda: 0 if case == "wrong-device" else 3,
    )
    parallel_state = types.ModuleType("sglang.srt.distributed.parallel_state")
    parallel_state.GroupCoordinator = GroupCoordinator
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, parallel_state.__name__, parallel_state)
    hooks = []
    registry = types.ModuleType("sglang.srt.plugins.hook_registry")
    registry.HookType = types.SimpleNamespace(AFTER="after")
    registry.HookRegistry = types.SimpleNamespace(register=lambda *args: hooks.append(args))
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    distributed = types.ModuleType("sglang.srt.distributed")
    monkeypatch.setitem(sys.modules, distributed.__name__, distributed)
    monkeypatch.setattr(ar_norm_aiter, "_installed", False)
    monkeypatch.setattr(ar_norm_aiter.hcu_env, "resolve",
                        lambda name: ("stock" if case == "stock" else "aiter", None))

    def register(*args, **kwargs):
        if case == "register-failure":
            raise RuntimeError("registration failed")

    def dispatch():
        events.append("admission")
        if case == "admission-failure":
            raise ImportError("consumer unavailable")

    monkeypatch.setattr(ar_norm_aiter, "register", register)
    monkeypatch.setattr(ar_norm_aiter, "_install_dispatch", dispatch)
    original = GroupCoordinator.__init__
    if case == "register-failure":
        with pytest.raises(RuntimeError, match="registration failed"):
            ar_norm_aiter.install()
        assert GroupCoordinator.__init__ is original
        assert not hooks
        return
    ar_norm_aiter.install()
    assert not events  # bootstrap must not import/construct the provider or query a device
    if case == "host-failure":
        with pytest.raises(ValueError, match="host constructor failed"):
            GroupCoordinator(3, sentinel="preserved")
        assert events == [("host", 3, "preserved")]
        return
    group = GroupCoordinator(3, sentinel="preserved")
    distributed.get_tp_group = lambda: group
    for target, callback, kind in hooks:
        assert target.endswith("initialize_model_parallel") and kind == "after"
        assert callback("host-result") == "host-result"
    assert group.ca_comm is (None if case == "missing" else ca)
    expected = [("host", 3, "preserved")]
    if case in ("ready", "admission-failure"):
        expected.append("admission")
    assert events == expected
