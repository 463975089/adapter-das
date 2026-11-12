# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""A failure after the public quantization registration must be commit-fatal.

Core has no `unregister`. Once `register_quantization_config(QUANTIZATION_METHOD)`
has run, this process holds a name it cannot remove: if activation then raised a
recoverable error, the extension framework would keep starting, the native model
could re-select the config, and the failure would surface later at construction
where it looks unrelated. A same-process retry would also publish the draft twice.

So two rules are asserted here: everything that can refuse does so *before* the
registry is touched, and anything failing *after* it latches `_failure`.

CPU only; the public quantization API is replaced with a stand-in.
"""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from hcu_adapter import dsv41_activation as activation
from hcu_adapter.extensions import ExtensionCommitError, ExtensionContext

METHOD = activation.QUANTIZATION_METHOD


class _Registry:
    """Stand-in for vllm's public quantization registry (no unregister)."""

    def __init__(self):
        self.methods = []
        self.configs = {}
        self.register_calls = 0

    def register_quantization_config(self, quantization):
        def apply(config_class):
            self.register_calls += 1
            self.methods.append(quantization)
            self.configs[quantization] = config_class
        return apply

    def get_quantization_config(self, quantization):
        return self.configs.get(quantization)

    @property
    def QUANTIZATION_METHODS(self):
        return self.methods


@pytest.fixture
def isolated(monkeypatch):
    """Fresh activation state with the public quant API and candidate stubbed."""
    registry = _Registry()
    quant = ModuleType("vllm.model_executor.layers.quantization")
    quant.register_quantization_config = registry.register_quantization_config
    quant.get_quantization_config = registry.get_quantization_config
    quant.QUANTIZATION_METHODS = registry.QUANTIZATION_METHODS

    config_class = type("HcuDSV41ChannelFP8Config", (), {})
    candidate = SimpleNamespace(
        HcuDSV41ChannelFP8Config=config_class, _is_hcu=lambda: False)

    real_import = activation._import_module

    def fake_import(name):
        if name == "vllm.model_executor.layers.quantization":
            return quant
        if name == "hcu_adapter.dsv41_channel_fp8":
            return candidate
        return real_import(name)

    monkeypatch.setattr(activation, "_import_module", fake_import)
    monkeypatch.setattr(activation, "_failure", None)
    monkeypatch.setattr(activation, "_publication", None)
    monkeypatch.setattr(activation, "_publishing", False)
    monkeypatch.setattr(activation, "_engram_edp_hooks", None)
    monkeypatch.setattr(activation, "_dspark_graph_hooks", None)
    monkeypatch.setattr(activation, "_provider_binder", None)
    monkeypatch.setenv(activation.MODE_VARIABLE, "native")
    monkeypatch.delenv("HCU_DISABLE_OPS", raising=False)
    return SimpleNamespace(registry=registry, config_class=config_class)


def _bind_guards(monkeypatch, install=None, check=None):
    monkeypatch.setattr(
        activation, "_engram_edp_hooks", (install or (lambda: None), check or (lambda: None)))
    monkeypatch.setattr(
        activation, "_dspark_graph_hooks", (lambda: None, lambda: None))


CONTEXT = ExtensionContext("0.1.0", "0.33.0")


def test_unbound_guards_refuse_before_touching_the_registry(isolated, monkeypatch):
    """The pre-check must run ahead of `register_quantization_config`."""
    with pytest.raises(activation.Dsv41ActivationError, match="guard providers are not bound"):
        activation.activate_extension(CONTEXT)
    assert isolated.registry.register_calls == 0
    assert activation._publication is None
    assert activation._failure is None


def test_guard_install_failure_after_registration_latches_commit_fatal(isolated, monkeypatch):
    """A guard that raises post-registration cannot be a recoverable error."""
    _bind_guards(monkeypatch, install=lambda: (_ for _ in ()).throw(RuntimeError("guard boom")))

    with pytest.raises(ExtensionCommitError, match="guard install failed"):
        activation.activate_extension(CONTEXT)

    assert isolated.registry.register_calls == 1, "registration had already happened"
    assert activation._failure is not None, "failure must latch, not be recoverable"
    assert activation._publication is None


def test_latched_failure_refuses_a_same_process_retry(isolated, monkeypatch):
    _bind_guards(monkeypatch, install=lambda: (_ for _ in ()).throw(RuntimeError("guard boom")))
    with pytest.raises(ExtensionCommitError):
        activation.activate_extension(CONTEXT)

    # Second attempt must refuse immediately, not re-register or re-publish.
    calls_before = isolated.registry.register_calls
    with pytest.raises(ExtensionCommitError):
        activation.activate_extension(CONTEXT)
    assert isolated.registry.register_calls == calls_before


def test_guard_install_failure_is_not_reported_as_a_plain_runtime_error(isolated, monkeypatch):
    """A bare RuntimeError would let the host treat this as a skippable extension."""
    _bind_guards(monkeypatch, install=lambda: (_ for _ in ()).throw(RuntimeError("guard boom")))
    with pytest.raises(ExtensionCommitError):
        activation.activate_extension(CONTEXT)


def test_healthy_activation_registers_once_and_publishes(isolated, monkeypatch):
    _bind_guards(monkeypatch)
    activation.activate_extension(CONTEXT)

    assert isolated.registry.register_calls == 1
    assert activation._publication is not None
    assert activation._failure is None
    assert isolated.registry.get_quantization_config(METHOD) is isolated.config_class


def test_stock_mode_returns_without_registering(isolated, monkeypatch):
    monkeypatch.setenv(activation.MODE_VARIABLE, "stock")
    _bind_guards(monkeypatch)
    activation.activate_extension(CONTEXT)

    assert isolated.registry.register_calls == 0
    assert activation._publication is None
    assert activation._failure is None


def test_health_refuses_a_registry_record_that_lost_its_owner(isolated, monkeypatch):
    """The registry record is part of the published state, not bookkeeping.

    Activation makes `_publication` and the record one atomic step. If the record
    is later removed or re-pointed, the process is no longer in the state it was
    admitted in, and `check_health()` used to say nothing about it.
    """
    _bind_guards(monkeypatch)
    activation.activate_extension(CONTEXT)
    activation.check_health()

    from hcu_adapter import registry as adapter_registry

    original = adapter_registry.get(activation.OP)
    assert original is not None
    # `_fatal` latches `_failure`, so restore the record AND clear the latch to
    # leave the next case a clean process.
    try:
        monkeypatch.setitem(adapter_registry._RECORDS, activation.OP, None)
        with pytest.raises(ExtensionCommitError, match="registry record"):
            activation.check_health()
    finally:
        adapter_registry._RECORDS[activation.OP] = original
        monkeypatch.setattr(activation, "_failure", None)


def test_health_refuses_a_registry_record_with_a_foreign_target(isolated, monkeypatch):
    """A second owner re-pointing the op must be refused, not reported healthy."""
    _bind_guards(monkeypatch)
    activation.activate_extension(CONTEXT)

    from hcu_adapter import registry as adapter_registry

    original = adapter_registry.get(activation.OP)
    try:
        forged = dict(original)
        forged["target"] = "some.other.module.attribute"
        monkeypatch.setitem(adapter_registry._RECORDS, activation.OP, forged)
        with pytest.raises(ExtensionCommitError, match="registry record"):
            activation.check_health()
    finally:
        adapter_registry._RECORDS[activation.OP] = original
        monkeypatch.setattr(activation, "_failure", None)
