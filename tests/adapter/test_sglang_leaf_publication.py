# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU publication/failure contracts; doubles do not certify GPU kernels."""
import inspect
import sys
import types

import pytest

from hcu_adapter.infra import registry
from hcu_adapter.bridges._sglang import grouped_topk, sampling
from hcu_adapter.bridges._sglang._publication import Binding, publish


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(registry, "_RECORDS", {"prior": {"op": "prior"}})
    monkeypatch.setattr(registry, "_BOUND", {"prior.target": "prior"})
    modules = {}

    def module(name, **attributes):
        if "." in name:
            parent, _, symbol = name.rpartition(".")
            if parent not in modules:
                module(parent)
        value = types.ModuleType(name)
        value.__path__ = []
        vars(value).update(attributes)
        modules[name] = value
        monkeypatch.setitem(sys.modules, name, value)
        if "." in name:
            setattr(modules[parent], symbol, value)
        return value

    def original(*args, **kwargs):
        return args, kwargs

    class Kernel:
        def run(self, *args, **kwargs):
            return args, kwargs

    kernel = Kernel()
    module("torch", version=types.SimpleNamespace(hip="test"),
           cuda=types.SimpleNamespace(get_device_properties=lambda _: types.SimpleNamespace(warp_size=64)))
    flash = module("flashinfer.sampling", top_k_renorm_probs=original, top_p_renorm_probs=original)
    tree = module("sglang.kernels.ops.speculative.sampling", tree_speculative_sampling_target_only=original)
    cached = module("sglang.srt.cached", tree_speculative_sampling_target_only=original,
                    biased_grouped_topk_gpu=original, biased_grouped_topk=original)
    topk = module("sglang.srt.layers.moe.topk", biased_grouped_topk_gpu=original,
                  biased_grouped_topk=original, _RENORMALIZE_SUM_EPSILON=1e-20)
    module("sglang.kernels.ops.grammar.bitmask_ops", apply_token_bitmask_inplace_kernel=kernel)
    provider = types.ModuleType("hcu_adapter.kernels.sglang_tree_sampling_triton")
    provider.tree_speculative_sampling_target_only = lambda *args, **kwargs: (args, kwargs)
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    from hcu_adapter.ops import selector
    monkeypatch.setattr(selector, "select", lambda *args: "triton")
    return types.SimpleNamespace(flash=flash, tree=tree, cached=cached, topk=topk,
                                 provider=provider, original=original, kernel=kernel)


def assert_stock(host):
    assert host.flash.top_k_renorm_probs is host.original
    assert host.flash.top_p_renorm_probs is host.original
    assert host.tree.tree_speculative_sampling_target_only is host.original
    assert host.cached.tree_speculative_sampling_target_only is host.original
    assert host.topk.biased_grouped_topk_gpu is host.original
    assert host.cached.biased_grouped_topk is host.original
    assert "run" not in vars(host.kernel)
    assert registry._RECORDS == {"prior": {"op": "prior"}}
    assert registry._BOUND == {"prior.target": "prior"}


@pytest.mark.parametrize("failure_at", [1, 2, 3, 4])
def test_sampling_registration_failure_publishes_nothing_and_allows_retry(host, monkeypatch, failure_at):
    original_register = registry.register
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        original_register(*args, **kwargs)
        calls += 1
        if calls == failure_at:
            raise RuntimeError("injected registration failure")

    with monkeypatch.context() as context:
        context.setattr(registry, "register", fail)
        with pytest.raises(RuntimeError, match="injected registration failure"):
            sampling.install()
    assert_stock(host)
    sampling.install()
    assert host.flash.top_k_renorm_probs is sampling.top_k_renorm_probs
    assert host.tree.tree_speculative_sampling_target_only is host.provider.tree_speculative_sampling_target_only
    assert host.cached.tree_speculative_sampling_target_only is host.tree.tree_speculative_sampling_target_only
    assert getattr(host.kernel.run, "_hcu_grammar_launch", False)


@pytest.mark.parametrize("missing", [
    "hcu_adapter.kernels.sglang_tree_sampling_triton",
    "sglang.kernels.ops.speculative.sampling",
    "sglang.kernels.ops.grammar.bitmask_ops",
])
def test_sampling_prepare_failure_leaves_all_symbols_and_records_stock(host, monkeypatch, missing):
    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(ImportError):
        sampling.install()
    assert_stock(host)


def test_grouped_topk_registration_failure_keeps_preimported_aliases(host, monkeypatch):
    original_register = registry.register

    def fail(*args, **kwargs):
        original_register(*args, **kwargs)
        raise RuntimeError("group registration failure")

    with monkeypatch.context() as context:
        context.setattr(registry, "register", fail)
        with pytest.raises(RuntimeError, match="group registration failure"):
            grouped_topk.install()
    assert_stock(host)
    grouped_topk.install()
    wrapped = host.topk.biased_grouped_topk_gpu
    assert wrapped is host.topk.biased_grouped_topk is host.cached.biased_grouped_topk_gpu
    assert wrapped is host.cached.biased_grouped_topk
    grouped_topk.install()
    assert host.topk.biased_grouped_topk_gpu is wrapped
    small = types.SimpleNamespace(ndim=2, shape=(1, 288))
    assert wrapped("hidden", small, "bias", 8, True) == (
        ("hidden", small, "bias", 8, True, None, None, 0, None, False), {},
    )


def test_sampling_keeps_existing_grammar_owner_and_idempotent_aliases(host):
    sampling.install_grammar_bitmask()
    grammar = host.kernel.run
    sampling.install()
    provider = host.tree.tree_speculative_sampling_target_only
    sampling.install()
    assert host.kernel.run is grammar
    assert host.cached.tree_speculative_sampling_target_only is provider
    logits = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"))
    assert grammar(logits, num_warps=32) == ((logits,), {"num_warps": 16})


def test_foreign_change_during_registration_is_not_overwritten(host):
    foreign = object()
    binding = Binding.prepare(host.flash, "top_k_renorm_probs", sampling.top_k_renorm_probs)

    def register():
        registry.register("sampling", sampling.top_k_renorm_probs, "test.foreign", registry.MECH_REBIND)
        host.flash.top_k_renorm_probs = foreign

    with pytest.raises(RuntimeError, match="changed before publication"):
        publish([binding], register)
    assert host.flash.top_k_renorm_probs is foreign
    assert registry._RECORDS == {"prior": {"op": "prior"}}
    assert registry._BOUND == {"prior.target": "prior"}


def test_interrupted_publication_rolls_back_existing_and_new_attributes(host):
    owner = types.SimpleNamespace(existing=object())
    old = owner.existing
    bindings = [Binding.prepare(owner, "new", object()), Binding.prepare(owner, "existing", object())]
    lines, start = inspect.getsourcelines(publish)
    write_line = start + next(i for i, line in enumerate(lines) if "binding.namespace[binding.name] = binding.implementation" in line)
    writes = 0

    def interrupt(frame, event, arg):
        nonlocal writes
        if frame.f_code is publish.__code__ and event == "line" and frame.f_lineno == write_line:
            writes += 1
            if writes == 2:
                raise KeyboardInterrupt("injected publication interruption")
        return interrupt

    def register():
        registry.register("sampling", "test", "test.interrupted", registry.MECH_REBIND)

    previous_trace = sys.gettrace()
    try:
        sys.settrace(interrupt)
        with pytest.raises(KeyboardInterrupt, match="injected publication interruption"):
            publish(bindings, register)
    finally:
        sys.settrace(previous_trace)
    assert writes == 2 and vars(owner) == {"existing": old}
    assert registry._RECORDS == {"prior": {"op": "prior"}}
    assert registry._BOUND == {"prior.target": "prior"}


def test_duplicate_bindings_fail_before_registration(host):
    binding = Binding.prepare(host.flash, "top_k_renorm_probs", object())
    with pytest.raises(ValueError, match="Duplicate"):
        publish([binding, binding], lambda: pytest.fail("register called before validation"))
    assert_stock(host)
