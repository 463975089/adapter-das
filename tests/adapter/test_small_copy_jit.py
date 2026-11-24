import types

import pytest

from hcu_adapter.features.jit import small_copy_jit
from hcu_adapter.infra import registry
from hcu_adapter.features.jit.small_copy_jit import adapt_kernel


def test_alias_identity_options_and_idempotence():
    calls = []
    class Standard:
        def __init__(self, fn, **kwargs):
            calls.append((fn, kwargs))
        def run(self, *args, **kwargs):
            return args, kwargs
    Fast = type("FastJITFunction", (), {"__module__": "triton.backends.hcu.jit"})
    kernel = Fast()
    kernel.fn = lambda: None
    kernel._init_kwargs = {"do_not_specialize": ["sizes"], "debug": False}
    alias = kernel
    assert adapt_kernel(kernel, Standard)
    assert alias is kernel
    assert alias.run(7, grid=(2,)) == ((7,), {"grid": (2,)})
    assert calls == [(kernel.fn, kernel._init_kwargs)]
    assert not adapt_kernel(kernel, Standard)
    assert len(calls) == 1


def test_other_jit_is_untouched():
    kernel = types.SimpleNamespace(run=object())
    original = kernel.run
    assert not adapt_kernel(kernel, None)
    assert kernel.run is original


@pytest.fixture
def installation(monkeypatch):
    import sys

    Fast = type("FastJITFunction", (), {"__module__": "triton.backends.hcu.jit"})
    kernel = Fast()
    kernel.fn = lambda: None
    kernel._init_kwargs = {"do_not_specialize": ["sizes"], "debug": False}
    kernel.run = original = object()
    constructed = []

    class Standard:
        def __init__(self, fn, **kwargs):
            constructed.append((fn, kwargs))

        def run(self, *args, **kwargs):
            return args, kwargs

    def module(name, **attributes):
        value = types.ModuleType(name)
        value.__path__ = []
        vars(value).update(attributes)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    module("torch", version=types.SimpleNamespace(hip="test"))
    module("sglang")
    module("sglang.kernels")
    module("sglang.kernels.ops")
    small_copy = module("sglang.kernels.ops.memory.small_copy", _small_copy_kernel=kernel)
    module("sglang.kernels.ops.memory", small_copy=small_copy)
    module("triton")
    module("triton.runtime")
    module("triton.runtime.jit", JITFunction=Standard)
    monkeypatch.setattr(registry, "_RECORDS", {"prior": {"op": "prior"}})
    monkeypatch.setattr(registry, "_BOUND", {"prior.target": "prior"})
    return types.SimpleNamespace(kernel=kernel, original=original, constructed=constructed)


class PublicationAbort(BaseException):
    pass


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("error", [RuntimeError, PublicationAbort])
def test_registration_failure_keeps_binding_and_allows_retry(installation, monkeypatch, partial, error):
    calls = []
    state = installation

    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        assert state.kernel.run is state.original
        assert not hasattr(state.kernel, "_hcu_small_copy_standard_jit")
        if partial:
            registry._RECORDS["small_copy"] = {"op": "small_copy"}
            registry._BOUND[kwargs["target"]] = "small_copy"
        raise error("injected registration failure")

    with monkeypatch.context() as context:
        context.setattr(registry, "register", fail)
        with pytest.raises(error, match="injected registration failure"):
            small_copy_jit.install()
    assert state.kernel.run is state.original
    assert not hasattr(state.kernel, "_hcu_small_copy_standard_jit")
    assert registry._RECORDS == {"prior": {"op": "prior"}}
    assert registry._BOUND == {"prior.target": "prior"}
    small_copy_jit.install()
    assert registry.get("small_copy")["implementation"] == "standard-jit"
    assert state.kernel.run(7, grid=(2,)) == ((7,), {"grid": (2,)})
    assert state.kernel._hcu_small_copy_standard_jit is state.kernel.run.__self__
    small_copy_jit.install()
    assert len(state.constructed) == 2
    assert len(calls) == 1


def test_provider_prepare_failure_does_not_register(installation, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("prepare failed")

    import sys
    monkeypatch.setattr(sys.modules["triton.runtime.jit"], "JITFunction", fail)
    with pytest.raises(RuntimeError, match="prepare failed"):
        small_copy_jit.install()
    assert installation.kernel.run is installation.original
    assert not hasattr(installation.kernel, "_hcu_small_copy_standard_jit")
    assert registry._RECORDS == {"prior": {"op": "prior"}}
    assert registry._BOUND == {"prior.target": "prior"}


def test_changed_owner_during_registration_is_not_overwritten(installation, monkeypatch):
    foreign = object()
    original_register = registry.register

    def change_owner(*args, **kwargs):
        original_register(*args, **kwargs)
        installation.kernel.run = foreign

    monkeypatch.setattr(registry, "register", change_owner)
    with pytest.raises(RuntimeError, match="changed before publication"):
        small_copy_jit.install()
    assert installation.kernel.run is foreign
    assert not hasattr(installation.kernel, "_hcu_small_copy_standard_jit")
    assert registry._RECORDS == {"prior": {"op": "prior"}}
    assert registry._BOUND == {"prior.target": "prior"}


@pytest.mark.parametrize("debug", [False, True])
def test_vendor_without_init_kwargs_preserves_jit_options(installation, debug):
    kernel = installation.kernel
    del kernel._init_kwargs
    representation = lambda signature: "copy:" + str(signature)
    metadata = lambda *args: {"name": "copy"}
    options = dict(version=0, do_not_specialize=["sizes", 2],
                   do_not_specialize_on_alignment=["cols"], debug=debug,
                   noinline=False, repr=representation, launch_metadata=metadata)
    for name, value in options.items():
        setattr(kernel, "_repr" if name == "repr" else name, value)

    small_copy_jit.install()

    assert installation.constructed == [(kernel.fn, options)]
    assert kernel.run.__self__ is kernel._hcu_small_copy_standard_jit
    assert registry.get("small_copy")["implementation"] == "standard-jit"
    small_copy_jit.install()
    assert len(installation.constructed) == 1


@pytest.mark.parametrize("missing", ["version", "do_not_specialize_on_alignment", "_repr", "launch_metadata"])
def test_incomplete_vendor_options_refuse_before_registration(installation, missing):
    kernel = installation.kernel
    del kernel._init_kwargs
    for name, value in dict(version=None, do_not_specialize=[],
                            do_not_specialize_on_alignment=[], debug=False,
                            noinline=None, _repr=None, launch_metadata=None).items():
        setattr(kernel, name, value)
    delattr(kernel, missing)

    with pytest.raises(TypeError, match="cannot preserve Triton JIT options"):
        small_copy_jit.install()

    assert installation.constructed == []
    assert kernel.run is installation.original
    assert not hasattr(kernel, "_hcu_small_copy_standard_jit")
    assert registry._RECORDS == {"prior": {"op": "prior"}}
    assert registry._BOUND == {"prior.target": "prior"}


def test_bootstrap_step_degrades_on_vendor_abi_rejection(monkeypatch, caplog):
    """A vendor-ABI refusal (TypeError) inside the small_copy bootstrap step
    must degrade to a warning and let the later fixes still install.

    Bot P2 (2026-11-23): the step used to catch only ImportError, so the
    pre-registration refusals raised by install() aborted the remaining
    bootstrap fixes (dsa top-k v2 fusion and the rest)."""
    import logging
    import sys

    from hcu_adapter.bridges import sglang_bootstrap_fixes

    def neutralize(parent, name):
        target = getattr(parent, name)
        monkeypatch.setattr(target, "install", lambda: None)
        return target

    # Earlier steps become no-ops so the chain reaches small_copy cleanly.
    from hcu_adapter.infra import dist_bc, kv_fix, watchdog_cold
    from hcu_adapter.features.jit import jit_build_log, jit_fp8_ffi, jit_sources

    for module in (dist_bc, watchdog_cold, kv_fix, jit_sources, jit_build_log, jit_fp8_ffi):
        monkeypatch.setattr(module, "install", lambda: None)
    # jit_eh_norm (bridges._sglang) needs the sglang tree; without it the
    # step already takes its own except-Exception path, so no mock needed.

    def refuse():
        raise TypeError("cannot preserve Triton JIT options: incomplete vendor ABI")

    monkeypatch.setattr(small_copy_jit, "install", refuse)

    # The step after small_copy must still run.
    later = []
    import hcu_adapter.features.dsa as dsa_pkg

    monkeypatch.setattr(
        dsa_pkg, "dsa_topk_v2",
        types.SimpleNamespace(install=lambda: later.append(1)), raising=False,
    )
    # The final step imports sglang; keep it failing safely on ImportError.
    monkeypatch.setitem(sys.modules, "sglang.srt.models.registry", None)

    caplog.set_level(logging.WARNING, logger="hcu.adapter")
    sglang_bootstrap_fixes.install()  # must not raise

    assert later == [1]
    assert any(
        "small_copy JIT fix not installed" in rec.message for rec in caplog.records
    )
