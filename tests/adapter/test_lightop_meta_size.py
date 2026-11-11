# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""aiter.meta_size stays a host-only sizeof query (CPU, no GPU, no aiter).

aiter's custom_all_reduce module calls ``ops.meta_size()`` at import time.
The vendor torch_guard wrapper allocates ``torch.empty(1, device=cuda)`` as a
dispatch dummy, which opens a context on the current device (0 before
set_device) in every rank. An import-time property query has already marked
torch initialized by then, so the wrapper must not gate on
``torch.cuda.is_initialized()``.

Run: pytest tests/adapter/test_lightop_meta_size.py -v
"""

import sys
import types

from hcu_adapter import lightop_metadata, registry

_SIZEOF_SIGNAL = 25984


def _install_fakes(monkeypatch, native):
    calls = {"vendor": 0}

    def vendor_meta_size():
        calls["vendor"] += 1
        return _SIZEOF_SIGNAL

    aiter = types.ModuleType("aiter")
    aiter.__path__ = []
    aiter.meta_size = vendor_meta_size
    jit = types.ModuleType("aiter.jit")
    jit.__path__ = []
    core = types.ModuleType("aiter.jit.core")
    core.get_module = lambda name: native(name)
    for name, mod in (("aiter", aiter), ("aiter.jit", jit), ("aiter.jit.core", core)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "triton.backends.driver", raising=False)
    monkeypatch.setattr(lightop_metadata, "_RESOLUTION_TARGETS", {})
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    registrations = []
    monkeypatch.setattr(registry, "register", lambda *a, **k: registrations.append((a, k)))
    return aiter, calls, registrations


def _native_ok(name):
    assert name == "module_custom_all_reduce"
    return types.SimpleNamespace(meta_size=lambda: _SIZEOF_SIGNAL)


def test_native_sizeof_even_after_torch_initialized(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_initialized=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    aiter, calls, registrations = _install_fakes(monkeypatch, _native_ok)

    lightop_metadata.install()

    assert aiter.meta_size() == _SIZEOF_SIGNAL
    assert calls["vendor"] == 0
    assert registrations[0][1]["target"] == "aiter.meta_size"
    assert registrations[0][1]["mechanism"] == registry.MECH_REBIND


def test_vendor_fallback_when_native_module_missing(monkeypatch):
    def native_missing(name):
        raise ImportError(name)

    aiter, calls, _ = _install_fakes(monkeypatch, native_missing)

    lightop_metadata.install()

    assert aiter.meta_size() == _SIZEOF_SIGNAL
    assert calls["vendor"] == 1


def test_install_is_idempotent(monkeypatch):
    aiter, _, registrations = _install_fakes(monkeypatch, _native_ok)

    lightop_metadata.install()
    wrapped = aiter.meta_size
    lightop_metadata.install()

    assert aiter.meta_size is wrapped
    assert len(registrations) == 1


def test_wrapper_does_not_import_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    aiter, _, _ = _install_fakes(monkeypatch, _native_ok)

    lightop_metadata.install()
    aiter.meta_size()

    assert "torch" not in sys.modules
