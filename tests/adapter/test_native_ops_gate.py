# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Serve-time native-ops hard gate (bridges/sglang_native_ops_gate.py).

A server launch without native sgl_kernel ops must abort loudly instead of
silently serving on runtime ninja JIT (2026-11-19 directive). The gate arms
on the ModelRunner import and checks ``sgl_kernel._native.native_available``
in ``ModelRunner.__init__``.
"""

import importlib
import sys
import types

import pytest

import hcu_adapter.bridges.sglang_native_ops_gate as gate

TARGET = "sglang.srt.model_executor.model_runner"
_PRISTINE_IMPORT = importlib.import_module  # before any gate install


def _fresh_runner():
    """A NEW class per use: the gate mutates cls.__init__ in place, so a
    shared module-level class would carry the closure across tests."""
    class _ModelRunner:
        def __init__(self, *args, **kwargs):
            self.started = True

    return _ModelRunner


@pytest.fixture()
def runner_module(monkeypatch):
    mod = types.ModuleType(TARGET)
    mod.ModelRunner = _fresh_runner()
    monkeypatch.setitem(sys.modules, TARGET, mod)
    return mod


def _arm(runner_module, *, env=None):
    """Fresh class -> install -> trigger arming THROUGH the new wrapper."""
    if env is not None:
        import os

        os.environ["HCU_REQUIRE_NATIVE"] = env
    runner_module.ModelRunner = _fresh_runner()
    gate.install()
    importlib.import_module(TARGET)  # current attr IS the new wrapper: arms
    importlib.import_module = _PRISTINE_IMPORT
    if env is not None:
        del os.environ["HCU_REQUIRE_NATIVE"]


def _set_native(monkeypatch, available: bool):
    import sgl_kernel._native as nat

    monkeypatch.setattr(nat, "native_available", lambda: available)


@pytest.fixture()
def armed(runner_module):
    _arm(runner_module)
    yield runner_module


def test_launch_aborts_without_native(armed, monkeypatch):
    _set_native(monkeypatch, False)
    with pytest.raises(RuntimeError, match="native sgl_kernel ops"):
        armed.ModelRunner()


def test_launch_passes_with_native(armed, monkeypatch):
    _set_native(monkeypatch, True)
    runner = armed.ModelRunner()
    assert runner.started


def test_env0_downgrades_to_warning(runner_module, monkeypatch, caplog):
    monkeypatch.setenv("HCU_REQUIRE_NATIVE", "0")
    _arm(runner_module)  # reads the env: required=False
    _set_native(monkeypatch, False)
    with caplog.at_level("WARNING", logger="hcu.adapter"):
        runner = runner_module.ModelRunner()  # must NOT raise
    assert runner.started
    assert any("native sgl_kernel ops" in r.message for r in caplog.records)
