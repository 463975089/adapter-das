# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""The adapter bootstrap must not import AITER's GPU all-reduce module."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from hcu_adapter import hcu_config, hcu_env


def test_ar_norm_probe_checks_module_without_import(monkeypatch, tmp_path):
    package = tmp_path / "aiter"
    module = package / "dist" / "device_communicators" / "custom_all_reduce.py"
    module.parent.mkdir(parents=True)
    module.write_text("raise AssertionError('module was imported')\n")

    def find_spec(name):
        assert name == "aiter"
        return SimpleNamespace(submodule_search_locations=[str(package)])

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(torch.version, "hip", "7.0", raising=False)
    monkeypatch.setattr(hcu_env, "_auto_aiter_ar", lambda: "allowed")

    assert hcu_env.has_aiter_custom_allreduce()
    assert hcu_env._auto_ar_norm_impl() == "aiter"
    module.unlink()
    assert not hcu_env.has_aiter_custom_allreduce()
    assert hcu_env._auto_ar_norm_impl() == "stock"


def test_ar_norm_probe_without_aiter(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(torch.version, "hip", "7.0", raising=False)
    assert not hcu_env.has_aiter_custom_allreduce()
    assert hcu_env._auto_ar_norm_impl() == "stock"


@pytest.mark.parametrize("host", ["sglang", "vllm"])
@pytest.mark.parametrize("module_present", [False, True])
def test_ar_norm_layout_gate_is_host_specific(monkeypatch, host, module_present):
    checked = []

    def check_layout():
        checked.append(True)
        return module_present

    monkeypatch.delenv("HCU_AR_NORM_IMPL", raising=False)
    monkeypatch.setattr(hcu_config, "lookup", lambda name: None)
    monkeypatch.setattr(torch.version, "hip", "7.0", raising=False)
    monkeypatch.setattr(hcu_env, "_auto_aiter_ar", lambda: "allowed")
    monkeypatch.setattr(hcu_env, "has_aiter_custom_allreduce", check_layout)

    expected = "aiter" if host == "sglang" or module_present else "stock"
    assert hcu_env.resolve("HCU_AR_NORM_IMPL", host=host) == (expected, "auto")
    assert checked == ([] if host == "sglang" else [True])


@pytest.mark.parametrize("hip,policy,override,expected,source", [
    (None, "allowed", None, "stock", "auto"),
    ("7.0", "refused(old-aiter graph bug)", None, "stock", "auto"),
    ("7.0", "allowed", "stock", "stock", "explicit"),
])
def test_sglang_ar_norm_preserves_safety_and_override(
        monkeypatch, hip, policy, override, expected, source):
    monkeypatch.delenv("HCU_AR_NORM_IMPL", raising=False)
    if override is not None:
        monkeypatch.setenv("HCU_AR_NORM_IMPL", override)
    monkeypatch.setattr(hcu_config, "lookup", lambda name: None)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.setattr(hcu_env, "_auto_aiter_ar", lambda: policy)
    assert hcu_env.resolve("HCU_AR_NORM_IMPL", host="sglang") == (expected, source)
