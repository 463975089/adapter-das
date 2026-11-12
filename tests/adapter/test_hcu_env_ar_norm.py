# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""The adapter bootstrap must not import AITER's GPU all-reduce module."""

import importlib.util
from types import SimpleNamespace

import torch

from hcu_adapter import hcu_env


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
