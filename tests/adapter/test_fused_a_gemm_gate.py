# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fused QKV-A GEMM eligibility gate: dispatch/registration only, no GPU.

Pins the HIP behaviour behind the Kimi-K3.1 capture failure: the
predicate must read False in its own module and in every importer that
bound it by name, an importer loaded later must be covered by the post-load
hook, and a CUDA host must be left untouched.
"""

import sys
import types

import pytest
import torch

from hcu_adapter import fused_a_gemm_gate as gate, registry

_GEMM = "sglang.kernels.ops.gemm.fused_a_gemm"
_DEEPSEEK = "sglang.srt.models.deepseek_v2"
_NEMOTRON = "sglang.srt.models.nemotron_h"


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test-hip")
    monkeypatch.setattr(registry, "_RECORDS", {})
    monkeypatch.setattr(registry, "_BOUND", {})
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))

    def original(layer):
        return True

    module = types.ModuleType(_GEMM)
    module.fused_a_gemm_weight_eligible = original
    monkeypatch.setitem(sys.modules, _GEMM, module)
    return module, original


def _importer(name, predicate):
    module = types.ModuleType(name)
    module.fused_a_gemm_weight_eligible = predicate
    return module


def test_predicate_rebound_everywhere_and_idempotent(host, monkeypatch):
    module, original = host
    deepseek = _importer(_DEEPSEEK, original)
    monkeypatch.setitem(sys.modules, _DEEPSEEK, deepseek)

    gate.install()
    replaced = module.fused_a_gemm_weight_eligible
    assert replaced is deepseek.fused_a_gemm_weight_eligible
    assert replaced is not original
    assert replaced(None) is False
    assert replaced.__wrapped__ is original
    record = registry.get("fused_a_gemm")
    assert record["implementation"] == "cuda-only-kernel-off"
    assert record["mechanism"] == registry.MECH_REBIND
    assert record["target"] == _GEMM + ".fused_a_gemm_weight_eligible"

    gate.install()
    assert module.fused_a_gemm_weight_eligible is replaced


def test_late_importer_covered_by_post_load_hook(host, monkeypatch):
    module, original = host
    monkeypatch.setitem(sys.modules, _DEEPSEEK, _importer(_DEEPSEEK, original))
    monkeypatch.delitem(sys.modules, _NEMOTRON, raising=False)
    gate.install()
    finder = next(
        f for f in sys.meta_path if getattr(f, "lib_name", None) == _NEMOTRON
    )
    # nemotron_h imports the name under `_is_cuda`; a loader that ran before
    # the gate would still hold the original object.
    later = _importer(_NEMOTRON, original)
    finder.post_hooks[_NEMOTRON](later)
    assert later.fused_a_gemm_weight_eligible is module.fused_a_gemm_weight_eligible
    assert later.fused_a_gemm_weight_eligible(None) is False
    assert registry.get("fused_a_gemm") is not None


def test_non_hip_leaves_host_untouched(host, monkeypatch):
    module, original = host
    monkeypatch.setattr(torch.version, "hip", None)
    gate.install()
    assert module.fused_a_gemm_weight_eligible is original
    assert registry.get("fused_a_gemm") is None
    assert not any(
        getattr(f, "lib_name", None) == _NEMOTRON for f in sys.meta_path
    )


def test_registration_failure_does_not_publish_importers(host, monkeypatch):
    module, original = host
    deepseek = _importer(_DEEPSEEK, original)
    monkeypatch.setitem(sys.modules, _DEEPSEEK, deepseek)

    def reject(*args, **kwargs):
        raise ValueError("registration refused")

    monkeypatch.setattr(registry, "register", reject)
    with pytest.raises(ValueError, match="registration refused"):
        gate.install()
    assert deepseek.fused_a_gemm_weight_eligible is original
    assert registry.get("fused_a_gemm") is None
