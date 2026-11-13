# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""PCP admission reads the same operator knob the bridge records.

`select_attention` is the function that actually decides which attention class
runs, so it has to honour `HCU_VLLM_PCP_IMPL` and the `vllm_pcp` kill switch. If
only the bridge's bookkeeping read them, `stock` (the default) and the kill switch
would be inert while the owned classes were still built and executed, and an
operator triaging a fault could not get back to the native path.

CPU only: the native selector is replaced, so no vLLM platform is required.
"""
from __future__ import annotations

import pytest

from hcu_adapter.bridges import vllm_pcp
from hcu_adapter.kernels import pcp_fork


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("HCU_VLLM_PCP_IMPL", raising=False)
    monkeypatch.delenv("HCU_DISABLE_OPS", raising=False)
    return monkeypatch


def test_default_mode_is_stock_and_matches_the_bridge(clean_env):
    assert pcp_fork.selected_mode() == "stock"
    assert pcp_fork.selected_mode() == vllm_pcp.selected_mode()


def test_native_mode_is_read_by_both_readers(clean_env):
    clean_env.setenv("HCU_VLLM_PCP_IMPL", "native")
    assert pcp_fork.selected_mode() == "native"
    assert pcp_fork.selected_mode() == vllm_pcp.selected_mode()


def test_kill_switch_wins_over_an_explicit_opt_in(clean_env):
    clean_env.setenv("HCU_VLLM_PCP_IMPL", "native")
    clean_env.setenv("HCU_DISABLE_OPS", "vllm_pcp")
    assert pcp_fork.selected_mode() == "stock"
    assert pcp_fork.selected_mode() == vllm_pcp.selected_mode()


def test_unknown_mode_is_refused(clean_env):
    clean_env.setenv("HCU_VLLM_PCP_IMPL", "bogus")
    with pytest.raises(pcp_fork.PcpAttentionError, match="stock or native"):
        pcp_fork.selected_mode()


class _NativeSelector:
    """Stands in for vllm's native selector so no vLLM is needed."""

    def __init__(self):
        self.calls = []

    def _select_dsv4_attn_cls(self, cfg):
        self.calls.append(cfg)
        return "native-attention"


@pytest.fixture
def native(monkeypatch):
    """Patch the vLLM module `select_attention` imports lazily.

    The import happens inside the function, so the fake has to be installed in
    `sys.modules` before the call rather than patched on the real module.
    """
    import sys
    from types import ModuleType

    selector = _NativeSelector()
    # `select_attention` runs `from vllm.models.deepseek_v41.amd import model`,
    # so Python imports the whole parent chain before it reaches the leaf. A leaf
    # stub alone is not enough: without vLLM the parents are missing, and with
    # vLLM installed the real `amd/__init__` pulls in modules that read the leaf
    # we replaced. Build every level with an empty __path__ so the import
    # machinery stops at our doubles, and bind the leaf as a parent attribute.
    leaf = "vllm.models.deepseek_v41.amd.model"
    chain = ("vllm", "vllm.models", "vllm.models.deepseek_v41",
             "vllm.models.deepseek_v41.amd", leaf)
    for name in chain:
        package = ModuleType(name)
        package.__path__ = []
        if name == leaf:
            package._select_dsv4_attn_cls = selector._select_dsv4_attn_cls
        monkeypatch.setitem(sys.modules, name, package)
    # Bind each child as an attribute of its parent, so
    # `from ...amd import model` resolves to the double.
    for parent, child in zip(chain, chain[1:]):
        setattr(sys.modules[parent], child.rsplit(".", 1)[-1], sys.modules[child])
    return selector


def test_stock_mode_never_builds_the_owned_classes(clean_env, native, monkeypatch):
    def explode():
        raise AssertionError("owned PCP classes were built in stock mode")

    monkeypatch.setattr(pcp_fork, "build_classes", explode)
    monkeypatch.setattr(pcp_fork, "validate_config", lambda cfg: True)
    cfg = object()
    assert pcp_fork.select_attention(cfg) == "native-attention"
    assert native.calls == [cfg]


def test_kill_switch_never_builds_the_owned_classes(clean_env, native, monkeypatch):
    clean_env.setenv("HCU_VLLM_PCP_IMPL", "native")
    clean_env.setenv("HCU_DISABLE_OPS", "vllm_pcp")
    monkeypatch.setattr(
        pcp_fork, "build_classes",
        lambda: (_ for _ in ()).throw(AssertionError("built under kill switch")))
    monkeypatch.setattr(pcp_fork, "validate_config", lambda cfg: True)
    assert pcp_fork.select_attention(object()) == "native-attention"


def test_native_mode_builds_the_owned_classes(clean_env, native, monkeypatch):
    clean_env.setenv("HCU_VLLM_PCP_IMPL", "native")
    monkeypatch.setattr(pcp_fork, "validate_config", lambda cfg: True)
    monkeypatch.setattr(pcp_fork, "build_classes", lambda: {"attention": "owned-attention"})
    assert pcp_fork.select_attention(object()) == "owned-attention"
    assert native.calls == []


def test_non_pcp_config_stays_native_even_in_native_mode(clean_env, native, monkeypatch):
    clean_env.setenv("HCU_VLLM_PCP_IMPL", "native")
    monkeypatch.setattr(pcp_fork, "validate_config", lambda cfg: False)
    monkeypatch.setattr(
        pcp_fork, "build_classes",
        lambda: (_ for _ in ()).throw(AssertionError("built for a non-PCP config")))
    cfg = object()
    assert pcp_fork.select_attention(cfg) == "native-attention"
    assert native.calls == [cfg]
