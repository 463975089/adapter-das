# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Explicit Flash must use a unique active process-local capability owner."""

from types import ModuleType, SimpleNamespace
import os
import sys

import pytest

from hcu_adapter.bridges import vllm as bridge
from hcu_adapter.extensions import ExtensionReport, ExtensionStatus


CAP = "hcu-flash-attention"


def report(name="flash", status=ExtensionStatus.ACTIVE, capabilities=(CAP,)):
    return ExtensionReport(name, name, status, frozenset(capabilities), "provider missing")


@pytest.fixture
def platform(monkeypatch):
    """CPU platform/config doubles; real Adapter selector/report storage."""
    class RocmPlatform:
        @classmethod
        def get_device_capability(cls):
            return "gfx936-test-only"

        @classmethod
        def get_attn_backend_cls(cls, selected, config, num_heads=None):
            return selected, config, num_heads

    rocm = ModuleType("vllm.platforms.rocm")
    rocm.RocmPlatform = RocmPlatform
    registry = ModuleType("vllm.v1.attention.backends.registry")
    registry.AttentionBackendEnum = SimpleNamespace(FLASH_ATTN="FLASH", TRITON_ATTN="TRITON")
    monkeypatch.setitem(sys.modules, rocm.__name__, rocm)
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    monkeypatch.delitem(bridge.__dict__, "HcuPlatform", raising=False)
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", bridge.ProcessLocalActions())
    imports, validations = [], []

    class Backend:
        @classmethod
        def validate_configuration(cls, **kwargs):
            validations.append(kwargs)
            return []

    original = bridge.import_module

    def load(name):
        if name == "hcu_vllm_das.attention.flash_attn":
            imports.append(name)
            return SimpleNamespace(HcuFlashAttentionBackend=Backend)
        return original(name)

    monkeypatch.setattr(bridge, "import_module", load)
    config = SimpleNamespace(_asdict=lambda: {"block_size": 64})
    return bridge.HcuPlatform, config, imports, validations


@pytest.mark.parametrize("reports", [
    (),
    (report(status=ExtensionStatus.FAILED),),
    (report(), report("another")),
    (report(), report("failed-duplicate", ExtensionStatus.FAILED)),
    (report(capabilities=("unrelated",)),),
])
def test_missing_failed_or_ambiguous_owner_refuses_before_backend_import(platform, reports):
    cls, config, imports, validations = platform
    bridge._RUNTIME_ACTIONS.record_extension_reports(reports)
    with pytest.raises(bridge.LatchedVllmRuntimeError, match="hcu-flash-attention"):
        cls.get_attn_backend_cls("FLASH", config)
    assert not imports and not validations


def test_unique_flash_owner_is_independent_of_clean_failed_legacy_bundle(platform):
    cls, config, imports, validations = platform
    active = report()
    bridge._RUNTIME_ACTIONS.record_extension_health(active, lambda: None)
    bridge._RUNTIME_ACTIONS.record_extension_reports((
        report("legacy", ExtensionStatus.FAILED, ("aiter-int8-moe",)), active,
    ))
    assert cls.get_attn_backend_cls("FLASH", config) == ("FLASH", config, None)
    assert imports == validations == []


@pytest.mark.parametrize("selected,use_mla,use_sparse,want", [
    (None, False, False, "TRITON"),
    (None, True, False, None),
    (None, True, True, None),
    ("TRITON", False, False, "TRITON"),
    ("OTHER", False, False, "OTHER"),
])
def test_official_routes_do_not_require_or_import_flash(platform, selected, use_mla, use_sparse, want):
    cls, config, imports, validations = platform
    config.use_mla, config.use_sparse = use_mla, use_sparse
    assert cls.get_attn_backend_cls(selected, config, 8) == (want, config, 8)
    assert not imports and not validations


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires actual fork")
def test_actual_fork_does_not_accept_parent_active_report(platform):
    cls, config, _, _ = platform
    bridge._RUNTIME_ACTIONS.record_extension_reports((report(),))
    pid = os.fork()
    if pid == 0:
        try:
            cls.get_attn_backend_cls("FLASH", config)
        except bridge.LatchedVllmRuntimeError:
            os._exit(0 if not bridge.extension_reports() else 2)
        except BaseException:
            os._exit(3)
        os._exit(4)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert bridge.extension_reports() == (report(),)
