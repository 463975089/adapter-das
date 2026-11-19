# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""R-08 A-tier platform env defaults: semantics pin.

platform_seam.platform_env_defaults() is pure os.environ.setdefault calls —
it must (a) fill every A-tier var when absent, (b) NEVER clobber an explicit
export (env三原则: explicit override is top priority), (c) be idempotent.
No GPU / no sglang import needed — the function is import-safe standalone.

Pytest-collectable (L1 kernels segment collects tests/adapter); direct run
also works: python tests/adapter/test_platform_env.py
(2026-10-19: module-level sys.exit crashed pytest collection -> INTERNALERROR,
refactored to function + __main__ guard, same 3 checks).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hcu_adapter.platform_seam import platform_env_defaults  # noqa: E402

A_TIER = {
    "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
    "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
    "SGLANG_RUST_BUILD_MODE": "never",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
    "HCU_DEEPGEMM_BRIDGE": "1",
}


class _Env:
    """save A_TIER on enter, restore on exit -- order-independent tests."""

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in A_TIER}
        for k in A_TIER:
            os.environ.pop(k, None)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_platform_env_defaults():
    # 1. fills when absent
    with _Env():
        platform_env_defaults()
        got = {k: os.environ.get(k) for k in A_TIER}
        assert got == A_TIER, f"fills-absent: {got}"

    # 2. explicit export wins (unset-one-by-one across both value kinds)
    with _Env():
        for k, explicit in (
            ("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "true"),
            ("HCU_DEEPGEMM_BRIDGE", "0"),
            ("SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND", "rust"),
        ):
            os.environ[k] = explicit
        platform_env_defaults()
        assert os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] == "true", (
            "explicit-wins")
        assert os.environ["HCU_DEEPGEMM_BRIDGE"] == "0", "explicit-wins"
        assert os.environ["SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"] == "rust", (
            "explicit-wins")

    # 3. idempotent
    with _Env():
        platform_env_defaults()
        platform_env_defaults()
        assert all(os.environ.get(k) for k in A_TIER), "idempotent"


def test_gfx_guard_precedes_runner_utils_import():
    import builtins
    import functools
    from types import ModuleType, SimpleNamespace
    from unittest.mock import Mock, patch
    from hcu_adapter import hcu_env, hip_metadata, platform_seam, registry

    modules = {}
    for name in (
        "sglang", "sglang.srt", "sglang.srt.platforms", "sglang.srt.utils",
        "sglang.srt.utils.common", "sglang.srt.utils.log_utils", "sglang.srt.model_executor",
        "sglang.srt.model_executor.runner_utils",
        "sglang.srt.layers", "sglang.srt.layers.moe", "sglang.srt.layers.moe.utils",
    ):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)
    common = modules["sglang.srt.utils.common"]
    package = modules["sglang.srt.utils"]
    common.is_gfx1250_supported = lambda: True
    package.is_gfx1250_supported = common.is_gfx1250_supported
    common.is_cuda = lambda: False
    package.is_cuda = common.is_cuda
    common._check_cuda_device_version = lambda: True
    common.is_sm90_supported = functools.lru_cache()(functools.partial(common._check_cuda_device_version))
    package.is_sm90_supported = common.is_sm90_supported
    original_capability = Mock(return_value=(7, 5))
    original_capability._hcu_device_capability = False
    original_current_device = Mock(return_value=7)
    original_current_device._hcu_current_device = False
    common.get_nvgpu_memory_capacity = lambda: 0
    common.get_device_sm = lambda: 0
    common.get_device_core_count = lambda: 0
    common.get_device_capability = lambda device_id=0: torch.cuda.get_device_capability(device_id)
    package.get_device_capability = common.get_device_capability
    observed = []
    event_module = SimpleNamespace(make_external_event=lambda _: None)

    def get_runner_attribute(name):
        if name != "shared_read_event":
            raise AttributeError(name)
        observed.append((common.is_gfx1250_supported(), package.is_gfx1250_supported(),
                         common.is_cuda(), package.is_cuda(), package.is_sm90_supported()))
        return event_module

    modules["sglang.srt.model_executor.runner_utils"].__getattr__ = get_runner_attribute

    def parse_device(value):
        if not isinstance(value, str):
            return value
        kind, _, index = value.partition(":")
        return SimpleNamespace(type=kind, index=int(index) if index else None)

    torch = SimpleNamespace(
        version=SimpleNamespace(hip="6.4"),
        cuda=SimpleNamespace(is_available=lambda: True, is_initialized=Mock(return_value=False),
                             get_device_capability=original_capability,
                             current_device=original_current_device),
        float8_e4m3fn=object(), finfo=lambda _: SimpleNamespace(max=448),
        device=parse_device,
    )
    with patch.dict(sys.modules, {**modules, "torch": torch}), patch.object(
        hcu_env, "resolve", return_value=("cuda", "default")
    ), patch.object(registry, "register"), patch("hcu_adapter.lightop_metadata.install"), patch.object(
        builtins, "FP8_E4M3_MAX", None, create=True
    ), patch.object(builtins, "FP8_E4M3_MIN", None, create=True), patch.object(
        hip_metadata, "compute_capability", return_value=(9, 3)
    ) as capability, patch.object(hip_metadata, "current_device", return_value=3) as current:
        platform_seam.install()
        assert torch.cuda.current_device() == 3
        original_current_device.assert_not_called()
        assert package.get_device_capability is common.get_device_capability
        assert common.get_device_capability(3) == (9, 3)
        capability.assert_called_once_with(3)
        original_capability.assert_not_called()
        assert torch.cuda.get_device_capability("cuda:2") == (9, 3)
        capability.assert_called_with(2)
        assert torch.cuda.get_device_capability() == (9, 3)
        capability.assert_called_with(None)
        assert torch.cuda.get_device_capability(SimpleNamespace(type="cuda", index=5)) == (9, 3)
        capability.assert_called_with(5)
        torch.cuda.is_initialized.return_value = True
        assert torch.cuda.current_device() == 7
        original_current_device.assert_called_once()
        assert common.get_device_capability(4) == (7, 5)
        original_capability.assert_called_once_with(4)
        torch.cuda.is_initialized.return_value = False
        current.return_value = None
        assert torch.cuda.current_device() == 7
        capability.return_value = None
        assert torch.cuda.get_device_capability(2) == (7, 5)
    assert observed and all(pair == (False, False, True, True, False) for pair in observed)


def test_moe_sbo_guard_uses_physical_hip_identity_and_updates_aliases():
    from types import ModuleType, SimpleNamespace
    from unittest.mock import patch
    import pytest
    from hcu_adapter import platform_seam, registry

    module = ModuleType("sglang.srt.layers.moe.utils")
    scheduler = ModuleType("sglang.srt.managers.scheduler")
    torch = SimpleNamespace(version=SimpleNamespace(hip="6.4"), cuda=SimpleNamespace(
        get_device_capability=lambda: (9, 3),
    ))
    requested = SimpleNamespace(sbo=True, tail=17)
    state = SimpleNamespace()
    module.__dict__.update(torch=torch, requested=requested, state=state, is_cuda=lambda: True)
    exec("def initialize_moe_config():\n"
         "    state.sbo_enabled = requested.sbo\n"
         "    if state.sbo_enabled and is_cuda() and torch.cuda.get_device_capability()[0] == 9:\n"
         "        raise ValueError('SM90 SBO rejected')\n"
         "    state.tail = requested.tail\n", module.__dict__)
    original = module.initialize_moe_config
    scheduler.initialize_moe_config = original
    with patch.dict(sys.modules, {
        "torch": torch, module.__name__: module, scheduler.__name__: scheduler,
    }), patch.object(registry, "register"):
        with pytest.raises(ValueError, match="SM90"):
            original()
        platform_seam._install_moe_hardware_guard(module)
        assert scheduler.initialize_moe_config is module.initialize_moe_config
        scheduler.initialize_moe_config()
        assert state.sbo_enabled and state.tail == 17
        assert module.is_cuda()
        requested.tail = 29
        module.initialize_moe_config()
        assert state.tail == 29
        torch.version.hip = None
        with pytest.raises(ValueError, match="SM90"):
            module.initialize_moe_config()


if __name__ == "__main__":
    try:
        test_platform_env_defaults()
        print("PASS test_platform_env_defaults")
        print("RESULT: PASS")
    except AssertionError as e:
        print(f"FAIL test_platform_env_defaults: {e}")
        print("RESULT: FAIL")
        sys.exit(1)
