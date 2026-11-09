"""platform_seam stale is_cuda rebind (CPU, no GPU, no sglang import).

dsa.utils is imported by the early shared_read_event chain before the seam
swap, so its `is_cuda` stays the original (False on HIP) and in-graph DSA
prefill trips the split-op dispatch assert. _rebind_stale_is_cuda must:

1. rebind only `is_cuda` on an already-imported dsa.utils (is_hip untouched);
2. record it as a MECH_REBIND platform_detect binding;
3. be a no-op when the module is absent or already rebound (idempotent).

Run: pytest tests/adapter/test_platform_seam_stale_bindings.py -v
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hcu_adapter import platform_seam, registry  # noqa: E402

_NAME = "sglang.srt.layers.attention.dsa.utils"


class _Registry:
    MECH_REBIND = registry.MECH_REBIND

    def __init__(self):
        self.calls = []

    def register(self, op, impl, target, mechanism, **kw):
        self.calls.append((op, impl, target, mechanism))


def _hcu_is_cuda():
    return True


def _with_module(mod):
    saved = sys.modules.get(_NAME)
    if mod is None:
        sys.modules.pop(_NAME, None)
    else:
        sys.modules[_NAME] = mod
    return saved


def _restore(saved):
    if saved is None:
        sys.modules.pop(_NAME, None)
    else:
        sys.modules[_NAME] = saved


def test_rebinds_only_is_cuda():
    def orig_is_cuda():
        return False

    def orig_is_hip():
        return True

    mod = types.ModuleType(_NAME)
    mod.is_cuda, mod.is_hip = orig_is_cuda, orig_is_hip
    reg = _Registry()
    saved = _with_module(mod)
    try:
        platform_seam._rebind_stale_is_cuda(_hcu_is_cuda, reg)
        platform_seam._rebind_stale_is_cuda(_hcu_is_cuda, reg)
    finally:
        _restore(saved)
    assert mod.is_cuda is _hcu_is_cuda
    assert mod.is_hip is orig_is_hip
    assert reg.calls == [
        ("platform_detect", _hcu_is_cuda, f"{_NAME}:is_cuda", registry.MECH_REBIND)
    ]


def test_absent_module_is_noop():
    reg = _Registry()
    saved = _with_module(None)
    try:
        platform_seam._rebind_stale_is_cuda(_hcu_is_cuda, reg)
        assert _NAME not in sys.modules
    finally:
        _restore(saved)
    assert reg.calls == []


if __name__ == "__main__":
    test_rebinds_only_is_cuda()
    test_absent_module_is_noop()
    print("PASS")
