"""platform_seam pins custom AR platform flags to HIP truth (CPU, no sglang).

custom_all_reduce freezes is_cuda()/is_hip()/is_musa() into module globals at
import time. Imported after the seam, it froze _is_cuda=True on HCU, so the
dispatch took the native CA and its NVML NVLink probe failed (RCCL fallback).
_install_ar_hw_truth must:

1. pin an already-imported module's flags to (cuda=False, hip=True, musa=False)
   and record each as a MECH_REBIND platform_detect binding;
2. when the module is not yet imported, pin it right after its first exec,
   whatever the seam answered at that time, then leave sys.meta_path clean;
3. be idempotent.

Run: pytest tests/adapter/test_platform_seam_ar_hw_truth.py -v
"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hcu_adapter import platform_seam, registry  # noqa: E402

_NAME = platform_seam._AR_MODULE
_HW = {"_is_cuda": False, "_is_hip": True, "_is_musa": False}


class _Registry:
    MECH_REBIND = registry.MECH_REBIND

    def __init__(self):
        self.calls = []

    def register(self, op, impl, target, mechanism, **kw):
        self.calls.append((op, impl, target, mechanism))


def _flags(mod):
    return {k: mod.__dict__.get(k) for k in _HW}


def _clean_meta_path():
    sys.meta_path[:] = [
        f for f in sys.meta_path
        if not isinstance(f, platform_seam._ArHwTruthFinder)
    ]


def test_pins_already_imported_module():
    mod = types.ModuleType(_NAME)
    mod._is_cuda, mod._is_hip, mod._is_musa = True, False, False
    reg = _Registry()
    saved = sys.modules.get(_NAME)
    sys.modules[_NAME] = mod
    try:
        platform_seam._install_ar_hw_truth(reg)
        platform_seam._install_ar_hw_truth(reg)
    finally:
        if saved is None:
            sys.modules.pop(_NAME, None)
        else:
            sys.modules[_NAME] = saved
    assert _flags(mod) == _HW
    assert reg.calls == [
        ("platform_detect", False, f"{_NAME}:_is_cuda", registry.MECH_REBIND),
        ("platform_detect", True, f"{_NAME}:_is_hip", registry.MECH_REBIND),
    ]
    assert not any(isinstance(f, platform_seam._ArHwTruthFinder) for f in sys.meta_path)


def test_pins_module_imported_after_seam():
    # Fake package tree whose leaf freezes CUDA semantics at import time,
    # as the real module does once the seam has flipped is_cuda().
    pkg_names = _NAME.split(".")
    saved = {n: sys.modules.pop(n) for n in list(sys.modules)
             if n == pkg_names[0] or n.startswith(pkg_names[0] + ".")}
    with tempfile.TemporaryDirectory() as root:
        path = root
        for part in pkg_names[:-1]:
            path = os.path.join(path, part)
            os.makedirs(path)
            open(os.path.join(path, "__init__.py"), "w").close()
        with open(os.path.join(path, pkg_names[-1] + ".py"), "w") as f:
            f.write("_is_cuda = True\n_is_hip = False\n_is_musa = False\n")
        sys.path.insert(0, root)
        reg = _Registry()
        try:
            platform_seam._install_ar_hw_truth(reg)
            platform_seam._install_ar_hw_truth(reg)
            assert sum(isinstance(f, platform_seam._ArHwTruthFinder)
                       for f in sys.meta_path) == 1
            import importlib

            mod = importlib.import_module(_NAME)
            assert _flags(mod) == _HW
            assert len(reg.calls) == 2
            assert not any(isinstance(f, platform_seam._ArHwTruthFinder)
                           for f in sys.meta_path)
        finally:
            sys.path.remove(root)
            _clean_meta_path()
            for n in list(sys.modules):
                if n == pkg_names[0] or n.startswith(pkg_names[0] + "."):
                    sys.modules.pop(n)
            sys.modules.update(saved)


if __name__ == "__main__":
    test_pins_already_imported_module()
    test_pins_module_imported_after_seam()
    print("PASS")
