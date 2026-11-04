# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""triton extra.cuda libdevice -> hcu/hip redirect (is_cuda semantics gap).

With §0.5 is_cuda semantics, upstream kernels import
`triton.language.extra.libdevice` and triton resolves the module attributes
via the *cuda* backend package (whose ops emit `__nv_*` LLVM intrinsics).
The active triton backend on HCU is *hcu*, and its make_llir hard-rejects
`__nv_` conversions ("please use triton.language.extra.<op> instead of
triton.language.extra.cuda.<op>").

Redirect at the module layer: first import of
`triton.language.extra.cuda.libdevice` returns the **hip** libdevice module
(OCML `__ocml_*` intrinsics — same backend family as hcu), for the symbol
set hip actually implements. Names hip lacks get a loud Missing that points
at the gap (they only fail when actually called, and prior warning notes
them once). Userspace keeps importing cuda-side names unchanged.
"""

import logging
import sys

LOG = logging.getLogger("hcu.adapter")


def install():
    from hcu_adapter import registry

    import importlib

    hip = importlib.import_module("triton.language.extra.hip.libdevice")
    cuda = importlib.import_module("triton.language.extra.cuda.libdevice")

    missing = [
        n
        for n in dir(cuda)
        if not n.startswith("_") and not hasattr(hip, n)
    ]

    class _HipShim:
        """hip libdevice + loud fill-ins for names hip lacks."""

        def __getattr__(self, name):
            v = getattr(hip, name, None)
            if v is not None:
                return v
            if name in _MISSING_REPORTED:
                return _MISSING_REPORTED[name]

            def _absent(*a, **kw):
                raise NotImplementedError(
                    f"triton libdevice.{name}: no hip/OCML equivalent "
                    f"bridged yet (cuda-only symbol). If your kernel hit "
                    f"this, port the call to tl math or add the ocml "
                    f"mapping in hcu_adapter/triton_libdev.py."
                )

            _absent.__name__ = name
            _MISSING_REPORTED[name] = _absent
            LOG.warning(
                "HCU adapter: triton libdevice.%s has no OCML mapping; "
                "calls will raise (gap list)", name,
            )
            return _absent

    _MISSING_REPORTED = {}
    shim = _HipShim()
    sys.modules["triton.language.extra.cuda.libdevice"] = shim

    # also alias the whole cuda package's libdevice attr if already imported
    try:
        import triton.language.extra.cuda as _tcu

        _tcu.libdevice = shim
    except Exception:
        pass

    registry.register(
        "triton_libdevice",
        shim,
        target="triton.language.extra.cuda.libdevice",
        mechanism=registry.MECH_REBIND,
        detail=f"hip OCML redirect; {len(missing)} cuda-only names loud-stubbed",
    )
    LOG.warning(
        "HCU adapter: triton extra.cuda libdevice redirected to hip/OCML "
        "(%d names loud-stubbed: %s...)",
        len(missing), ", ".join(missing[:6]),
    )
