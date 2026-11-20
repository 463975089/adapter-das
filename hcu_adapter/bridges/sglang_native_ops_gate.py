# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Serve-time hard gate: native sgl_kernel ops (AOT) must be present.

Runtime ninja JIT as a serve fallback is no longer tolerated (2026-11-19):
a pure-python shadow install makes every covered op silently degrade to
minute-plus per-variant hipcc compiles on first use -- cold serve starts
burned tens of minutes this way. When a server launch (ModelRunner init)
finds no native ``sgl_kernel::*`` registration in the process, it now
aborts with an actionable error instead of degrading. ``HCU_REQUIRE_NATIVE=0``
downgrades the error to a loud warning for dev flows that deliberately run
without the native wheel (tier tests, emu bring-up).
"""

from __future__ import annotations

import importlib
import logging

LOG = logging.getLogger("hcu.adapter")

_TARGET_MODULE = "sglang.srt.model_executor.model_runner"


def install(_he=None) -> None:
    from hcu_adapter.infra import hcu_env
    from hcu_adapter.infra import registry

    mode, _src = hcu_env.resolve("HCU_REQUIRE_NATIVE")
    required = str(mode).strip().lower() not in ("", "0", "off", "none", "false")
    registry.register(
        "platform_detect",
        "native-ops-required" if required else "native-ops-warn",
        # distinct target: the bare :ModelRunner.__init__ is already bound to
        # the model_env seam (registry keys duplicate checks per target)
        target=_TARGET_MODULE + ":ModelRunner.__init__#native-ops-gate",
        mechanism=registry.MECH_SEAM,
        detail="serve-time AOT gate: launch aborts when no native "
        "sgl_kernel ops are registered (ninja-JIT fallback not "
        "tolerated); HCU_REQUIRE_NATIVE=0 downgrades to warning",
    )

    orig_import = importlib.import_module
    patched = [False]

    def _check() -> None:
        from sgl_kernel._native import native_available

        if native_available():
            LOG.warning(
                "HCU adapter: native sgl_kernel ops present (AOT tier armed)"
            )
            return
        message = (
            "HCU adapter: native sgl_kernel ops are NOT registered in this "
            "process -- serving would silently fall back to runtime ninja "
            "JIT (minute-plus per-variant hipcc compiles on first use). "
            "Refusing to serve. Install the native wheel (bash "
            "packaging/build.sh && pip install --no-deps dist/sgl_kernel-*.whl) "
            "or build in place (AMDGPU_TARGETS=gfx936 python setup_hip.py "
            "build_ext --inplace). HCU_REQUIRE_NATIVE=0 downgrades this "
            "error to a warning (dev only)."
        )
        if required:
            raise RuntimeError(message)
        LOG.warning("%s", message)

    def _import(name, *args, **kwargs):
        mod = orig_import(name, *args, **kwargs)
        if name == _TARGET_MODULE and not patched[0]:
            patched[0] = True
            runner_cls = getattr(mod, "ModelRunner", None)
            if runner_cls is not None and not getattr(
                runner_cls, "_hcu_native_ops_gated", False
            ):
                runner_cls._hcu_native_ops_gated = True
                orig_init = runner_cls.__init__

                def _init(self, *args, **kwargs):
                    _check()
                    return orig_init(self, *args, **kwargs)

                runner_cls.__init__ = _init
                LOG.warning(
                    "HCU adapter: native-ops serve gate armed (%s)",
                    "required" if required else "warn-only",
                )
        return mod

    importlib.import_module = _import
