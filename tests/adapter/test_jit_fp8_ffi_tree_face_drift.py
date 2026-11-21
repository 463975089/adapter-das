# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tree-face drift lock: _install_targets must survive absent factories.

Upstream v0.5.23 dropped sglang.kernels.ops.moe.dsv4._jit_silu_mul_quant_
contig_module (varlen remains). The old install() loop hard-getattr'd every
target and the AttributeError aborted the whole bridge bootstrap mid-loop,
which cascaded into every L1 gate that boots the bridge (hcu_config /
aiter_moe_gate first, 2026-11-14 bw47 run). This locks the pit: an absent
factory is skipped loudly, the surviving targets are still installed and
registered.
"""

import sys
import types

import hcu_adapter.features.jit.jit_fp8_ffi as jff


class _Registry:
    MECH_REBIND = "rebind"

    def __init__(self):
        self.targets = []

    def register(self, *args, **kwargs):
        self.targets.append(kwargs.get("target"))


def test_absent_factory_skipped_survivors_installed(monkeypatch):
    present_mod = types.ModuleType("tests_face_present")
    present_mod.keep = object()
    absent_mod = types.ModuleType("tests_face_absent")  # attribute intentionally missing
    monkeypatch.setitem(sys.modules, "tests_face_present", present_mod)
    monkeypatch.setitem(sys.modules, "tests_face_absent", absent_mod)

    installed = []
    monkeypatch.setattr(
        jff, "_install_factory", lambda factory, entry: installed.append((factory, entry))
    )

    registry = _Registry()
    count = jff._install_targets(
        [
            ("tests_face_present", "keep", "run"),
            ("tests_face_absent", "gone", "run"),
        ],
        registry,
    )

    assert count == 1
    assert installed == [(present_mod.keep, "run")]
    assert registry.targets == ["tests_face_present.keep"]
