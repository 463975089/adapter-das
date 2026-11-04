# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Triton GCN PDL stripper — gfx936 assembler compatibility.

is_cuda semantics let upstream kernels enable Programmatic Dependent Launch
whenever `torch.cuda.get_device_capability()` reports major>=9, which HCU
does (gfx93x -> (9,3)) even though the HIP/ROCm toolchain on this install
has no `griddepcontrol.*` instruction support — the gcn->hsaco stage fails
with "invalid instruction: griddepcontrol.launch_dependents" for every
kernel compiled with ENABLE_PDL=[True] (fused qk rmsnorm-rope-gate, decode
attention, fla layernorm-gated, ...).

The PDL instructions the triton pipeline emits (`griddepcontrol.launch_dependents`
+ `griddepcontrol.wait`) are pure scheduling hints: upstream GPU code waits
for the prior kernel at the point it needs its results when PDL is off.
Stripping both lines at the amdgcn stage is therefore semantically identical
to running with PDL disabled — exactly what the same kernel does on hardware
without sm90 PDL. This rebinds `HCUBackend.make_amdgcn` to drop those lines
from the asm text before it reaches clang; every kernel benefits, no
per-call-site patch needed.
"""

import logging

LOG = logging.getLogger("hcu.adapter")

_PDL_LINES = (
    "griddepcontrol.launch_dependents",
    "griddepcontrol.wait",
)


def install():
    from hcu_adapter import registry
    import triton.backends.hcu.compiler as _hcu

    if getattr(_hcu, "_hcu_pdl_stripped", False):
        return
    owner = getattr(_hcu, "HIPBackend", None) or getattr(
        _hcu, "HCUBackend", None
    )
    if owner is None or not hasattr(owner, "make_amdgcn"):
        raise RuntimeError(
            "triton_pdl: no HIPBackend/HCUBackend.make_amdgcn in "
            "triton.backends.hcu.compiler -- upstream layout changed"
        )
    orig = owner.make_amdgcn

    @staticmethod
    def make_amdgcn_no_pdl(src, metadata, options):
        text = src
        if "griddepcontrol." in text:
            kept = [
                line
                for line in text.splitlines(True)
                if not any(p in line for p in _PDL_LINES)
            ]
            text = "".join(kept)
        return orig(text, metadata, options)

    owner.make_amdgcn = make_amdgcn_no_pdl
    _hcu._hcu_pdl_stripped = True

    registry.register(
        "triton_pdl",
        make_amdgcn_no_pdl,
        target="triton.backends.hcu:HCUBackend.make_amdgcn",
        mechanism=registry.MECH_REBIND,
        detail="strip griddepcontrol.* asm lines (gfx936 has no PDL; "
               "semantically = PDL-off)",
    )
    LOG.warning(
        "HCU adapter: triton hcu make_amdgcn PDL stripper installed "
        "(gfx936 griddepcontrol unsupported)"
    )
