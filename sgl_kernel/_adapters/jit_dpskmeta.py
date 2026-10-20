"""JIT paged_mqa_metadata portability — DTK has no NVIDIA cub, only hipCUB.

Upstream v0.5.22 JIT-compiles the DSV4 indexer metadata builder from
`sglang/kernels/jit/csrc/deepseek_v4/paged_mqa_metadata.cuh`. On DTK three
CUDA-isms kill the hipcc build (DSV4-Flash decode graph capture, nmz16
gfx938, 2026-10-20):

1. `#include <cub/block/block_scan.cuh>` — DTK ships hipCUB only
   (`/opt/dtk/include/hipcub`, rocPRIM backend). The AMD path of hipCUB
   declares the CUB-mirroring API in `namespace hipcub` (the
   `using namespace cub` in hipcub/config.hpp is the NVIDIA path), so the
   swap needs a namespace alias alongside the umbrella include. Same
   convention as the tree's own lora/moe_lora_align_kernel.cu.
2. `__shfl_up_sync(0xffffffff, v, o)` and `__shfl_sync(0xffffffff, v, 31)`
   in the Phase-1/2 hand-rolled 32-lane prefix scans — DTK declares only
   the non-mask shuffle family. The explicit-width HIP forms
   `__shfl_up(v, o, 32)` / `__shfl(v, 31, 32)` keep the exact CUDA
   32-lane warp segmentation on wave64 hardware.

Text-patch discipline per jit_ptgq: needle missing in an existing file
raises loud (upstream changed); file absent skips quietly. v1 of this
patch wrote only the include swap; it is rolled back first so the needle
set applies from a known state on already-patched trees.
"""

import logging
import pathlib

LOG = logging.getLogger("hcu.adapter")

_RELPATH = "deepseek_v4/paged_mqa_metadata.cuh"

_V1_REPL = "#include <hipcub/hipcub.hpp>  // HCU: DTK has hipCUB only (namespace cub)"

_NEEDLES = [
    (
        "#include <cub/block/block_scan.cuh>",
        "#include <hipcub/hipcub.hpp>  // HCU: DTK has hipCUB (rocPRIM backend)\n"
        "namespace cub = hipcub;  // HCU: AMD hipCUB mirrors the cub API",
    ),
    (
        "uint32_t y = __shfl_up_sync(0xffffffff, v, o);",
        "uint32_t y = __shfl_up(v, o, 32);  // HCU: no mask-form shuffle, width=32",
    ),
    (
        "running = __shfl_sync(0xffffffff, v, 31);",
        "running = __shfl(v, 31, 32);  // HCU: no mask-form shuffle, width=32",
    ),
]


def install():
    import sglang.kernels.jit as _jit_root
    import sgl_kernel._adapters.registry as _reg

    csrc = pathlib.Path(_jit_root.__file__).parent / "csrc"
    p = csrc / _RELPATH
    if not p.exists():
        LOG.info("jit_dpskmeta: %s absent (tree predates the module)", _RELPATH)
        return
    text = p.read_text(encoding="utf-8")

    # self-heal trees carrying the v1 single-swap patch
    if _V1_REPL in text:
        text = text.replace(
            _V1_REPL, "#include <cub/block/block_scan.cuh>"
        )

    for needle, repl in _NEEDLES:
        if repl in text:
            continue
        if needle not in text:
            raise RuntimeError(
                f"jit_dpskmeta: needle missing in {_RELPATH}: {needle[:60]!r} -- "
                "upstream changed; re-derive (governance §3)"
            )
        text = text.replace(needle, repl)

    p.write_text(text, encoding="utf-8")

    target = f"text:sglang/kernels/jit/csrc/{_RELPATH}"
    for needle, _ in _NEEDLES:
        _reg.register(
            "platform_detect",
            "dtk-hipcc-portability",
            target=target,
            mechanism=_reg.MECH_TEXT,
            detail=needle,
        )
    LOG.warning(
        "HCU adapter: JIT paged_mqa_metadata portability patch installed"
        " (cub->hipcub alias + 32-lane width shuffles, %d needles)",
        len(_NEEDLES),
    )
