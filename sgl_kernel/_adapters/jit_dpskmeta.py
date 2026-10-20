"""JIT paged_mqa_metadata portability — DTK has no NVIDIA cub, only hipCUB.

Upstream v0.5.22 JIT-compiles the DSV4 indexer metadata builder from
`sglang/kernels/jit/csrc/deepseek_v4/paged_mqa_metadata.cuh`, which includes
NVIDIA CUB (`cub/block/block_scan.cuh`) for BlockScan prefix sums. DTK ships
hipCUB only (`/opt/dtk/include/hipcub`, umbrella `hipcub.hpp`); the JIT
include path has no `cub/` tree, so hipcc dies at `file not found` — first
hit: DSV4-Flash decode graph capture on nmz16 gfx938 (2026-10-20), same JIT
family as the c_plan/topk_v2 build failures in the CI mech ledger.

hipCUB intentionally mirrors the cub API inside `namespace cub`, so the
umbrella include is a drop-in replacement; same convention as the tree's
own lora/moe_lora_align_kernel.cu (`#include <hipcub/hipcub.hpp>`).

Text-patch discipline per jit_ptgq: needle missing in an existing file
raises loud (upstream changed); file absent skips quietly.
"""

import logging
import pathlib

LOG = logging.getLogger("hcu.adapter")

_RELPATH = "deepseek_v4/paged_mqa_metadata.cuh"
_NEEDLE = "#include <cub/block/block_scan.cuh>"
_REPL = (
    "#include <hipcub/hipcub.hpp>  // HCU: DTK has hipCUB only (namespace cub)"
)


def install():
    import sglang.kernels.jit as _jit_root
    import sgl_kernel._adapters.registry as _reg

    csrc = pathlib.Path(_jit_root.__file__).parent / "csrc"
    p = csrc / _RELPATH
    if not p.exists():
        LOG.info("jit_dpskmeta: %s absent (tree predates the module)", _RELPATH)
        return
    text = p.read_text(encoding="utf-8")
    if _REPL not in text:
        if _NEEDLE not in text:
            raise RuntimeError(
                f"jit_dpskmeta: needle missing in {_RELPATH}: {_NEEDLE[:60]!r} -- "
                "upstream changed; re-derive (governance §3)"
            )
        p.write_text(text.replace(_NEEDLE, _REPL), encoding="utf-8")

    _reg.register(
        "platform_detect",
        "dtk-no-nvidia-cub",
        target=f"text:sglang/kernels/jit/csrc/{_RELPATH}",
        mechanism=_reg.MECH_TEXT,
        detail=_NEEDLE,
    )
    LOG.warning(
        "HCU adapter: JIT paged_mqa_metadata portability patch installed"
        " (cub block_scan -> hipcub umbrella)"
    )
