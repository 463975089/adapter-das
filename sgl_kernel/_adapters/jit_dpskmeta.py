"""JIT DSV4-family portability needles — DTK hipcc vs CUDA-isms.

Upstream v0.5.22 JIT-compiles the DeepSeek-V4 attention/indexer kernel
family from `sglang/kernels/jit/csrc/deepseek_v4/`. On DTK gfx938 several
CUDA-isms kill the hipcc build (DSV4-Flash boot chain, nmz16, 2026-10-20;
walking the same family the CI mech ledger already mapped as the
topk_v2/c_plan failures):

paged_mqa_metadata.cuh (module sgl_kernel_jit_dpsk_v4_metadata):
1. `#include <cub/block/block_scan.cuh>` — DTK ships hipCUB only
   (`/opt/dtk/include/hipcub`, rocPRIM backend). The AMD path of hipCUB
   declares the CUB-mirroring API in `namespace hipcub` (the
   `using namespace cub` in hipcub/config.hpp is the NVIDIA path), so the
   swap needs a namespace alias alongside the umbrella include.
2. mask-form `__shfl_up_sync/__shfl_sync(0xffffffff, ...)` in the
   Phase-1/2 hand-rolled 32-lane prefix scans — DTK declares only the
   non-mask family. Explicit-width HIP forms keep the exact CUDA 32-lane
   warp segmentation on wave64.

c_plan.cuh (module sgl_kernel_jit_dpsk_v4_compress_plan):
3. `const auto [ragged_id, batch_id] = unpack_w(plan_w);` — structured
   binding into HIP_vector_type (anonymous union, indecomposable by
   clang); the 4th site of the ABI fact already needled for rope / dot
   product / per-token-quant. unpack_w returns a real uint2, so member
   reads are type-identical.

Backlog from the same scan (no build error yet — land with evidence):
store.cuh / main_norm_rope.cuh / mega_moe_pre_dispatch.cuh /
silu_and_mul_masked_post_quant.cuh cast<fp32x2_t> binds,
topk_bf16_small.cuh mask shuffles + laneid PTX, c4.cuh uint4 bind,
topk_v2.cuh:1141 split_plan bind.

Text-patch discipline per jit_ptgq: needle missing in an existing file
raises loud (upstream changed); file absent skips quietly. The v1 of this
patch wrote only the include swap; it is rolled back first so the needle
set applies from a known state on already-patched trees.
"""

import logging
import os
import pathlib
import tempfile

LOG = logging.getLogger("hcu.adapter")


def _atomic_write(p: pathlib.Path, text: str) -> None:
    """tmp-file + os.replace so concurrent install() runs (parent + 8 TP
    scheduler children import sitecustomize within seconds of each other)
    can never tear the file: readers see either the old or the new content
    in full. A torn non-atomic write cost us a 119-line tail of
    paged_mqa_metadata.cuh on nmz16 (2026-10-20, diagnosed via tree git diff).
    """
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".hcu_dpsk_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

_V1_REPL = "#include <hipcub/hipcub.hpp>  // HCU: DTK has hipCUB only (namespace cub)"

_PATCHES = {
    "deepseek_v4/paged_mqa_metadata.cuh": [
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
    ],
    "deepseek_v4/c_plan.cuh": [
        (
            "    const auto [ragged_id, batch_id] = unpack_w(plan_w);",
            "    const uint2 unpacked_w = unpack_w(plan_w);  // HCU: uint2 not decomposable\n"
            "    const uint32_t ragged_id = unpacked_w.x;\n"
            "    const uint32_t batch_id = unpacked_w.y;",
        ),
    ],
    # main decode attention kernel (module dpsk_v4_main_k_norm_rope_*):
    # bare __syncwarp (rope precedent -> __syncwarp_full_mask shim), float2
    # `freq` structured bind, mask-form xor shuffle in the butterfly reduce.
    "deepseek_v4/main_norm_rope.cuh": [
        (
            "  __syncwarp();",
            "  __syncwarp_full_mask();  // HCU portability shim (jit_rope precedent)",
        ),
        (
            "  const auto [freq_real, freq_imag] = freq;",
            "  const float freq_real = freq.x;  // HCU: float2 not decomposable\n"
            "  const float freq_imag = freq.y;",
        ),
        (
            "        const float other = __shfl_xor_sync(0xFFFFFFFFu, data[i], mask, kWarpThreads);",
            "        const float other = __shfl_xor(data[i], mask, kWarpThreads);"
            "  // HCU: no mask-form shuffle",
        ),
    ],
    # top-k bf16 small: %laneid PTX (concat_mla precedent -> utils.cuh HIP
    # identity) + two mask-form shuffles (suffix-sum guarded by
    # lane+offset<kWarpThreads; lane-31 broadcast).
    "deepseek_v4/topk_bf16_small.cuh": [
        (
            '  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));',
            "  // HCU: %laneid is NVIDIA-only; canonical HIP identity below\n"
            "  lane_id = static_cast<uint32_t>(::sglang::device::get_lane_id());",
        ),
        (
            "    const auto t = __shfl_down_sync(device::warp::kFullMask, inc, offset);",
            "    const auto t = __shfl_down(inc, offset);  // HCU: no mask-form shuffle",
        ),
        (
            "  warp_base = __shfl_sync(warp::kFullMask, warp_base, kWarpThreads - 1);",
            "  warp_base = __shfl(warp_base, kWarpThreads - 1);"
            "  // HCU: no mask-form shuffle",
        ),
    ],
    # flashmla scheduler metadata: butterfly xor reduce, partners stay within
    # the 32-lane group for offsets <= 16.
    "deepseek_v4/flashmla_sched_meta.cuh": [
        (
            "    partial += __shfl_xor_sync(uint32_t(-1), partial, offset);",
            "    partial += __shfl_xor(partial, offset);  // HCU: no mask-form shuffle",
        ),
    ],
    # prefill norm+rope variant: same float2 freq bind as main_norm_rope.
    "deepseek_v4/fused_norm_rope.cuh": [
        (
            "    const auto [freq_real, freq_imag] = freq;",
            "    const float freq_real = freq.x;  // HCU: float2 not decomposable\n"
            "    const float freq_imag = freq.y;",
        ),
    ],
}


def install():
    import sglang.kernels.jit as _jit_root
    import sgl_kernel._adapters.registry as _reg

    csrc = pathlib.Path(_jit_root.__file__).parent / "csrc"
    for relpath, needles in _PATCHES.items():
        p = csrc / relpath
        if not p.exists():
            LOG.info("jit_dpskmeta: %s absent (tree predates the module)", relpath)
            continue
        text = p.read_text(encoding="utf-8")

        # self-heal trees carrying the v1 single-swap patch
        if relpath.endswith("paged_mqa_metadata.cuh") and _V1_REPL in text:
            text = text.replace(
                _V1_REPL, "#include <cub/block/block_scan.cuh>"
            )

        for needle, repl in needles:
            if repl in text:
                continue
            if needle not in text:
                raise RuntimeError(
                    f"jit_dpskmeta: needle missing in {relpath}:"
                    f" {needle[:60]!r} -- upstream changed; re-derive"
                    " (governance §3)"
                )
            text = text.replace(needle, repl)

        _atomic_write(p, text)

        target = f"text:sglang/kernels/jit/csrc/{relpath}"
        for needle, _ in needles:
            _reg.register(
                "platform_detect",
                "dtk-hipcc-portability",
                target=target,
                mechanism=_reg.MECH_TEXT,
                detail=needle,
            )
        LOG.warning(
            "HCU adapter: JIT %s portability patch installed (%d needles)",
            relpath,
            len(needles),
        )
