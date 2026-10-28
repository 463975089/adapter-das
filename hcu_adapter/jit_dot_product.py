"""JIT sgl_kernel/gemm/dot_product.cuh portability patch — HIP arm.

`dot_product_vec` (bf16 dot-product accumulation inside the tiny_gemm /
attention inner kernels) decomposes `fp32x2_t` — a HIP `float2`, i.e. a
class with an anonymous union member — via structured bindings, which DTK
hipcc rejects (third occurrence of the same ABI fact after rope.cuh and
fused_store_index_cache.cuh). Replace with explicit `.x/.y` member reads;
identical math (two fused multiplies into `acc`).
"""

import logging
import pathlib

LOG = logging.getLogger("hcu.adapter")

_RELPATH = "include/sgl_kernel/gemm/dot_product.cuh"

# (needle, replacement) — exact 0.5.21 strings
_PATCHES = (
    (
        "    const auto [a0, a1] = cast<fp32x2_t>(a[i]);",
        "    fp32x2_t _va = cast<fp32x2_t>(a[i]);"
        " const auto a0 = _va.x, a1 = _va.y;",
    ),
    (
        "    const auto [b0, b1] = cast<fp32x2_t>(b[i]);",
        "    fp32x2_t _vb = cast<fp32x2_t>(b[i]);"
        " const auto b0 = _vb.x, b1 = _vb.y;",
    ),
)


def install():
    import sglang.kernels.jit as _jit_root

    p = pathlib.Path(_jit_root.__file__).parent / _RELPATH
    text = p.read_text(encoding="utf-8")
    registered = []
    for needle, repl in _PATCHES:
        if repl in text:
            continue  # already patched (spawn'd child / second install)
        if needle not in text:
            raise RuntimeError(
                f"jit_dot_product: needle missing in {_RELPATH}: {needle[:70]!r} -- "
                "upstream changed; re-derive (governance §3)"
            )
        text = text.replace(needle, repl)
        registered.append(needle)
    from hcu_adapter.atomic_io import atomic_write

    atomic_write(p, text)

    import hcu_adapter.registry as _reg

    for needle in registered:
        _reg.register(
            "platform_detect",  # ISA-portability patch; header included by
            # several JIT kernels (tiny_gemm, ...) — no single catalog op
            "gfx93x-hipcc-portability",
            target=f"text:sglang/kernels/jit/{_RELPATH}",
            mechanism=_reg.MECH_TEXT,
            detail=needle,
        )
    LOG.warning(
        "HCU adapter: JIT dot_product portability patch installed (%d edits)",
        len(registered),
    )
