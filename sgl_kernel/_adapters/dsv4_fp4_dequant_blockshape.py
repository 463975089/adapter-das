"""DSV4 fp4->fp8 expert dequant: align the advertised block shape.

``SGLANG_DSV4_FP4_DEQUANT=1`` routes DSV4 mxfp4 routed experts through
``cast_e2m1fn_to_e4m3fn`` (fp8.py process_weights_after_loading, plain-else
arm), which unpacks the weights but KEEPS the MXFP4 (1,32) row-fine scale
grid. ``self.weight_block_size`` still advertises the checkpoint's
``[32, 32]``, so the triton/aiter W8A8 block path asserts
``cdiv(N, 32) == N`` in invoke_fused_moe_kernel (gfx938 nmz26 boot
2026-10-21 20:05: V4.1-Flash dp8ep8+dspark, a2a=none).

The fix is nominal: ``[1, 32]`` describes the produced layout exactly
(group_n=1 is generic in the fused-moe kernel: ``offs_bn // group_n``),
and keeps activation per-token-group quant at 32 to match the weight
groups. [1,32] is an established config in the same file (MXFP8 path).
Numerics untouched -- scales and weights are consumed as produced.
"""

import logging
import pathlib

LOG = logging.getLogger("hcu.adapter")

_NEEDLE = (
    '                self.is_fp4_expert = False\n'
    '                logger.warning_once("Dequantized FP4 expert weights to FP8.")'
)
_REPL = (
    '                self.is_fp4_expert = False\n'
    '                logger.warning_once("Dequantized FP4 expert weights to FP8.")\n'
    '                # HCU: cast keeps the MXFP4 (1,32) row-fine scale grid, but\n'
    '                # weight_block_size still says [32,32] -- the W8A8 block path\n'
    '                # would assert cdiv(N,32)==N on the scale rows. [1,32] matches\n'
    '                # the produced layout exactly (group_n=1 is generic in the\n'
    '                # fused-moe kernel; MXFP8 uses the same shape).\n'
    '                self.weight_block_size = [1, 32]'
)


def install():
    import sglang.srt.layers.quantization.fp8 as _fp8

    p = pathlib.Path(_fp8.__file__)
    text = p.read_text(encoding="utf-8")
    if _REPL in text:
        return  # already applied
    if _NEEDLE not in text:
        LOG.warning(
            "HCU adapter: fp4-dequant block-shape needle changed upstream "
            "(fp8.py); SGLANG_DSV4_FP4_DEQUANT MoE may assert on scale rows "
            "-- gap list"
        )
        return
    from sgl_kernel._adapters.atomic_io import atomic_write

    atomic_write(p, text.replace(_NEEDLE, _REPL))
    import sgl_kernel._adapters.registry as _reg

    _reg.register(
        "fused_moe",
        "dsv4-fp4-dequant-blockshape",
        target="text:sglang/srt/layers/quantization/fp8.py",
        mechanism=_reg.MECH_TEXT,
        detail=_NEEDLE.splitlines()[-1],
    )
    LOG.warning(
        "HCU adapter: DSV4 fp4->fp8 dequant block shape aligned to [1,32]"
    )
