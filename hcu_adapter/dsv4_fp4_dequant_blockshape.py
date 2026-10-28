"""DSV4 fp4->fp8 expert dequant: align the advertised block shape.

``SGLANG_DSV4_FP4_DEQUANT=1`` routes DSV4 mxfp4 routed experts through
``cast_e2m1fn_to_e4m3fn`` (fp8.py process_weights_after_loading, plain-else
arm). The cast REQUANTIZES to [128,128] blocks -- it returns
[N/128, K/128] e8m0 scales (the caller converts them to float32 and clears
format_ue8m0) -- but ``self.weight_block_size`` still advertises the
checkpoint's ``[32, 32]``, so the triton/aiter W8A8 block path asserts
``cdiv(N, 32) == N/128`` in invoke_fused_moe_kernel (gfx938 nmz26 boots
2026-10-21 20:05/20:18: V4.1-Flash dp8ep8+dspark, a2a=none).

The fix is nominal: ``[128, 128]`` describes the produced grid exactly --
the same value the aiter-native dequant branch sets in the same file.
Numerics untouched.

v1 of this patch wrongly advertised [1,32] (misread the cast as keeping
the MXFP4 row-fine grid); this version re-needles trees that carry v1.
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
    '                # HCU: cast_e2m1fn_to_e4m3fn REQUANTIZES to [128,128] blocks\n'
    '                # (returns [N/128, K/128] e8m0 scales; the caller floats them)\n'
    '                # -- but weight_block_size still says the checkpoint [32,32],\n'
    '                # so the W8A8 block path asserts cdiv(N,32)!=N/128 on scale\n'
    '                # rows. [128,128] matches the produced grid (same as the\n'
    '                # aiter-native dequant branch sets).\n'
    '                self.weight_block_size = [128, 128]'
)

# v1 of this patch advertised [1,32] on a misread of the cast; re-needle.
_NEEDLE_V1 = (
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
    already = _REPL in text
    if not already:
        if _NEEDLE_V1 in text:
            text = text.replace(_NEEDLE_V1, _REPL)
        elif _NEEDLE in text:
            text = text.replace(_NEEDLE, _REPL)
        else:
            LOG.warning(
                "HCU adapter: fp4-dequant block-shape needle changed upstream "
                "(fp8.py); SGLANG_DSV4_FP4_DEQUANT MoE may assert on scale "
                "rows -- gap list"
            )
            return
        from hcu_adapter.atomic_io import atomic_write

        atomic_write(p, text)
    import hcu_adapter.registry as _reg

    _reg.register(
        # op aligns with the sibling text patch on the same fp8.py target
        # (dsv4_ep_bf16, platform_detect): the registry forbids a second op
        # binding one target ("extend the existing entry").
        "platform_detect",
        "dsv4-fp4-dequant-blockshape",
        target="text:sglang/srt/layers/quantization/fp8.py",
        mechanism=_reg.MECH_TEXT,
        detail="weight_block_size=[128,128] after FP4->FP8 expert dequant",
    )
    LOG.warning(
        "HCU adapter: DSV4 fp4->fp8 dequant block shape aligned to [128,128]"
    )
