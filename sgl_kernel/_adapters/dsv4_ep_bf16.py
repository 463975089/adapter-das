"""DSV4-Flash DeepEP bf16 experts: fp8-block -> bf16 load-time dequant.

Why: on the DTK stack the DeepEP expert compute can only run through the
deepgemm fork's grouped GEMMs (deepep_ll/deepep_normal formats have no
triton runner). The fork's fp8 masked/contiguous kernels take per-channel
2-D scales only -- the R-04 conviction (2026-10-18) -- while DSV4-Flash
experts are 128x128 block-scaled fp8, so the fp8 arms are numerically
unusable. The fork's *bf16* arms are validated (bridge pack-cache masked
wrapper + triton contiguous reimpl), and sglang's deep_gemm moe runner
branches on ``quant_info.w13_weight.dtype``: bf16 weights take
``_run_masked_bf16_gemm`` / ``_run_bf16_contiguous_gemm`` natively.

This needle patch extends the existing SGLANG_DSV4_FP4_DEQUANT arm (fp4 ->
block-fp8) with one env-gated extra step: dequantize the block-fp8 expert
weights to plain bf16 Parameters and drop the scale params. Launch with

    SGLANG_HCU_EP_BF16=1 --deepep-dispatcher-output-dtype bf16

(the server arg keeps the DeepEP dispatch itself in bf16/quant_type=0;
weights bf16 makes the runner take the bf16 GEMM arms). TP-mode servers
are unaffected: they never set the env.
"""

from __future__ import annotations

import logging
import pathlib

from . import registry as _reg

logger = logging.getLogger("hcu.adapter")

_PKG = "sglang/srt/layers/quantization/fp8.py"

_METHOD_NEEDLE = """        self.is_fp4_expert = False
        logger.warning_once("Dequantized FP4 MoE expert weights to FP8.")

    def process_weights_after_loading_block_quant(self, layer: Module) -> None:
"""

_METHOD_REPL = """        self.is_fp4_expert = False
        logger.warning_once("Dequantized FP4 MoE expert weights to FP8.")

    def _hcu_ep_bf16_dequant(self, layer: Module) -> None:
        \"\"\"HCU dsv4_ep_bf16: block-fp8 experts -> bf16 for the DeepEP
        bf16 dispatch/compute path (fork fp8 arms are scale-recipe
        incompatible; see module docstring). Idempotent via dtype check.
        \"\"\"
        if layer.w13_weight.dtype == torch.bfloat16:
            return
        import os

        def _dq(w, s):
            # w: [E, N, K] fp8 blockwise; s: [E, N/128, K/128] float scales
            E, N, K = w.shape
            w5 = w.to(torch.float32).view(E, N // 128, 128, K // 128, 128)
            s5 = s.view(E, N // 128, 1, K // 128, 1)
            return (w5 * s5).reshape(E, N, K).to(torch.bfloat16)

        layer.w13_weight = torch.nn.Parameter(
            _dq(layer.w13_weight.data, layer.w13_weight_scale_inv.data.float()),
            requires_grad=False,
        )
        layer.w2_weight = torch.nn.Parameter(
            _dq(layer.w2_weight.data, layer.w2_weight_scale_inv.data.float()),
            requires_grad=False,
        )
        layer.w13_weight_scale_inv = None
        layer.w2_weight_scale_inv = None
        logger.warning_once(
            "HCU dsv4_ep_bf16: DeepEP expert weights dequantized to bf16 "
            "(fork fp8 grouped GEMMs are block-scale incompatible)."
        )

    def process_weights_after_loading_block_quant(self, layer: Module) -> None:
"""

_CALL_NEEDLE = """                if _use_aiter and runner_is_aiter:
                    layer.w13_weight.data = shuffle_weight(
                        layer.w13_weight.contiguous(), (16, 16)
                    )
                    layer.w2_weight.data = shuffle_weight(
                        layer.w2_weight.contiguous(), (16, 16)
                    )
                return

        if self.convert_mxfp8_to_block:
"""

_CALL_REPL = """                if _use_aiter and runner_is_aiter:
                    layer.w13_weight.data = shuffle_weight(
                        layer.w13_weight.contiguous(), (16, 16)
                    )
                    layer.w2_weight.data = shuffle_weight(
                        layer.w2_weight.contiguous(), (16, 16)
                    )
                if __import__("os").environ.get("SGLANG_HCU_EP_BF16", "") == "1":
                    self._hcu_ep_bf16_dequant(layer)
                return

        if self.convert_mxfp8_to_block:
"""


def _atomic_write(path: pathlib.Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".hcu_tmp")
    tmp.write_text(text)
    tmp.replace(path)


def install() -> None:
    import sglang.kernels.jit as _jit_root

    pkg_root = pathlib.Path(_jit_root.__file__).parents[2]
    p = pkg_root / "srt" / "layers" / "quantization" / "fp8.py"
    text = p.read_text()

    if _CALL_REPL in text:
        return  # already patched (idempotent)

    if _METHOD_NEEDLE not in text or _CALL_NEEDLE not in text:
        raise RuntimeError(
            "HCU adapter: dsv4_ep_bf16 needle drifted in "
            f"{_PKG} (method={_METHOD_NEEDLE in text}, call={_CALL_NEEDLE in text}); "
            "re-anchor before upgrading the sglang tree."
        )

    text = text.replace(_METHOD_NEEDLE, _METHOD_REPL, 1)
    text = text.replace(_CALL_NEEDLE, _CALL_REPL, 1)
    _atomic_write(p, text)

    _reg.register(
        "platform_detect",
        "dsv4-ep-bf16-experts",
        target=f"text:{_PKG}",
        mechanism=_reg.MECH_TEXT,
        detail=(
            "SGLANG_HCU_EP_BF16=1: block-fp8 routed experts dequantized to "
            "bf16 at load (DTK deepgemm fork fp8 grouped GEMMs are "
            "per-channel-scale-only per R-04); pair with "
            "--deepep-dispatcher-output-dtype bf16."
        ),
    )
    logger.warning(
        "HCU adapter: dsv4_ep_bf16 installed (DeepEP bf16 experts, "
        "env SGLANG_HCU_EP_BF16 gates the dequant)"
    )
