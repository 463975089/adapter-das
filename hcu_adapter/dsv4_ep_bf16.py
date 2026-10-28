"""DSV4-Flash DeepEP bf16 experts: fp4 -> bf16 load-time dequant.

Why: on the DTK stack the DeepEP expert compute can only run through the
deepgemm fork's grouped GEMMs (deepep_ll/deepep_normal formats have no
triton runner). The fork's fp8 masked/contiguous kernels take per-channel
2-D scales only -- the R-04 conviction (2026-10-18) -- while DSV4-Flash
experts are 128x128 block-scaled fp8, so the fp8 arms are numerically
unusable. The fork's *bf16* arms are validated (bridge pack-cache masked
wrapper + triton contiguous reimpl), and sglang's deep_gemm moe runner
branches on ``quant_info.w13_weight.dtype``: bf16 weights take
``_run_masked_bf16_gemm`` / ``_run_bf16_contiguous_gemm`` natively.

The needle patch hooks the TOP of
``Fp8MoEMethod.process_weights_after_loading_block_quant``: under
``SGLANG_HCU_EP_BF16=1`` it runs the stock fp4->block-fp8 dequant
(``_dequantize_aiter_fp4_experts``) and then converts block-fp8 -> bf16,
returning before every fp8/aiter-processing arm. The stock dequant lives
inside ``if _use_aiter and self.is_fp4_expert:`` -- under the platform
seam ``_is_hip`` is False, so that arm (and any hook inside it) never
runs in the server process; hence the method-top placement. Launch with

    SGLANG_HCU_EP_BF16=1 --deepep-dispatcher-output-dtype bf16

TP-mode servers are unaffected: they never set the env.
"""

from __future__ import annotations

import logging
import pathlib

from . import registry as _reg

logger = logging.getLogger("hcu.adapter")

_PKG = "sglang/srt/layers/quantization/fp8.py"
_DETAIL = (
    "SGLANG_HCU_EP_BF16=1: fp4 routed experts dequantized to "
    "bf16 at load (DTK deepgemm fork fp8 grouped GEMMs are "
    "per-channel-scale-only per R-04); pair with "
    "--deepep-dispatcher-output-dtype bf16."
)

_DEF_LINE = (
    "    def process_weights_after_loading_block_quant("
    "self, layer: Module) -> None:\n"
)
# fp8.py defines process_weights_after_loading_block_quant TWICE (an
# earlier quant class at ~732 and Fp8MoEMethod at ~1767); _DEF_LINE alone
# would hit the wrong class, so every guard insertion is anchored on the
# Fp8MoEMethod-specific context (the v1 helper's closing or the pristine
# warning_once block).

_GUARD = (
    "        # HCU dsv4_ep_bf16: fp4 -> fp8 -> bf16 in one step, ahead of every\n"
    "        # fp8/aiter arm (the stock dequant arm is gated on _use_aiter, which\n"
    "        # the platform seam keeps False in the server process).\n"
    "        if (\n"
    "            __import__(\"os\").environ.get(\"SGLANG_HCU_EP_BF16\", \"\") == \"1\"\n"
    "            and self.is_fp4_expert\n"
    "            and layer.w13_weight.dtype != torch.bfloat16\n"
    "        ):\n"
    "            self._dequantize_aiter_fp4_experts(layer)\n"
    "            self._hcu_ep_bf16_dequant(layer)\n"
    "            self.weight_block_size = [128, 128]\n"
    "            return\n"
)

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
""" + _GUARD

# v1 call-site (inside the _use_aiter arm) -- superseded by the guard; the
# guard returns first, so removal is cosmetic but keeps the tree clean.
_V1_CALL = """                if __import__("os").environ.get("SGLANG_HCU_EP_BF16", "") == "1":
                    self._hcu_ep_bf16_dequant(layer)
                return
"""
# v1 helper-method tail + def line: the Fp8MoEMethod-scoped anchor for
# adding the guard to a v1-patched tree.
_V1_DEF_CONTEXT = (
    "        logger.warning_once(\n"
    "            \"HCU dsv4_ep_bf16: DeepEP expert weights dequantized to bf16 \"\n"
    "            \"(fork fp8 grouped GEMMs are block-scale incompatible).\"\n"
    "        )\n"
    "\n"
    "    def process_weights_after_loading_block_quant(self, layer: Module) -> None:\n"
)


def _atomic_write(path: pathlib.Path, text: str) -> None:
    # a needle patch that breaks the target's syntax takes the whole
    # import chain down (platform seam included) -- compile-gate the new
    # text so a bad patch raises here instead of at server boot.
    compile(text, str(path), "exec")
    tmp = path.with_suffix(path.suffix + ".hcu_tmp")
    tmp.write_text(text)
    tmp.replace(path)


def install() -> None:
    import sglang.kernels.jit as _jit_root

    pkg_root = pathlib.Path(_jit_root.__file__).parents[2]
    p = pkg_root / "srt" / "layers" / "quantization" / "fp8.py"
    text = p.read_text()

    if "HCU dsv4_ep_bf16: fp4 -> fp8 -> bf16 in one step" not in text:
        if _V1_CALL in text:
            text = text.replace(
                _V1_CALL, "                return\n", 1
            )
        if _METHOD_NEEDLE in text:
            # pristine tree: helper method + guard in one shot
            text = text.replace(_METHOD_NEEDLE, _METHOD_REPL, 1)
        elif _V1_DEF_CONTEXT in text:
            # v1-patched tree: helper already present, add the guard only
            text = text.replace(
                _V1_DEF_CONTEXT, _V1_DEF_CONTEXT + _GUARD, 1
            )
        else:
            raise RuntimeError(
                "HCU adapter: dsv4_ep_bf16 needle drifted in "
                f"{_PKG}; re-anchor before upgrading the sglang tree."
            )
        _atomic_write(p, text)

    _reg.register(
        "platform_detect",
        "dsv4-ep-bf16-experts",
        target=f"text:{_PKG}",
        mechanism=_reg.MECH_TEXT,
        detail=_DETAIL,
    )
    logger.warning(
        "HCU adapter: dsv4_ep_bf16 installed (DeepEP bf16 experts, "
        "env SGLANG_HCU_EP_BF16 gates the dequant)"
    )
