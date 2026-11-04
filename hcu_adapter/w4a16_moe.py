# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""W4A16 MoE bridge: DTK deepgemm masked w4a16 grouped GEMM under the
stock ``MoeWNA16`` (moe_wna16) runner (R-05 implementation segment).

Why
---
The production W4A16 line (AWQ/GPTQ-int4 MoE) currently rides the stock
triton ``MoeWNA16`` arm (``--quantization moe_wna16``, MoeRunnerBackend
TRITON). The DTK deepgemm wheel ships a numerically-healthy C++

    grouped_gemm_w4a16_nt_masked_entry(a, b_marlin, s, c, actual, M)

(R-05 recon: rel 0.0022 vs an independent fp64 golden, layout fully
documented), plus its own repacker ``w4a16_marlin_weight``. This bridge
uses it as the experts-GEMM compute tier behind a hand-rolled dispatch /
silu / combine pipeline (eager v1 -- per-expert row counts are
content-dependent, so capture paths stay on the stock triton kernel).

Layout alignment (the reason this bridge is cheap)
--------------------------------------------------
moe_wna16's stored qweight is ``[E, N, K//2]`` u8 with byte = lo nibble
(even k) | hi nibble (odd k) -- exactly deepgemm's ``b`` layout, and the
symmetric (no-zeros) convention ``(q - 8) * s`` matches deepgemm's fixed
``(q - 8) * s``. So engagement needs no weight-format conversion, only
the wheel's marlin repack (cached once per weight tensor) and, when the
checkpoint group is 64/128, a one-time scale expansion to the fixed
group-32 grid deepgemm expects.

Guards (anything unmet -> stock triton, log-once per shape)
-----------------------------------------------------------
* ``HCU_W4A16_MOE_IMPL=deepgemm`` explicitly set (auto keeps triton
  until a serving A/B lands -- candidate tier);
* the call is the int4-w4a16 face with NO zeros (``w1_zp is None``) and
  plain SiLU gating, no gemm1 clamp knobs, not apply-on-input, not
  inplace, not CUDA-graph capturing (v1 eager only -- decode graphs
  keep the stock kernel);
* bf16 activations, ``g in {32, 64, 128}``, deepgemm alignment
  (``K % 128 == 0``, ``2I % 16 == 0`` for gemm1, ``I % 128 == 0`` for
  gemm2 -- I is the intermediate per-partition size).

Knob: ``HCU_W4A16_MOE_IMPL = auto | triton | deepgemm`` (auto default).
"""

from __future__ import annotations

import logging
import os

import torch

LOG = logging.getLogger("hcu.adapter")

_MODE_TRITON = ("triton", "torch", "off", "0")
_MODE_DG = ("deepgemm", "dg", "1")

_CACHE_ATTR = "_hcu_w4a16_marlin_cache"
_GROUP_OK = (32, 64, 128)
_miss_log_once: set = set()
_engaged_log: set = set()
_deepep_installed = False


def _mode() -> str:
    from . import hcu_env

    v, _src = hcu_env.resolve("HCU_W4A16_MOE_IMPL")
    return (v or "auto").strip().lower()


def _dbg(reason: str):
    if os.environ.get("HCU_W4A16_MOE_DEBUG", ""):
        LOG.warning("HCU w4a16 moe: %s", reason)


def deepgemm_w4a16_deepep_ll(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int | None = None,
) -> torch.Tensor:
    """Run the W4A16 expert-major path after DeepEP low-latency dispatch.

    DeepEP low-latency dispatch already places tokens in
    ``[expert, capacity, hidden]`` order and returns one valid-row count per
    local expert.  The regular ``fused_experts`` bridge cannot be reused here:
    it performs its own token sort/dispatch and would destroy the dispatcher
    layout.  This function therefore keeps the expert-major layout through
    both masked grouped GEMMs and returns the same layout for DeepEP combine.
    """
    if hidden_states.ndim != 3:
        raise ValueError(
            "DeepEP W4A16 expects expert-major hidden states [E, M, K], "
            f"got shape={tuple(hidden_states.shape)}"
        )
    if masked_m.ndim != 1 or masked_m.numel() != hidden_states.shape[0]:
        raise ValueError(
            "DeepEP W4A16 masked_m must contain one count per local expert"
        )
    if w1.shape[0] != hidden_states.shape[0] or w2.shape[0] != hidden_states.shape[0]:
        raise ValueError("DeepEP W4A16 expert count does not match weights")
    capacity = hidden_states.shape[1]
    counts = masked_m.to(device=hidden_states.device, dtype=torch.int32)
    if bool(torch.any(counts < 0)) or bool(torch.any(counts > capacity)):
        raise ValueError(
            f"DeepEP W4A16 masked_m must be within [0, {capacity}]"
        )

    b1, s1 = _prepare(w1, w1_scale)
    b2, s2 = _prepare(w2, w2_scale)
    c1 = _grouped_gemm(
        hidden_states.contiguous(),
        b1,
        s1,
        counts,
        capacity,
        w1.shape[1],
        hidden_states.dtype,
    )
    intermediate = torch.nn.functional.silu(c1[..., : w1.shape[1] // 2])
    intermediate = intermediate * c1[..., w1.shape[1] // 2 :]
    c2 = _grouped_gemm(
        intermediate.contiguous(),
        b2,
        s2,
        counts,
        capacity,
        w2.shape[1],
        hidden_states.dtype,
    )
    return c2


def _is_deepep_w4a16_layer(layer) -> bool:
    """Identify the CT-W4A8 -> W4A16 layer without depending on class names."""
    return bool(
        getattr(layer, "w13_weight", None) is not None
        and getattr(layer.w13_weight, "dtype", None) == torch.uint8
        and getattr(layer, "w2_weight", None) is not None
        and getattr(layer, "w13_weight_scale", None) is not None
        and getattr(layer, "w2_weight_scale", None) is not None
        and not bool(getattr(layer, "use_w4afp8", False))
    )


def install_deepep_w4a16():
    """Add the adapter W4A16 branch to the upstream DeepEP LL seam.

    The patch is intentionally narrow: it only handles dispatch objects with
    ``masked_m``/``expected_m`` and only layers whose loaded weights are the
    uint8 W4A16 face.  BF16, FP8/W4AFP8, normal dispatch, and all other
    upstream routes retain their original implementation.
    """
    global _deepep_installed
    if _deepep_installed:
        return
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE

    original = DeepEPMoE.run_moe_core
    if getattr(original, "_hcu_w4a16_deepep", False):
        _deepep_installed = True
        return

    def run_moe_core(self, dispatch_output):
        if (
            hasattr(dispatch_output, "masked_m")
            and hasattr(dispatch_output, "expected_m")
            and _is_deepep_w4a16_layer(self)
        ):
            try:
                output = deepgemm_w4a16_deepep_ll(
                    dispatch_output.hidden_states,
                    self.w13_weight,
                    self.w2_weight,
                    self.w13_weight_scale,
                    self.w2_weight_scale,
                    dispatch_output.masked_m,
                    dispatch_output.expected_m,
                )
                from sglang.srt.layers.moe.token_dispatcher.deepep import (
                    DeepEPLLCombineInput,
                )

                return DeepEPLLCombineInput(
                    hidden_states=output,
                    topk_ids=dispatch_output.topk_ids,
                    topk_weights=dispatch_output.topk_weights,
                )
            except Exception:
                LOG.warning(
                    "HCU w4a16 moe: DeepEP LL path failed; preserving "
                    "the upstream error contract",
                    exc_info=True,
                )
                raise
        return original(self, dispatch_output)

    run_moe_core._hcu_w4a16_deepep = True
    DeepEPMoE.run_moe_core = run_moe_core
    _deepep_installed = True
    try:
        from hcu_adapter import registry

        registry.register(
            "w4a16_moe",
            "hcu_adapter.w4a16_moe.deepgemm_w4a16_deepep_ll",
            "sglang.srt.layers.moe.ep_moe.DeepEPMoE.run_moe_core",
            registry.MECH_REBIND,
            detail="CT W4A8->W4A16 DeepEP low-latency expert-major branch",
        )
    except Exception:
        # The route remains functional in minimal/fake test interpreters where
        # the full runtime catalog is intentionally absent.
        LOG.debug("HCU w4a16 moe: registry entry unavailable", exc_info=True)
    LOG.info(
        "HCU w4a16 moe: DeepEP low-latency expert-major branch installed"
    )


def _prepare(experts_qweight: torch.Tensor, scales: torch.Tensor):
    """Repack one expert stack to the deepgemm/marlin consume form.

    Returns (b_marlin repacked for the wheel, s32 [E, N, K//32] bf16 on
    the fixed group-32 grid). Cached on the qweight tensor -- weights
    are frozen post load and the expansion is a pure function of them.
    """
    cached = getattr(experts_qweight, _CACHE_ATTR, None)
    if cached is not None:
        return cached
    from deepgemm import w4a16_marlin_weight

    if experts_qweight.dtype != torch.uint8:
        raise TypeError(f"qweight dtype {experts_qweight.dtype}")
    b = experts_qweight  # already [E, N, K//2] u8 lo-even (moe_wna16 store)
    g = 2 * b.shape[-1] // scales.shape[-1]  # checkpoint group size
    if g not in _GROUP_OK:
        raise ValueError(f"group {g} not expandable to 32")
    if g != 32:
        s = scales.repeat_interleave(g // 32, dim=-1)
    else:
        s = scales.contiguous()
    out = (w4a16_marlin_weight(b.contiguous()),
           s.to(torch.bfloat16).contiguous())
    try:
        setattr(experts_qweight, _CACHE_ATTR, out)
    except Exception:
        pass
    return out


def _grouped_gemm(a_pad: torch.Tensor, bm, s32, counts: torch.Tensor,
                  cap: int, out_feats: int, out_dtype: torch.dtype):
    """Masked grouped w4a16 GEMM. NOTE: ``bm`` is the wheel's marlin
    repack (6-D hardware view, not [E,N,K//2]) so the output width must
    be supplied -- it is always the qweight's logical N."""
    import deepgemm

    c = torch.zeros(a_pad.shape[0], cap, out_feats, device=a_pad.device,
                    dtype=out_dtype)
    deepgemm.grouped_gemm_w4a16_nt_masked_entry(
        a_pad.contiguous(), bm, s32, c, counts.to(torch.int32), int(cap)
    )
    return c


def deepgemm_w4a16_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    routed_scaling=None,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor | None:
    """Masked-w4a16 expert pipeline over the deepgemm C++ grouped GEMMs.

    Inputs mirror the stock triton ``fused_experts`` int4 face: standard
    moe_wna16 qweights ``[E, 2I, K//2]`` / ``[E, K, I//2]`` u8 (lo-even),
    scales ``[E, 2I, K//g]`` / ``[E, K, I//g]`` in the activation dtype,
    row-major ``topk_ids [M, topk]``. May return None if the routing is
    pathologically skewed; the wrapper falls back to the stock kernel.
    """
    M, K = hidden_states.shape
    E, N1, _ = w1.shape          # N1 = 2I
    N2 = w2.shape[1]             # N2 = K
    I = N1 // 2
    dev = hidden_states.device
    act_dtype = hidden_states.dtype

    if apply_router_weight_on_input:
        _dbg("apply_router_weight_on_input unsupported")
        return None

    topk = topk_ids.shape[-1]
    flat_ids = topk_ids.reshape(-1).to(torch.int64)
    T = flat_ids.numel()
    sort_e, perm = torch.sort(flat_ids, stable=True)
    counts = torch.bincount(sort_e, minlength=E).to(torch.int32)
    # capacity = max tokens per expert, quantized to a 128 tile
    cap = int(counts.max()) if T else 0
    cap = max((cap + 127) // 128 * 128, 128)
    if cap > 4096 and cap * 4 > T:
        # pathological skew: padding would blow past the stock kernel's
        # own working set -- let it take this call
        return None

    bm1, s1 = _prepare(w1, w1_scale)
    bm2, s2 = _prepare(w2, w2_scale)

    # padded gather: sorted flat row r belongs to expert sort_e[r] at
    # slot (r - b0[sort_e[r]]). A sequential view-fill would only line
    # up under uniform per-expert counts -- real routing skews.
    b0 = torch.zeros(E + 1, dtype=torch.int64, device=dev)
    b0[1:] = torch.cumsum(counts.to(torch.int64), 0)
    dest = sort_e * cap + (torch.arange(T, device=dev) - b0[sort_e])
    a_flat = torch.zeros(E * cap, K, device=dev, dtype=act_dtype)
    a_flat[dest] = hidden_states.index_select(0, perm // topk)
    a_pad = a_flat.view(E, cap, K)

    c1 = _grouped_gemm(a_pad, bm1, s1, counts, cap, N1, act_dtype)
    # gate = first half, up = second half (sglang w13 packing); silu in
    # the activation dtype -- tiles over the padded rows harmlessly
    h_pad = torch.nn.functional.silu(c1[..., :I]) * c1[..., I:]
    c2 = _grouped_gemm(h_pad, bm2, s2, counts, cap, N2,
                       act_dtype)   # N2 = K

    # fold back: c2[valid] gathers expert-major valid prefixes, which is
    # exactly the sort_e/perm order (stable sort, ascending flat slots)
    valid = (torch.arange(cap, device=dev)[None, :] < counts[:, None])
    flat_valid = c2[valid]                                   # [T, K]
    flat_w = topk_weights.reshape(-1).to(act_dtype)[perm]
    contrib = flat_valid.to(act_dtype) * flat_w[:, None]
    out = torch.zeros_like(hidden_states)
    out.index_add_(0, perm // topk, contrib)
    rs = routed_scaling
    if rs is not None and float(rs) != 1.0:
        out = out * float(rs)
    if len(_engaged_log) < 3:
        _engaged_log.add((M, E, I, K))
        LOG.warning(
            "HCU w4a16 moe: deepgemm ENGAGED M=%d E=%d I=%d K=%d "
            "topk=%d cap=%d", M, E, I, K, topk, cap)
    return out


def install():
    """Wrap the stock triton runner's ``fused_experts`` for the int4 face."""
    mode = _mode()
    if mode in _MODE_TRITON:
        LOG.info("HCU w4a16 moe bridge: off (HCU_W4A16_MOE_IMPL=%s)", mode)
        return
    if mode not in _MODE_DG:
        LOG.info("HCU w4a16 moe bridge: unknown mode %s -> skip", mode)
        return

    import importlib

    mod_name = "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
    try:
        fm = importlib.import_module(mod_name)
    except ImportError:
        LOG.info("HCU w4a16 moe bridge: %s not importable -> skip", mod_name)
        return

    current = getattr(fm, "fused_experts", None)
    if current is None or getattr(current, "_hcu_w4a16_moe", False):
        return

    def fused_experts_w4a16(
        hidden_states, w1, w2, topk_output, moe_runner_config, **kwargs
    ):
        try:
            if (
                kwargs.get("use_int4_w4a16")
                and not torch.cuda.is_current_stream_capturing()
                and not bool(getattr(moe_runner_config, "inplace", False))
                and kwargs.get("w1_zp") is None
                and kwargs.get("w2_zp") is None
            ):
                out = deepgemm_w4a16_fused_experts(
                    hidden_states,
                    w1,
                    w2,
                    topk_output[0],
                    topk_output[1],
                    kwargs.get("w1_scale"),
                    kwargs.get("w2_scale"),
                    routed_scaling=getattr(
                        moe_runner_config, "routed_scaling_factor", None),
                    apply_router_weight_on_input=bool(
                        getattr(moe_runner_config,
                                "apply_router_weight_on_input", False)),
                )
                if out is not None:
                    return out
        except Exception:
            # never take the server down for a vendor-kernel experiment
            LOG.warning("HCU w4a16 moe: call failed -> stock triton",
                        exc_info=True)
        return current(hidden_states, w1, w2, topk_output, moe_runner_config,
                       **kwargs)

    fused_experts_w4a16._hcu_w4a16_moe = True
    fm.fused_experts = fused_experts_w4a16
    # triton_utils re-exports fused_experts at package level; rebind that
    # copy so from-imports land in the wrapper, same as the aiter arm
    try:
        pkg = importlib.import_module(
            "sglang.srt.layers.moe.moe_runner.triton_utils")
        if getattr(pkg, "fused_experts", None) is current:
            pkg.fused_experts = fused_experts_w4a16
    except ImportError:
        pass
    LOG.info(
        "HCU w4a16 moe bridge: installed (HCU_W4A16_MOE_IMPL=%s, "
        "explicit deepgemm tier over the stock moe_wna16 runner)", mode)
