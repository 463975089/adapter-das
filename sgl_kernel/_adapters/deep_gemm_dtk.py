"""Alias the DTK ``deepgemm`` wheel as upstream ``deep_gemm`` (EP bridge).

The DTK stack ships Hygon's deepgemm fork (2.1.0+dtk2604) under the
one-word package name ``deepgemm`` with upstream-compatible entry points
(``m_grouped_bf16_gemm_nt_masked`` / ``m_grouped_fp8_gemm_nt_masked[_ll]``
/ ``..._contiguous`` etc.). sglang only ever imports ``deep_gemm``, so on
this stack the DeepGEMM-backed MoE runners (the only deepep_ll-capable
runners besides aiter/npu ones) are unreachable.

``install()`` (called from sitecustomize under ``HCU_DEEPGEMM_BRIDGE=1``)
builds a ``deep_gemm`` module in ``sys.modules`` backed by the DTK package
plus the ``deep_gemm.utils.layout`` helper module the wrapper imports at
module load. The utils.layout scale transforms have no verified DTK
equivalent — they raise loudly instead of silently computing wrong
numbers, which keeps fp8 DeepEP off the table until validated (bf16
grouped GEMM needs no scales).
"""

from __future__ import annotations

import sys
import types

import torch


def _make_layout_module() -> types.ModuleType:
    mod = types.ModuleType("deep_gemm.utils.layout")

    def _unimplemented(name):
        def _raise(*args, **kwargs):
            raise NotImplementedError(
                f"deep_gemm.utils.layout.{name}: no verified DTK deepgemm "
                "equivalent (fp8 DeepEP scale layouts unvalidated on HCU); "
                "bf16 grouped GEMM paths must not reach here"
            )

        return _raise

    mod.get_mn_major_tma_aligned_tensor = _unimplemented(
        "get_mn_major_tma_aligned_tensor"
    )
    mod.get_col_major_tma_aligned_tensor = _unimplemented(
        "get_col_major_tma_aligned_tensor"
    )
    mod.transform_sf_into_required_layout = _unimplemented(
        "transform_sf_into_required_layout"
    )
    return mod


def fp8_gemm_nt_dequant(lhs, rhs, out):
    """Dense fp8 NT GEMM with 2-D block scales via explicit dequant.

    sglang hands (q, scales) pairs: lhs = (A [M, K] fp8, As), rhs = (B [N, K]
    fp8, Bs). As is per-token-group-128 [M, K/128] — row-major or the
    column-major TMA-aligned VIEW (stride (1, aligned_M), sliced on dim 0;
    ``reshape`` restores the logical [M, K/128] either way). Bs is per-block
    [N/128, K/128] (128x128 recipe) or per-row [N, K/128] (1x128 recipe);
    both reduce to the same broadcast after expanding the N dim to blocks.

    out is written in place and returned (entrypoint contract).
    """
    a, a_s = lhs[0], lhs[1]
    b, b_s = rhs[0], rhs[1]
    m, k = a.shape
    a_s_groups = a_s.reshape(m, -1).shape[-1] if a_s.numel() and m else 0
    if a_s_groups == 0 or k % a_s_groups != 0:
        # Degenerate warmup/compile probe (empty scales, m=0, non-divisible
        # k): no valid block semantics exist — produce zeros rather than
        # SIGFPE-ing in the view math (rank dies mid-warmup otherwise).
        out.zero_()
        return out
    block_k = k // a_s_groups
    n_block = b_s.shape[0] if b_s.numel() else 0
    if n_block == 0 or b.shape[0] % n_block != 0:
        out.zero_()
        return out
    block_n = b.shape[0] // n_block
    a_dq = (
        a.to(torch.float32).view(m, -1, block_k)
        * a_s.reshape(m, -1).unsqueeze(-1)
    ).view(m, k)
    # b rows are physically (n_block, block_n, k); scale per (n_block, k/128)
    # broadcasts over the inner-row dim: [n_block, 1, k_groups, 1].
    b_dq = (
        b.to(torch.float32).view(n_block, block_n, -1, block_k)
        * b_s.reshape(n_block, -1)[:, None, :, None]
    ).view(b.shape[0], k)
    torch.matmul(a_dq.to(out.dtype), b_dq.to(out.dtype).t(), out=out)
    return out


def install():
    """Bind ``deep_gemm`` (+ ``deep_gemm.utils.layout``) to the DTK wheel.

    No-op when a real ``deep_gemm`` is importable or an alias already
    exists, so a future upstream-wheel install wins automatically.
    """
    if "deep_gemm" in sys.modules:
        return sys.modules["deep_gemm"]
    try:
        import importlib.util

        if importlib.util.find_spec("deep_gemm") is not None:
            return None  # real deep_gemm present; do not shadow it
    except Exception:
        pass
    import deepgemm

    shim = types.ModuleType("deep_gemm")
    shim.__dict__.update(
        {k: v for k, v in vars(deepgemm).items() if not k.startswith("__")}
    )
    shim.__package__ = "deep_gemm"
    shim.__file__ = getattr(deepgemm, "__file__", None)
    shim.dtk_deepgemm = deepgemm  # escape hatch to the raw fork

    _dtk_masked = getattr(deepgemm, "m_grouped_bf16_gemm_nt_masked", None)
    _dtk_pack_bf16 = getattr(deepgemm, "pack_bf16_masked_weight", None)
    if _dtk_masked is not None:

        _bf16_packed = {}  # (data_ptr, shape, dtype) -> packed weight

        def _bf16_masked(a, b, d, masked_m=None, expected_m=None,
                         expected_m_per_group=None, config=None, **unknown):
            # sglang names the 5th arg `expected_m` (entrypoint positional,
            # compile_utils warmup passes it as a kwarg); the DTK fork calls
            # it `expected_m_per_group` and puts `config` last. The default
            # masked mode (2000, BM64/BN256/BK128 HIPC) reads the weight in a
            # packed layout — raw (G,N,K) weights give garbage numbers, so
            # weights are packed once and cached by storage identity.
            if unknown:
                raise TypeError(
                    f"m_grouped_bf16_gemm_nt_masked: unexpected kwargs "
                    f"{sorted(unknown)}"
                )
            if expected_m is None:
                expected_m = expected_m_per_group
            if masked_m is None or expected_m is None:
                raise TypeError(
                    "m_grouped_bf16_gemm_nt_masked: masked_m and "
                    "expected_m are required"
                )
            if a.dtype != torch.bfloat16:
                # dispatch can hand us fp8/fp32 activations depending on the
                # runner dtype config; the DTK masked bf16 kernel is strictly
                # bf16-in — upcast a (b is handled by the pack path below).
                a = a.to(torch.bfloat16)
            if b.dtype != torch.bfloat16:
                b = b.to(torch.bfloat16)
            if _dtk_pack_bf16 is not None and b.dim() == 3:
                key = (b.data_ptr(), tuple(b.shape), b.dtype)
                packed = _bf16_packed.get(key)
                if packed is None:
                    packed = _dtk_pack_bf16(b.contiguous())
                    _bf16_packed[key] = packed
                b = packed
            if config is not None:
                return _dtk_masked(a, b, d, masked_m, expected_m, config)
            return _dtk_masked(a, b, d, masked_m, expected_m)

        shim.m_grouped_bf16_gemm_nt_masked = _bf16_masked

    utils = types.ModuleType("deep_gemm.utils")
    layout = _make_layout_module()
    utils.layout = layout
    shim.utils = utils

    # DSA indexer (v0.5.21) resolves the paged-MQA trio from deep_gemm on the
    # is_cuda semantic branch (the aiter arm is is_hip-gated and HCU never
    # takes it). The DTK wheel lacks all three; route to the torch
    # implementation (same numerics as the proven MQA emu).
    _missing_paged_mqa = (
        getattr(deepgemm, "fp8_paged_mqa_logits", None) is None
        or getattr(deepgemm, "get_paged_mqa_logits_metadata", None) is None
        or getattr(deepgemm, "get_num_sms", None) is None
    )
    if _missing_paged_mqa:
        from . import dsa_paged_mqa as _dpm

        if getattr(shim, "get_num_sms", None) is None:
            shim.get_num_sms = _dpm.get_num_sms
        if getattr(shim, "get_paged_mqa_logits_metadata", None) is None:
            shim.get_paged_mqa_logits_metadata = _dpm.get_paged_mqa_logits_metadata
        if getattr(shim, "fp8_paged_mqa_logits", None) is None:
            shim.fp8_paged_mqa_logits = _dpm.fp8_paged_mqa_logits

    # sm-count scheduler hooks (configure_deep_gemm_num_sms context) and the
    # PDL toggle: no DTK equivalents — setters are stateful no-ops.
    if getattr(shim, "set_num_sms", None) is None:
        def _set_num_sms(n):
            return None

        shim.set_num_sms = _set_num_sms
    if getattr(shim, "set_pdl", None) is None:
        def _set_pdl(enable):
            return None

        shim.set_pdl = _set_pdl

    # v0.5.21 mHC pre GEMM: upstream name tf32_hc_prenorm_gemm, DTK fork spells
    # it pernorm; identical (a, b, d, sqr_sum, num_splits) signature.
    if (
        getattr(shim, "tf32_hc_prenorm_gemm", None) is None
        and getattr(deepgemm, "tf32_hc_pernorm_gemm", None) is not None
    ):
        shim.tf32_hc_prenorm_gemm = deepgemm.tf32_hc_pernorm_gemm

    # v0.5.21 dense fp8 w8a8 block-linear (Glm5Next FP8 checkpoints): sglang
    # entrypoint.gemm_nt_f8f8bf16 calls deep_gemm.fp8_gemm_nt(lhs, rhs, out).
    # The DTK fork spells the dense fp8 GEMM `fp8_gemm(a, b, d, ...)` — but a
    # name-only alias is numerically WRONG: nmz26 GPU probe (probe10/12) shows
    # the fork's kernel only honors per-row scales [rows, 1]; given sglang's
    # per-token-group-128 As [M, K/128] (col-major TMA-aligned) and per-block
    # Bs [N/128, K/128] it either ignores As or reads them OOB (cos=0.004,
    # maxrel=4e5 vs dequant reference — the T1 word-soup root cause). Instead
    # dequantize both operands with their fp32 block scales and run one bf16
    # matmul: the fork's own dense-fp8 arm has no correct 128x128-block mode
    # to forward to, and the bf16 matmul it degenerates to was ~60x faster
    # than the triton w8a8 fallback on the same tensors (0.14ms vs 7.1ms at
    # 512³ cold; bf16 core 0.02ms).
    if (
        getattr(shim, "fp8_gemm_nt", None) is None
        and getattr(deepgemm, "fp8_gemm", None) is not None
    ):

        def _fp8_gemm_nt(lhs, rhs, out, **_unknown):
            if _unknown:
                raise TypeError(
                    f"fp8_gemm_nt dequant wrapper: unexpected kwargs "
                    f"{sorted(_unknown)}"
                )
            return fp8_gemm_nt_dequant(lhs, rhs, out)

        shim.fp8_gemm_nt = _fp8_gemm_nt

    sys.modules["deep_gemm"] = shim
    sys.modules["deep_gemm.utils"] = utils
    sys.modules["deep_gemm.utils.layout"] = layout

    # contiguous-layout grouped GEMM (DeepEP normal / auto prefill): the
    # DTK w16a16 contiguous kernel is broken (2D input VMFault, 3D input
    # wrong numbers) and the fork lacks the mk-alignment get/set pair the
    # wrapper's contiguous_layout_alignment_scope needs. A triton
    # reimplementation (native [E, N, K] weights, alignment-padded row
    # blocks) replaces both unless HCU_TRITON_CONTIG=0 keeps the raw DTK
    # escape hatch.
    try:
        from . import hcu_env as _env

        if _env.enabled("HCU_TRITON_CONTIG"):
            from . import deep_gemm_contig_triton as _ct

            shim.m_grouped_bf16_gemm_nt_contiguous = (
                _ct.m_grouped_bf16_gemm_nt_contiguous
            )
            shim.get_mk_alignment_for_contiguous_layout = (
                _ct.get_mk_alignment_for_contiguous_layout
            )
            shim.set_mk_alignment_for_contiguous_layout = (
                _ct.set_mk_alignment_for_contiguous_layout
            )
    except Exception:
        import logging

        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: triton contiguous grouped GEMM not installed",
            exc_info=True,
        )
    return shim
