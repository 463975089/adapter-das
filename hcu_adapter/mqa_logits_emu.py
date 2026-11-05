# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""torch emulation of the aiter fp8 MQA-logits kernels (HCU gfx936-class).

The DTK aiter wheel's paged/ragged MQA-logits triton kernels hard-gate on
gfx942/gfx950. On gfx936-class HCUs (BW200: no fp8 matrix unit, no MFMA
definitions in triton 3.6) emulate them with torch: dequantize e4m3 via the
dtype catwalk (encoder and decoder always share sglang's is_fp8_fnuz choice,
so the convention matches by construction), then batched bf16 GEMMs with fp32
scales — memory-bound, exact products, fp32 accumulation. Q-side scales are
already folded into ``weights`` by the caller (_apply_q_scale_and_softmax_scale
/ _get_logits_head_gate), so only the per-token K scale multiplies here.

Only the legacy page_size=1 / no-preshufflepaged path is supported
(HCU_DSA_PRESHUFFLE=0), which is the upstream fallback for HIP anyway.

Gated by env ``HCU_MQA_EMU=1`` (usually together with HCU_DSA_PRESHUFFLE=0).
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

_HEAD_DIM_WITH_SF = 132  # 128 fp8 + 4B fp32 scale per index key


_MQA_CALL_LOG = {"n": 0}


def _note_call(tag):
    if os.environ.get("HCU_DUMP_MQA") == "1" and _MQA_CALL_LOG["n"] < 8:
        print(f"[HCU mqa-emu] {tag} called", flush=True)
        _MQA_CALL_LOG["n"] += 1


def _fp8_view_dtype():
    from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

    return torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


def _as_bf16(t):
    """fp8 (or raw u8/i8 storage, the DTK aiter convention) -> bf16, exact."""
    if t.dtype in (torch.uint8, torch.int8):
        t = t.contiguous().view(_fp8_view_dtype())
    return t.to(torch.bfloat16)


def _paged_mqa_logits(
    q_fp8,
    kv_cache_fp8,
    weights,
    seq_lens,
    block_tables,
    max_seq_len,
    *,
    preshuffle,
    kv_block_size,
):
    _note_call("paged")
    if preshuffle or kv_block_size != 1:
        raise NotImplementedError(
            "HCU MQA emu only supports legacy kv_block_size=1 no-preshuffle "
            f"(got preshuffle={preshuffle}, kv_block_size={kv_block_size})"
        )
    assert q_fp8.dim() == 3, q_fp8.shape
    q_b = _as_bf16(q_fp8).contiguous()  # [Q, H, D]
    Q, H, _ = q_b.shape
    dev = q_b.device
    kv = kv_cache_fp8.contiguous().view(-1, _HEAD_DIM_WITH_SF)
    k_b = _as_bf16(kv[:, :128]).contiguous()  # [T, D]
    k_scale = kv[:, 128:].contiguous().view(torch.float32).reshape(-1)  # [T]
    w = weights.reshape(Q, H).to(torch.float32)  # [Q, H], q_scale folded in
    seq = seq_lens.to(torch.int64).reshape(-1).clamp(min=0)
    bt = block_tables.to(torch.int64).contiguous()  # [Q, L] page-1 token ids
    L = min(bt.shape[1], int(max_seq_len))
    logits = torch.zeros((Q, int(max_seq_len)), dtype=torch.float32, device=dev)

    qt = int(os.environ.get("HCU_MQA_EMU_Q_TILE", "128"))
    tt = int(os.environ.get("HCU_MQA_EMU_T_TILE", "2048"))
    ar = torch.arange(tt, device=dev)
    for i0 in range(0, Q, qt):
        i1 = min(Q, i0 + qt)
        qb = q_b[i0:i1]  # [Qq, H, D]
        for t0 in range(0, L, tt):
            t1 = min(L, t0 + tt)
            pt = bt[i0:i1, t0:t1]  # [Qq, Tt]
            kg = k_b[pt]  # [Qq, Tt, D]
            sc = k_scale[pt].unsqueeze(1)  # [Qq, 1, Tt]
            scores = torch.matmul(qb, kg.transpose(1, 2)).float()  # [Qq, H, Tt]
            row = (scores * w[i0:i1].unsqueeze(-1) * sc).sum(dim=1)  # [Qq, Tt]
            valid = ar[: t1 - t0].unsqueeze(0) + t0 < seq[i0:i1].unsqueeze(1)
            logits[i0:i1, t0:t1] = torch.where(valid, row, 0.0)
    return logits


def _ragged_fp8_mqa_logits(
    q_fp8, kv, kv_scales, weights, cu_starts, cu_ends, **_ignored
):
    """Semantics of aiter fp8_mqa_logits: -inf outside each row's [ks, ke)."""
    _note_call("ragged")
    q_b = _as_bf16(q_fp8).contiguous()  # [Q, H, D]
    k_b = _as_bf16(kv).contiguous()  # [T, D]
    Q, H, _ = q_b.shape
    T = k_b.shape[0]
    dev = q_b.device
    k_scale = kv_scales.reshape(-1).to(torch.float32)
    w = weights.reshape(Q, H).to(torch.float32)
    ks = cu_starts.to(torch.int64).reshape(-1)
    ke = cu_ends.to(torch.int64).reshape(-1)
    logits = torch.full((Q, T), float("-inf"), dtype=torch.float32, device=dev)

    qt = int(os.environ.get("HCU_MQA_EMU_Q_TILE", "128"))
    tt = int(os.environ.get("HCU_MQA_EMU_T_TILE", "2048"))
    ar = torch.arange(tt, device=dev)
    for i0 in range(0, Q, qt):
        i1 = min(Q, i0 + qt)
        qb = q_b[i0:i1]  # [Qq, H, D]
        for t0 in range(0, T, tt):
            t1 = min(T, t0 + tt)
            kt = k_b[t0:t1].transpose(0, 1)  # [D, Tt]
            scores = torch.matmul(qb, kt.unsqueeze(0))  # [Qq, H, Tt]
            sc = k_scale[t0:t1]
            row = (scores.float() * sc.view(1, 1, -1) * w[i0:i1].unsqueeze(-1)).sum(
                dim=1
            )
            valid = (ar[: t1 - t0].unsqueeze(0) + t0 >= ks[i0:i1].unsqueeze(1)) & (
                ar[: t1 - t0].unsqueeze(0) + t0 < ke[i0:i1].unsqueeze(1)
            )
            logits[i0:i1, t0:t1] = torch.where(valid, row, float("-inf"))
    return logits


def _triton_or_emu_faces():
    """(paged_fn, ragged_fn, impl_label) — triton arms first, emu per-call
    fallback for the faces the triton tier refuses (fnuz convention,
    non-132B rows, preshuffle pools): each refusal logs once, never silent.
    HCU_MQA_EMU_IMPL=triton (default) | emu (golden / A-B probe)."""
    from hcu_adapter import hcu_env

    impl, src = hcu_env.resolve("HCU_MQA_EMU_IMPL")
    if impl != "triton":
        return _paged_mqa_logits, _ragged_fp8_mqa_logits, f"emu ({src})"
    from .mqa_logits_triton import (
        triton_paged_ps1_mqa_logits as _tp,
        triton_ragged_mqa_logits as _tr,
    )

    warned = {"paged": 0, "ragged": 0}

    def _paged(q, kv, w, lens, bt, msl, *, preshuffle, kv_block_size, **kw):
        try:
            return _tp(
                q, kv, w, lens, bt, msl,
                preshuffle=preshuffle, kv_block_size=kv_block_size, **kw
            )
        except NotImplementedError as e:
            if warned["paged"] < 1:
                logger.warning(
                    "[hcu] MQA paged triton arm refused (%s) -> emu for this"
                    " face (and subsequent)", e)
                warned["paged"] += 1
            return _paged_mqa_logits(
                q, kv, w, lens, bt, msl,
                preshuffle=preshuffle, kv_block_size=kv_block_size, **kw
            )

    def _ragged(q, kv, kv_scales, weights, ks, ke, **kw):
        try:
            return _tr(q, kv, kv_scales, weights, ks, ke)
        except NotImplementedError as e:
            if warned["ragged"] < 1:
                logger.warning(
                    "[hcu] MQA ragged triton arm refused (%s) -> emu for"
                    " this face (and subsequent)", e)
                warned["ragged"] += 1
            return _ragged_fp8_mqa_logits(
                q, kv, kv_scales, weights, ks, ke, **kw
            )

    return _paged, _ragged, f"triton, emu fallback ({src})"


def install():
    import sglang.kernels.ops.attention.dsa as _dsa_pkg

    _paged_fn, _ragged_fn, _impl_label = _triton_or_emu_faces()

    if getattr(_dsa_pkg, "aiter_paged_mqa_logits", None) is not _paged_fn:
        assert _dsa_pkg.aiter_paged_mqa_logits is not None
        _dsa_pkg.aiter_paged_mqa_logits = _paged_fn

    import aiter.ops.triton.fp8_mqa_logits as _afml

    if getattr(_afml, "fp8_mqa_logits", None) is not _ragged_fn:
        _afml.fp8_mqa_logits = _ragged_fn

    try:
        from hcu_adapter import registry as _reg

        _reg.register(
            "attention",
            "mqa-logits-vllm-face",
            target=(
                "seam:sglang.kernels.ops.attention.dsa.aiter_paged_mqa_logits"
                " + aiter.ops.triton.fp8_mqa_logits.fp8_mqa_logits"
            ),
            mechanism=_reg.MECH_SEAM,
            detail=f"impl={_impl_label}",
        )
    except Exception:
        pass

    if os.environ.get("HCU_DUMP_MQA") == "1":
        _log = {"n": 0}

        def _norm(t):
            return (
                f"{tuple(t.shape)} {t.dtype}"
                f" nz={(t.float().abs() > 0).sum().item()}/{t.numel()}"
                if isinstance(t, torch.Tensor)
                else repr(t)[:40]
            )

        def _wrap_ragged(q_fp8, kv, kv_scales, weights, cu_starts, cu_ends,
                         **kw):
            out = _ragged_fn(
                q_fp8, kv, kv_scales, weights, cu_starts, cu_ends, **kw)
            if _log["n"] < 4:
                print(
                    f"[HCU mqa] ragged q={_norm(q_fp8)} kv={_norm(kv)}"
                    f" kv_sc={_norm(kv_scales)} w={_norm(weights)}"
                    f" ks={cu_starts.tolist()[:6]} ke={cu_ends.tolist()[:6]}"
                    f" -> out={_norm(out)}", flush=True)
                _log["n"] += 1
            return out

        _afml.fp8_mqa_logits = _wrap_ragged

        def _wrap_paged(q_fp8, kv_cache_fp8, weights, seq_lens, block_tables,
                        max_seq_len, **kw):
            out = _paged_fn(
                q_fp8, kv_cache_fp8, weights, seq_lens, block_tables,
                max_seq_len, **kw)
            if _log["n"] < 4:
                print(
                    f"[HCU mqa] paged q={_norm(q_fp8)} kv={_norm(kv_cache_fp8)}"
                    f" w={_norm(weights)} lens={seq_lens.tolist()[:6]}"
                    f" -> out={_norm(out)}", flush=True)
                _log["n"] += 1
            return out

        _dsa_pkg.aiter_paged_mqa_logits = _wrap_paged

    logger.warning(
        "[hcu] MQA-logits installed (impl=%s; paged/ragged triton arms with"
        " torch-emu fallback; gfx936 has no fp8 matrix unit so exact decode"
        " paths stay the numeric contract)", _impl_label
    )
