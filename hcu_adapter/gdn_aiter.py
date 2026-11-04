# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""GDN / gated-delta-rule linear attention -> DTK aiter CK kernels
(catalog "linear_attn"; §0.5 route tier-② vendor-first, tier-③ triton
stays the default until this graduates).

The upstream faces (sglang 0.5.21, ``sglang.kernels.ops.attention.fla``):

- ``chunk_gated_delta_rule_fwd_h`` (prefill state recursion, per-chunk
  HBM stream) — the heaviest prefill stage; aiter ships the CK
  ``chunk_gated_delta_rule_fwd_sglang_hip_blockdim64`` with the same
  (h, v_new) return contract and the same [B, NT, H, V, K] h layout
  (gfx93x wave64-tuned, blockdim64).
- ``fused_recurrent_gated_delta_rule_packed_decode`` (decode fused
  QKV+gating+recurrence) — aiter ships
  ``aiter_fused_recurrent_gated_delta_rule_packed_decode`` with the same
  args except: no state-pool index argument (gather/scatter on the Python
  side) and fp32 state in / state out.

What this bridge does (MECH_REBIND, two import sites per face):

- fwd_h: keep upstream cumsum + intra (k/w/u) and chunk_o on the upstream
  triton kernels (cheap, already correct); swap only the state recursion
  for the aiter CK kernel. Envelope-strided state pools are handled with
  the gather/compute/scatter dance the operator team's sglang integration
  uses (aiter's API has no stride_init_state).
- packed decode: gather the batch's state slots to a contiguous fp32
  buffer, run the aiter packed kernel, scatter states back to the pool
  (fp32 -> pool dtype) and return ``(out_view, ht)`` matching the
  upstream ``(out, final_state)`` contract. Gate math (softplus of
  ``-exp(A_log)*dt + dt_bias``) belongs to the kernel on both sides.

Gate: HCU_LINEAR_ATTN_GDN = auto|aiter|triton (auto = triton until the
numerics gate + e2e row graduate this to candidate/production).
"""

import logging

import torch

LOG = logging.getLogger("hcu.adapter")


def _gather_states(pool, indices):
    """Contiguous fp32 copy of the batch's state slots (aiter has no
    state-pool index argument and wants fp32)."""
    idx = indices.clamp_min(0).to(torch.long)
    local = torch.arange(indices.numel(), device=indices.device,
                         dtype=indices.dtype)
    states = pool.index_select(0, idx).to(dtype=torch.float32).contiguous()
    return states, local


def _scatter_states(pool, indices, local_indices, states):
    valid = indices >= 0
    if valid.any():
        pool.index_copy_(
            0, indices[valid].to(torch.long),
            states[valid].to(dtype=pool.dtype),
        )


def _install_fwd_h_aiter(registry):
    """Swap sglang.kernels...fla.chunk_delta_h.chunk_gated_delta_rule_fwd_h
    for the aiter CK state recursion; rebind the chunk.py consumer too."""
    from sglang.kernels.ops.attention.fla import chunk_delta_h as _cdh
    from sglang.kernels.ops.attention.fla import chunk as _chunk_mod

    _orig_fwd_h = _cdh.chunk_gated_delta_rule_fwd_h

    def chunk_gated_delta_rule_fwd_h(
        k, w, u, g=None, gk=None, initial_state=None,
        initial_state_indices=None, save_new_value=True,
        cu_seqlens=None, chunk_indices=None, use_exp2=False,
        inplace_update=True, track_state=None, track_chunk_idx=None,
    ):
        # aiter instantiates K=V=128 BT=64 (the GDN production shape);
        # anything else keeps the upstream triton kernel.
        if gk is not None or track_state is not None or use_exp2:
            return _orig_fwd_h(
                k, w, u, g, gk, initial_state, initial_state_indices,
                save_new_value, cu_seqlens, chunk_indices, use_exp2,
                inplace_update, track_state, track_chunk_idx,
            )
        B, T, Hg, K = k.shape
        H, V = u.shape[-2], u.shape[-1]
        if K != 128 or V != 128:
            return _orig_fwd_h(
                k, w, u, g, gk, initial_state, initial_state_indices,
                save_new_value, cu_seqlens, chunk_indices, use_exp2,
                inplace_update, track_state, track_chunk_idx,
            )
        import aiter.ops.fla as _afla

        if chunk_indices is None and cu_seqlens is not None:
            from sglang.kernels.ops.attention.fla.index import (
                prepare_chunk_indices,
            )

            chunk_indices = prepare_chunk_indices(cu_seqlens, 64)

        aiter_state = initial_state
        aiter_idx = initial_state_indices
        valid_mask = None
        if initial_state is not None and initial_state_indices is not None:
            valid_mask = initial_state_indices >= 0
            aiter_state, aiter_idx = _gather_states(
                initial_state, initial_state_indices)

        h, v_new = _afla.chunk_gated_delta_rule_fwd_sglang_hip_blockdim64(
            k=k, w=w, u=u, g=g, gk=None,
            initial_state=aiter_state,
            initial_state_indices=aiter_idx,
            output_final_state=True,
            chunk_size=64,
            save_new_value=save_new_value,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            transpose_state_layout=True,
        )
        if valid_mask is not None:
            _scatter_states(
                initial_state, initial_state_indices, aiter_idx, aiter_state)
        return h, v_new

    for ns in (_cdh, _chunk_mod):
        pass  # chunk.py imports fwd_h at module top; see rebind below
    _cdh.chunk_gated_delta_rule_fwd_h = chunk_gated_delta_rule_fwd_h
    # chunk.py bound the original symbol at import time (from-import);
    # the is_intel branch absent on HCU means its module-global
    # chunk_gated_delta_rule_fwd_h IS the chunk_delta_h one — rebind there
    # as well so ChunkGatedDeltaRuleFunction calls ours.
    import sys

    already = sys.modules.get("sglang.kernels.ops.attention.fla.chunk")
    if already is not None:
        already.chunk_gated_delta_rule_fwd_h = chunk_gated_delta_rule_fwd_h
    registry.register(
        "linear_attn_gdn",
        chunk_gated_delta_rule_fwd_h,
        target="sglang.kernels...fla.chunk_delta_h:chunk_gated_delta_rule_fwd_h",
        mechanism=registry.MECH_REBIND,
        detail="state recursion (fwd_h) -> aiter CK blockdim64; K=V=128 only, "
               "rest keeps upstream triton; state pool gather/scatter wrap",
    )
    LOG.warning(
        "HCU adapter: GDN prefill state recursion routed to aiter CK "
        "(chunk_gated_delta_rule_fwd_h; K=V=128, gather/scatter state pool)"
    )


def _install_packed_decode_aiter(registry):
    """Swap the packed decode fused kernel for aiter's triton fla variant
    (same contract incl. state-pool indices — the operator team's own
    sglang alignment; drop-in rebind)."""
    from sglang.kernels.ops.attention.fla import fused_recurrent as _fr
    from aiter.ops.triton.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode as _aiter_decode,
    )

    _fr.fused_recurrent_gated_delta_rule_packed_decode = _aiter_decode
    registry.register(
        "linear_attn_gdn",
        _aiter_decode,
        target="sglang.kernels...fla.fused_recurrent:fused_recurrent_gated_delta_rule_packed_decode",
        mechanism=registry.MECH_REBIND,
        detail="packed decode -> aiter triton fla (same contract, "
               "state-pool indices native)",
    )
    LOG.warning(
        "HCU adapter: GDN packed decode routed to aiter triton fla kernel "
        "(same contract drop-in)"
    )


def install():
    from hcu_adapter import hcu_env

    mode, _src = hcu_env.resolve("HCU_LINEAR_ATTN_GDN")
    if mode != "aiter":
        LOG.warning(
            "HCU adapter: HCU_LINEAR_ATTN_GDN=%s -- GDN stays on the "
            "upstream triton fla kernels (aiter route available via =aiter)",
            mode,
        )
        return
    try:
        import aiter  # noqa: F401
        import aiter.ops.fla  # noqa: F401
    except Exception:
        LOG.warning(
            "HCU adapter: aiter not importable; GDN stays on triton fla",
            exc_info=True,
        )
        return

    from hcu_adapter import registry

    _install_fwd_h_aiter(registry)
    _install_packed_decode_aiter(registry)
