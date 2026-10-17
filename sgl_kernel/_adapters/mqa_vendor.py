"""Vendor (DTK deepgemm wheel) paged MQA-logits bridge (R-02).

The paged MQA logits path is a decode hot spot; the vec-emu (e70f45e) is
the current production form and stays the parity gold standard. The DTK
``deepgemm`` wheel (2.1.0+dtk2604, already aliased behind
HCU_DEEPGEMM_BRIDGE) ships a native ``paged_mqa_logits``:

    paged_mqa_logits(q, fused_kv_cache, weights, context_lens,
                     block_table, max_model_len, *,
                     out=None, clean_logits=True, is_prefill=False)

which is signature-isomorphic to our emu entry (q/kv/weights/lens/bt/
max_len; page size 64 slots, kv row = 128 fp8 bytes + fp32 scale).
``aiter.ops.paged_mqa_logits`` does NOT exist in the on-box aiter wheel
(probed 2026-10-18, nmz26 dtk26041) — that tier is retired.

UNVALIDATED + LAYOUT CONTRACT FOUND (probe 2026-10-18, nmz26 GPU4): the
op runs with q=float8_e4m3fn (fnuz rejected by the metadata gate — matches
the is_cuda route where is_fp8_fnuz() is False), H in {8,16,32,64} (GLM-5.3
index_n_heads=32 fits), kv uint8 shaped (P,S,1,132), weights fp32 (N,H),
lens/bt int32 — but it READS the ORIGINAL preshuffle kpool page layout
(64 slots: 8192B of 16-tile-interleaved fp8 + 256B of fp32 scales), not
the flat [T,132] tensor the kpool READ-emu unpacks for the torch emu.
Proper wiring must feed it the RAW pool at the dsa layer (bypass the
read-emu unpack); until then HCU_MQA_LOGITS_IMPL=vendor is an A/B probe
tier and tests/kernels/test_mqa_vendor_gfx938.py pins the known mismatch
as a RED watchtower. ``aiter.ops.paged_mqa_logits`` does NOT exist in the
on-box aiter wheel (probed same day) — that tier is retired.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def deepgemm_paged_mqa_logits(
    q_fp8,
    kv_cache_fp8,
    weights,
    context_lens,
    block_table,
    schedule_metadata=None,
    max_context_len=None,
    clean_logits=False,
    **_ignored,
):
    """Adapter entry for HCU_MQA_LOGITS_IMPL=vendor (deepgemm wheel)."""
    from deepgemm.paged_mqa_logits import paged_mqa_logits as _dg

    q = q_fp8
    if q.dtype in (torch.uint8, torch.int8):
        # vendor hard-requires float8_e4m3fn (probed 2026-10-18: fnuz q is
        # rejected by the metadata gate) — matches the is_cuda platform
        # route where is_fp8_fnuz() is False
        q = q.contiguous().view(torch.float8_e4m3fn)
    if q.dim() == 3:  # vendor contract: 4D [B, R, H, 128] with B>0
        q = q.reshape(q.shape[0], 1, q.shape[1], q.shape[2])
    kv = kv_cache_fp8
    if kv.dtype in (torch.uint8, torch.int8):
        kv = kv.contiguous()  # fused row = 128 fp8 bytes + fp32 scale; the
        # uint8 storage passes as-is (mixed row cannot be dtype-viewed)
    if kv.dim() == 2:  # vendor contract: 4D [P, S, 1, D+scale]
        kv = kv.reshape(-1, 64, 1, kv.shape[-1])
    if weights.dim() == 3:
        weights = weights.reshape(weights.shape[0], -1)
    if context_lens.dim() == 2:
        context_lens = context_lens.reshape(-1)
    max_len = int(max_context_len or block_table.shape[1] * 64)
    if block_table.dtype == torch.int64:
        block_table = block_table.to(torch.int32)
    if context_lens.dtype == torch.int64:
        context_lens = context_lens.to(torch.int32)
    out = _dg(
        q,
        kv,
        weights,
        context_lens,
        block_table,
        max_len,
        clean_logits=bool(clean_logits),
    )
    return out
