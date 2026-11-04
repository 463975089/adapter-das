# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Vendor (DTK deepgemm wheel) paged MQA-logits bridge (R-02).

The paged MQA logits path is a decode hot spot; the vec-emu (e70f45e) is
the current production form and stays the parity gold standard. The DTK
``deepgemm`` wheel (2.1.0+dtk2604, already aliased behind
HCU_DEEPGEMM_BRIDGE) ships a native ``paged_mqa_logits``:

    paged_mqa_logits(q, fused_kv_cache, weights, context_lens,
                     block_table, max_model_len, *,
                     out=None, clean_logits=True, is_prefill=False)

which is signature-isomorphic to our emu entry (q/kv/weights/lens/bt/
max_len; page size 64 slots — see the recovered layout contract below).
``aiter.ops.paged_mqa_logits`` does NOT exist in the on-box aiter wheel
(probed 2026-10-18, nmz26 dtk26041) — that tier is retired.

LAYOUT CONTRACT RECOVERED + WHEEL BUG CONVICTED (probe campaign
2026-10-18, nmz26 GPU4, deterministic across repeats):

* Layout: fused_kv_cache (P,64,1,132) uint8, page = 8448B with k stored
  ROW-MAJOR (slot s -> bytes [s*128, s*128+128)) followed by 64 fp32
  scales at the page tail [8192,8448) — the upstream DeepGEMM SM90
  layout (csrc/apis/attention.hpp). CAVEAT (2026-10-18 writer census):
  that is the kpool=1 world's layout (fused_store_index_k_cache, the
  _is_cuda store arm). GLM-5.3 (IndexerKPool, kpool=4) never reaches
  that store — its pool is written by the kpool writer family + SetKAndS
  under the adapter-forced preshuffle gate, i.e. k in the 16-tile
  INTERLEAVE order + tail scales (see mqa_logits_triton.py / dsa_mqa_logits_emu.py).
  So even with a fixed wheel, wiring the vendor tier for kpool models
  needs a pool relayout or a writer switch; for kpool=1 models the
  bridge is layout-correct as-is. The wheel bug below blocks either way.

* FATAL WHEEL BUG: the gfx938 kernel clamps NEGATIVE fp8 values to
  zero on BOTH operands — logits == sum_h w_h * max(q_h,c, 0) *
  max(k_t,c, 0) * scale_t, EXACTLY (fitted to 4 decimals over 9 probe
  geometries; all-positive test data passes perfectly, single negative
  q or k byte contributes exactly 0.0). Real DSA q/k are ~half
  negative, so serving output would be garbage (rel~3.2 vs emu gold).
  A 4-call sign-split decomposition (q+/q- x k+/k-) recovers exact
  math in principle but needs a duplicated pool per call — not worth
  wiring; blocked on a DTK kernel fix.

Wheel: deepgemm 2.1.0+dtk2604.torch2110.2610131423.g112a50
(op.cpython .so, gfx938/paged_mqa_logits_fp8.hip). DTK gap-list item.
Until a fixed wheel lands, HCU_MQA_LOGITS_IMPL=vendor is a triage-only
override and the BACKEND_CATALOG marks the tier broken (selecting it
via hcu_adapter.yaml raises). tests/kernels/test_mqa_vendor_gfx938.py
pins BOTH facts: all-positive data matches the emu exactly (bridge +
layout correct) and signed data mismatches (bug signature stable).
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
