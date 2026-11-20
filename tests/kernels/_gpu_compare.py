# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""GPU-native comparison helper (W4-4, governance 3b era test standard).

The old pattern -- ``observed.cpu().double()`` then cosine + max-diff +
assert_close, each with its own host sync -- paid 2-3 round trips per
test across ~1900 tests. This helper keeps every reduction ON DEVICE
and syncs ONCE (a small scalar bundle); the first-mismatch neighborhood
is pulled to host only on the failure path.

Standard going forward (roadmap W4): new numeric tests use
:func:`gpu_compare`; existing sites migrate opportunistically.

Tolerance modes:
- ``rtol``/``atol``: elementwise ``diff <= atol + rtol * |ref|`` (the
  torch.allclose rule, evaluated on device).
- ``bf16_ulp=True``: scale-aware bound from the parity gates -- allowed
  grows to one bf16 ulp of the largest output,
  ``max(atol, 2**(floor(log2(max|ref|)) - 7))`` (double-rounding tiers
  legitimately differ by exactly one ulp at magnitude).
- ``cosine=``: optional floor on cosine similarity (fp32, on device).
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def gpu_compare(
    got: torch.Tensor,
    ref: torch.Tensor,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    bf16_ulp: bool = False,
    cosine: float | None = None,
) -> None:
    """Assert ``got`` matches ``ref``; raise AssertionError with the
    first-mismatch neighborhood on failure (host pull only then)."""
    g = got.detach().float()
    r = ref.detach().float()
    if g.shape != r.shape:
        raise AssertionError(f"shape mismatch: {tuple(g.shape)} vs {tuple(r.shape)}")
    diff = (g - r).abs()
    md = diff.max()
    mref = r.abs().max()
    # cosine-only mode (no atol/rtol budget): similarity IS the criterion
    elementwise = not (atol == 0.0 and rtol == 0.0 and cosine is not None)
    viol = (diff > (atol + rtol * r.abs())) if elementwise else diff.isinf()
    if bf16_ulp:
        # one ulp of the largest output (bf16 mantissa 7 bits)
        viol = viol & (diff > (atol + torch.exp2(
            torch.floor(torch.log2(mref.clamp(min=1e-30))) - 7)))
    bad = viol.any()
    cos = torch.ones((), device=g.device)
    if cosine is not None:
        cos = torch.nn.functional.cosine_similarity(
            g.flatten(), r.flatten(), dim=0)
    # ONE host sync: [maxdiff, maxref, any_violation, cosine]
    bundle = torch.stack([md, mref, bad.float(), cos]).tolist()
    md_v, mref_v, bad_v, cos_v = bundle
    if bf16_ulp and mref_v > 0:
        atol = max(atol, math.ldexp(1.0, math.floor(math.log2(mref_v)) - 7))
    problems = []
    if bad_v and elementwise:
        idx = viol.reshape(-1).nonzero()[0].item()  # failure path only
        g_f, r_f, d_f = g.reshape(-1), r.reshape(-1), diff.reshape(-1)
        ctx = slice(max(0, idx - 1), idx + 2)
        problems.append(
            "mismatch: md=%.6g mref=%.6g atol=%.6g rtol=%.6g first@%d "
            "got=%s ref=%s diff=%s ctx_got=%s ctx_ref=%s"
            % (md_v, mref_v, atol, rtol, idx,
               g_f[idx].item(), r_f[idx].item(), d_f[idx].item(),
               g_f[ctx].tolist(), r_f[ctx].tolist())
        )
    if cosine is not None and cos_v < cosine:
        problems.append("cosine %.6f < %.6f" % (cos_v, cosine))
    if problems:
        raise AssertionError("; ".join(problems))


@torch.no_grad()
def gpu_maxdiff(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """(max-abs-diff, max-abs-ref) with a single host sync."""
    g = got.detach().float()
    r = ref.detach().float()
    md = (g - r).abs().max()
    mref = r.abs().max()
    return torch.stack([md, mref]).tolist()
