"""Sampling ops for the §0.5 is_cuda semantics — torch reference tier.

On the is_cuda route upstream sglang imports its sampler from
`flashinfer.sampling` + `sgl_kernel` (top_k/top_p renorm). flashinfer does
not exist on HCU, so both import faces bind here. These are plain masked
filter -> renormalize -> multinomial sequences in fp32 on device; called
once per decode step, so the torch tier is the right first landing
(triton/vendor optimization can follow via the catalog). Ties: top-k keeps
exactly k entries (deterministic sort); top-p keeps the smallest prefix
whose cumulative mass reaches p (cumsum-before-current < p).
"""

import torch


def top_k_renorm_prob(probs, top_ks):
    probs = probs.float()
    k = top_ks.to(torch.long).clamp_min(1).view(-1, 1)
    sp, idx = torch.sort(probs, dim=-1, descending=True)
    keep = torch.arange(sp.shape[1], device=probs.device).expand_as(sp) < k
    sp = sp * keep
    sp = sp / sp.sum(dim=-1, keepdim=True).clamp_min(1e-10)
    out = torch.zeros_like(sp)
    out.scatter_(1, idx, sp)
    return out


def top_p_renorm_prob(probs, top_ps):
    probs = probs.float()
    p = top_ps.float().clamp(0.0, 1.0).view(-1, 1)
    sp, idx = torch.sort(probs, dim=-1, descending=True)
    cum = sp.cumsum(dim=-1)
    keep = (cum - sp) < p  # smallest prefix covering p; keeps at least 1
    sp = sp * keep
    sp = sp / sp.sum(dim=-1, keepdim=True).clamp_min(1e-10)
    out = torch.zeros_like(sp)
    out.scatter_(1, idx, sp)
    return out


def min_p_sampling_from_probs(probs, min_ps, generator=None):
    probs = probs.float()
    thresh = min_ps.float().view(-1, 1) * probs.max(dim=-1, keepdim=True).values
    mask = probs >= thresh
    filtered = probs * mask
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-10)
    return torch.multinomial(filtered, 1, generator=generator).squeeze(1)


def top_k_top_p_sampling_from_probs(probs, top_ks, top_ps, generator=None):
    filtered = top_p_renorm_prob(top_k_renorm_prob(probs, top_ks), top_ps)
    return torch.multinomial(filtered, 1, generator=generator).squeeze(1)


def top_k_renorm_probs_alias(probs, top_ks):  # noqa: N802 - upstream plural face
    return top_k_renorm_prob(probs, top_ks)
