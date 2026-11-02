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


# flashinfer.sampling.softmax face — dspark verify SoftmaxTemp hot path
# (dspark_accept.softmax_temp_flashinfer). Runs once per decode step on
# [bs*gamma, vocab] logits inside the CUDA-graph region, so it is a triton
# port, not the torch reference tier. Semantics identical to flashinfer:
# probs = softmax(logits / temperature) row-wise, fp32 out; temperature is
# per-row (or one scalar broadcast); temp <= 0 is treated as 1.0 (greedy
# callers normalize upstream, this only guards div-by-0).
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # CPU-only import contexts must not break package import
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _softmax_rows_kernel(
        logits_ptr,
        temp_ptr,
        out_ptr,
        vocab,
        row_stride,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        temp = tl.load(temp_ptr + row).to(tl.float32)
        temp = tl.where(temp > 0.0, temp, 1.0)
        inv_t = 1.0 / temp
        base = logits_ptr + row.to(tl.int64) * row_stride
        out_base = out_ptr + row.to(tl.int64) * vocab

        m = -float("inf")
        l = 0.0
        for v0 in range(0, vocab, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            vmask = offs < vocab
            x = (
                tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
                * inv_t
            )
            m_new = tl.maximum(m, tl.max(x, axis=0))
            p = tl.where(vmask, tl.exp(x - m_new), 0.0)
            l = l * tl.exp(m - m_new) + tl.sum(p, axis=0)
            m = m_new

        inv_l = 1.0 / l
        for v0 in range(0, vocab, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            vmask = offs < vocab
            x = (
                tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
                * inv_t
            )
            tl.store(out_base + offs, tl.exp(x - m) * inv_l, mask=vmask)


def softmax(logits, temperature):
    """Row-wise softmax(logits / temperature), fp32 out (flashinfer face)."""
    if not _HAS_TRITON:
        raise RuntimeError("sgl_kernel.sampling.softmax requires triton")
    assert logits.ndim == 2, f"expected 2-D [rows, vocab], got {logits.shape}"
    num_rows, vocab = logits.shape
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    if not torch.is_tensor(temperature):
        temperature = torch.as_tensor(
            temperature, dtype=torch.float32, device=logits.device
        )
    temperature = temperature.to(torch.float32).reshape(-1)
    if temperature.numel() == 1:
        temp = temperature.expand(num_rows).contiguous()
    else:
        assert temperature.numel() == num_rows, (
            f"temperature numel {temperature.numel()} != rows {num_rows}"
        )
        temp = temperature.contiguous()
    out = torch.empty((num_rows, vocab), dtype=torch.float32, device=logits.device)
    if num_rows == 0 or vocab == 0:
        return out
    _softmax_rows_kernel[(num_rows,)](
        logits,
        temp,
        out,
        vocab,
        logits.stride(0),
        BLOCK_V=8192,
        num_warps=8,
    )
    return out
