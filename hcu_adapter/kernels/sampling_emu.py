# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Sampling kernels for the pinned tilelang_ops fused_sample / random_sampling contracts.

Anchors: deepinfer tilelang_ops/fused_sample.py::fused_sample_ref and
random_sampling.py::random_sampling_ref @774ce24 (the upstream references,
which upstream documents may differ numerically from the TileLang kernel; its
tests check sampling properties, not bits). Device-only: no host reads.
"""

import torch

DRAFT_PROB_PENALTY = 5e-5
_GOLDEN = -7046029254386353131  # i64(0x9E3779B97F4A7C15)
_MUL1 = -4658895280553007687     # i64(0xBF58476D1CE4E5B9)
_MUL2 = -7723592293110705685     # i64(0x94D049BB133111EB)


def _urshift(x, n):
    return torch.bitwise_right_shift(x, n) & ((1 << (64 - n)) - 1)


def hash_to_uniform(seed):
    """SplitMix64(seed) -> uniform float64 in [0, 1); int64 arithmetic wraps like uint64."""
    z = seed.to(torch.int64) + _GOLDEN
    z = (z ^ _urshift(z, 30)) * _MUL1
    z = (z ^ _urshift(z, 27)) * _MUL2
    z = z ^ _urshift(z, 31)
    return _urshift(z, 11).to(torch.float64) * (1.0 / (1 << 53))


_SAMPLE_CHUNK_ELEMENTS = 1 << 26  # bound the FP64 CDF per row chunk (also keeps HIP grids in range)


def random_sampling(probs, seq_seeds, sample_positions, is_nextn_predict=False):
    sample_case = 1 if is_nextn_predict else 0
    seq = (sample_positions.to(torch.int64) * 3 + sample_case) * _GOLDEN
    combined = seq_seeds.to(torch.int64) + (seq ^ (seq * (1 << 13)))
    u = hash_to_uniform(combined).unsqueeze(-1)
    rows, vocab = probs.shape
    out = torch.empty(rows, dtype=torch.int32, device=probs.device)
    step = max(1, _SAMPLE_CHUNK_ELEMENTS // max(1, vocab))
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        cdf = probs[r0:r1].double().cumsum(dim=-1)
        out[r0:r1] = (cdf > u[r0:r1] * cdf[:, -1:]).to(torch.uint8).argmax(dim=-1).to(torch.int32)
    return out


def _apply_top_p_top_k(logits, top_p, top_k):
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True, stable=True)
    sorted_remove = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p.unsqueeze(1)
    remove = torch.zeros_like(sorted_remove).scatter_(1, sorted_idx, sorted_remove)
    logits = logits.masked_fill(remove, float("-inf"))
    vocab = logits.shape[1]
    order = torch.argsort(logits, dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(order).scatter_(1, order, torch.arange(vocab, device=logits.device).expand_as(order))
    remove = (top_k >= 0).unsqueeze(1) & (ranks >= top_k.clamp(max=vocab).unsqueeze(1))
    return logits.masked_fill(remove, float("-inf"))


def fused_sample(logits, temperatures, top_ps, top_ks, sample_positions, seq_seeds, is_verify, logprobs,
                 candidate_token_ids, last_nextn_token_logprobs, mtp_info_read_rows, last_next_n_token_ids,
                 accept_result_verify, draft_selected_logprobs, accept_probs, seed_offset, *,
                 do_sample, do_reject_sampling):
    work = logits.float() * (1.0 / temperatures).unsqueeze(1)  # logits are read-only
    work = _apply_top_p_top_k(work, top_ps, top_ks)
    lp = torch.log_softmax(work, dim=-1)
    logprobs.copy_(lp)
    if not do_sample:
        return
    probs = lp.exp()
    seed_block = seq_seeds.to(torch.int64) + seed_offset
    sample = random_sampling(probs, seed_block, sample_positions)
    if do_reject_sampling:
        rows = mtp_info_read_rows.long()
        draft_lp = last_nextn_token_logprobs.index_select(0, rows)
        draft_ids = last_next_n_token_ids.index_select(0, rows)
        sentinel = draft_ids < 0
        safe_ids = draft_ids.clamp(min=0).long()
        residual = (probs - (draft_lp - DRAFT_PROB_PENALTY).exp()).relu_()
        accept = torch.where(is_verify, (1.0 - residual.sum(-1)).clamp_min(0.0), 1.0)
        draft_token_lp = draft_lp.gather(1, safe_ids[:, None]).squeeze(1)
        target_lp = lp.gather(1, safe_ids[:, None]).squeeze(1)
        accept_lp = torch.clamp(target_lp - draft_token_lp, max=0)
        verify_seq = sample_positions.to(torch.int64) * 3 + 2
        u = hash_to_uniform(seed_block ^ (verify_seq * _GOLDEN)).to(accept_lp.dtype)
        accepted = (accept_lp.exp() > u).masked_fill(sentinel, False)
        verdict = torch.where(is_verify, accepted, torch.ones_like(accepted))
        reject_sample = random_sampling(residual, seed_block, sample_positions, is_nextn_predict=True)
        sample = torch.where(is_verify, torch.where(verdict, draft_ids, reject_sample), sample)
        accept_result_verify.copy_(verdict)
        draft_selected_logprobs.copy_(draft_token_lp.masked_fill(sentinel, float("-inf")))
        accept_probs.copy_(accept)
    candidate_token_ids.copy_(sample)
