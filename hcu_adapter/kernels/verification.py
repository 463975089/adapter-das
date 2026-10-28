"""Sampler verification kernels for the pinned tilelang_ops sampler/sum_smaller_probs contracts.

Anchors: deepinfer tilelang_ops/sampler.py::{prepare_verification_metadata,expand_sampling_params,
map_lookup,remap_draft_topk,get_accept_result,finalize_verification}_ref and
sum_smaller_probs.py::sum_smaller_probs_ref @774ce24. The references use boolean-mask
indexing (rows[is_final]); these rewrites stay on device (scatter/gather/where only), so
they run under the forward sync guard and graph capture.
"""

import torch

DRAFT_POS_HASH_MUL = 10000019  # tilelang_ops/common.py @774ce24


def prepare_verification_metadata(ftp, seqlen_q, row_to_req, local_step, is_verify):
    """Packed request rows -> (request, local step, is_verify); rows outside every segment untouched."""
    total_q = row_to_req.shape[0]
    if not total_q or not ftp.shape[0]:
        return
    rows = torch.arange(total_q, dtype=torch.int32, device=ftp.device)
    req = (torch.searchsorted(ftp.contiguous(), rows, right=True) - 1).clamp(min=0)
    step = rows - ftp[req]
    length = seqlen_q[req]
    inside = (step >= 0) & (step < length)
    row_to_req.copy_(torch.where(inside, req.to(torch.int32), row_to_req))
    local_step.copy_(torch.where(inside, step, local_step))
    is_verify.copy_(torch.where(inside, step + 1 < length, is_verify))


def expand_sampling_params(temperatures, top_ps, top_ks, sample_positions, seq_seeds, row_to_req, local_step,
                           temperatures_q, top_ps_q, top_ks_q, sample_positions_q, seq_seeds_q):
    req = row_to_req.long()
    temperatures_q.copy_(temperatures[req])
    top_ps_q.copy_(top_ps[req])
    top_ks_q.copy_(top_ks[req])
    # int32 arithmetic wraps like the kernel.
    sample_positions_q.copy_(sample_positions[req] + local_step * DRAFT_POS_HASH_MUL)
    seq_seeds_q.view(torch.int64).copy_(seq_seeds.view(torch.int64)[req])


def map_lookup(key, value, query, out, *, by_key):
    """by_key: value whose key == query (-inf if none); else value[row, query] (OOB -> -inf / -1)."""
    q = query[:, 0]
    if by_key:
        hit = key == q[:, None]
        result = torch.where(hit, value, float("-inf")).amax(dim=-1)
    else:
        width = value.shape[-1]
        in_bounds = (q >= 0) & (q < width)
        picked = value.gather(1, torch.where(in_bounds, q, 0).long()[:, None])[:, 0]
        result = torch.where(in_bounds, picked, float("-inf") if value.dtype.is_floating_point else -1)
    out[:, 0].copy_(result)


def remap_draft_topk(draft_topk_index, draft_topk_logprobs, draft_token_ids_pool, target_topk_index, read_rows,
                     aligned_logprobs, draft_local_ids, identity_rows):
    """Draft top-k logprobs re-expressed in the target top-k order (-inf where absent).

    Per-row sorted search instead of the reference's dense [rows, vocab] scatter.
    """
    rows = read_rows.long()
    draft_idx = draft_topk_index.index_select(0, rows)
    draft_lp = draft_topk_logprobs.index_select(0, rows)
    draft_ids = draft_token_ids_pool.index_select(0, rows)
    sorted_idx, order = torch.sort(draft_idx, dim=1, stable=True)
    target = target_topk_index.contiguous()
    pos = torch.searchsorted(sorted_idx, target).clamp(max=draft_idx.shape[1] - 1)
    hit = sorted_idx.gather(1, pos) == target
    lp = draft_lp.gather(1, order.gather(1, pos))
    aligned_logprobs.copy_(torch.where(hit, lp, float("-inf")))
    match = target == draft_ids[:, None]
    draft_local_ids.copy_(torch.where(match.any(1), match.to(torch.uint8).argmax(1), -1))
    identity_rows.copy_(torch.arange(target.shape[0], dtype=torch.int32, device=target.device))


def get_accept_result(raw_accept, golden, draft_token_ids_pool, read_rows, candidate_token_ids, is_verify, ftp,
                      row_to_req, token_ids_out, accept_out, is_final_out, *, has_golden):
    accept = raw_accept
    tokens = candidate_token_ids
    if has_golden:
        draft = draft_token_ids_pool.index_select(0, read_rows.long())
        use_golden = golden != -1
        accept = torch.where(use_golden, golden == draft, accept)
        tokens = torch.where(use_golden, golden, tokens)
    blocked = ~(is_verify & accept)
    prefix = torch.cat([blocked.new_zeros(1, dtype=torch.int32), blocked.cumsum(0, dtype=torch.int32)])
    alive = prefix[:-1] == prefix[ftp[row_to_req.long()].long()]
    token_ids_out.copy_(tokens)
    accept_out.copy_(alive & is_verify & accept)
    is_final_out.copy_(alive & blocked)


def finalize_verification(effective_accept, is_final, effective_token_ids, token_logprobs, sum_smaller_probs,
                          top_logprobs, top_token_ids, ftp, seqlen_q, row_to_req, is_verify, accept_probs,
                          confidence_pool, mtp_info_read_rows, stop_token_ids, last_valid_offset,
                          num_generated_tokens, num_verified_tokens, batch_confidence_error, emitted_stop_token,
                          sampled_token_ids, sampled_token_logprobs, sampled_sum_smaller_probs, sampled_top_logprobs,
                          sampled_top_token_ids, verified_token_ids, verified_token_logprobs,
                          verified_sum_smaller_probs, verified_top_logprobs, verified_top_token_ids, *,
                          has_sum_smaller_probs, has_confidence, num_steps):
    """One final row per request; the final row is found by scatter instead of rows[is_final]."""
    q = effective_accept.shape[0]
    req_cnt = ftp.shape[0]
    device = ftp.device
    rows = torch.arange(q, dtype=torch.int32, device=device)
    req = row_to_req.long()
    # Non-final rows land in a spare slot req_cnt (exactly one final row per request upstream).
    slot = torch.where(is_final, req, req_cnt)
    final_row = torch.zeros(req_cnt + 1, dtype=torch.int32, device=device).scatter_(0, slot, rows)[:req_cnt]
    final_index = final_row.long()
    last_valid_offset.copy_(final_row - ftp)

    step_index = (rows - ftp[req]).clamp(max=num_steps - 1).long()
    num_generated_tokens.zero_().scatter_add_(0, step_index, effective_accept.to(torch.int32))
    num_verified_tokens.zero_().scatter_add_(0, step_index, is_verify.to(torch.int32))

    if has_confidence:
        confidence = confidence_pool.index_select(0, mtp_info_read_rows.long())
        batch_confidence_error.copy_(torch.where(is_verify, torch.abs(accept_probs - confidence), 0.0).sum().view(1))
        row_stops = stop_token_ids.index_select(0, req)
        stop_hits = (effective_token_ids[:, None] == row_stops).any(dim=1)
        emitted_rows = (effective_accept | is_final) & stop_hits
        hits = torch.zeros(req_cnt, dtype=torch.int32, device=device).scatter_add_(0, req, emitted_rows.to(torch.int32))
        emitted_stop_token.copy_(hits > 0)
    else:
        batch_confidence_error.zero_()
        emitted_stop_token.zero_()

    sampled_token_ids.copy_(effective_token_ids.index_select(0, final_index))
    sampled_token_logprobs.copy_(token_logprobs.index_select(0, final_index))
    verified_token_ids.copy_(torch.where(effective_accept, effective_token_ids, -1))
    verified_token_logprobs.copy_(torch.where(effective_accept, token_logprobs, 0.0))
    if has_sum_smaller_probs:
        sampled_sum_smaller_probs.copy_(sum_smaller_probs.index_select(0, final_index))
        verified_sum_smaller_probs.copy_(torch.where(effective_accept, sum_smaller_probs, 0.0))
    if top_logprobs.shape[-1]:
        sampled_top_logprobs.copy_(top_logprobs.index_select(0, final_index))
        sampled_top_token_ids.copy_(top_token_ids.index_select(0, final_index))
        verified_top_logprobs.copy_(torch.where(effective_accept[:, None], top_logprobs, 0.0))
        verified_top_token_ids.copy_(torch.where(effective_accept[:, None], top_token_ids, 0))


_SSP_CHUNK_ELEMENTS = 1 << 26  # bound the [rows, vocab] masks per chunk


def sum_smaller_probs(logprobs, sample_results, output):
    """Inclusive ascending-probability CDF through the sampled token (ties by column); -1 rows -> 0."""
    rows, vocab = logprobs.shape
    if not rows:
        return
    step = max(1, _SSP_CHUNK_ELEMENTS // max(1, vocab))
    col = torch.arange(vocab, device=logprobs.device)[None, :]
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        lp = logprobs[r0:r1]
        sample = sample_results[r0:r1].long()
        invalid = sample < 0
        safe = sample.clamp(min=0)
        pivot = lp.gather(1, safe[:, None])
        keep = (lp < pivot) | ((lp == pivot) & (col <= safe[:, None]))
        out = (lp.exp() * keep).sum(dim=1)
        output[r0:r1] = out.masked_fill(invalid, 0.0)
