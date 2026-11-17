# SPDX-License-Identifier: Apache-2.0
"""Shared DSpark query/verification capacity contract, without framework imports."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenCapacity:
    query_per_request: int
    query_rows: int
    verify_rows: int
    budget: int


def token_capacity(cfg):
    spec, scheduler = cfg.speculative_config, cfg.scheduler_config
    n = spec.num_speculative_tokens
    requests = scheduler.max_num_seqs
    budget = scheduler.max_num_batched_tokens
    if any(type(value) is not int or value <= 0 for value in (n, requests, budget)):
        raise ValueError('DSpark requires positive integer n/requests/token budget')
    anchor = getattr(spec.draft_model_config.hf_config, 'sample_from_anchor', True)
    if type(anchor) is not bool:
        raise ValueError('DSpark sample_from_anchor must be bool')
    query = n if anchor else n + 1
    result = TokenCapacity(query, requests * query, requests * (n + 1), budget)
    if max(result.query_rows, result.verify_rows) > budget:
        raise ValueError(
            f'DSpark capacity exceeds original budget: query={result.query_rows}, '
            f'verify={result.verify_rows}, budget={budget}')
    return result
