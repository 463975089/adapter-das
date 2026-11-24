# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""LightOp provider for Core's expert unpermute/reduce calling convention."""


def unpermute_and_reduce(a, topk_ids, topk_weights, inv_perm, expert_map, output):
    from lightop.moe import ep_gather

    ep_gather(a, topk_ids, topk_weights, inv_perm, expert_map, output)
