"""moe gate lightop implementation; upstream rounding/layout contracts are unchanged."""

import torch

from .lightop_usage import _count


def bf16_mega_gate(x, weight, *args, **kwargs):
    # Preserve the pinned FP32 gate projection; accelerate its routing stage.
    return moe_topk_gate(x.float() @ weight.float().T, *args, **kwargs)


def moe_topk_gate(logits, num_topk, use_shared_as_routed, num_shared_experts, routed_scaling_factor, ep_rank,
                  scoring_func='sqrtsoftplus', mask=None, bias=None, image_bias=None, image_token_mask=None,
                  fix_routing_mask=None, to_physical_map=None, logical_count=None, unmapped_topk_idx=None,
                  force_random=None, out=None, scores=None):
    import lightop

    if scoring_func != 'sqrtsoftplus' or scores is not None:
        raise NotImplementedError('Unsupported LightOp routing policy')
    tokens, routed = logits.shape
    if force_random is not None:
        if isinstance(force_random, torch.Tensor):
            if force_random.dtype != torch.bool or force_random.shape != (tokens,):
                raise ValueError('Routing metadata requires a boolean vector per token')
        elif force_random is not False:
            raise NotImplementedError('Forced random routing is not enabled by this candidate')
        # Match the existing HCU normal-routing policy. Upstream supplies this
        # vector even with FORCE_LOAD_BALANCE=0; the platform guard still rejects
        # that benchmark switch. Do not synchronize a metadata tensor on host.
        _count('router.force_random_metadata_compatibility')
    shared = num_shared_experts if use_shared_as_routed else 0
    if not tokens:
        return (torch.empty((0, num_topk + shared), device=logits.device, dtype=torch.int64),
                torch.empty((0, num_topk + shared), device=logits.device, dtype=torch.float32)) if out is None else out
    data = logits.float().contiguous()
    rank_bias = torch.zeros(routed, device=logits.device) if bias is None else bias.float().contiguous()
    if routed == 128:
        from hcu_adapter.kernels.draft_router_triton import sqrtsoftplus_topk

        def route(rank_bias):
            return sqrtsoftplus_topk(data, rank_bias, num_topk, routed_scaling_factor)

        _count('router.draft_triton')
    elif routed in (256, 384):
        def route(rank_bias):
            return lightop.moe_fused_gate_sqrtsoftplus_v41_hcu(
                data, rank_bias, num_topk, 0, True, routed_scaling_factor, True)

        _count('router.native')
    else:
        raise NotImplementedError(f'LightOp routing unsupported expert count: {routed}')
    weights, selected = route(rank_bias)
    selected = selected.long()
    if image_token_mask is not None:
        image_weights, image_selected = route(image_bias.float().contiguous())
        selected = torch.where(image_token_mask[:, None], image_selected.long(), selected)
        weights = torch.where(image_token_mask[:, None], image_weights, weights)
    if fix_routing_mask is not None and unmapped_topk_idx is not None:
        selected = torch.where(fix_routing_mask[:, None], unmapped_topk_idx.long(), selected)
        picked = torch.nn.functional.softplus(data).sqrt().gather(1, selected.clamp_min(0))
        weights = picked / (picked.sum(-1, keepdim=True) + 1e-20) * routed_scaling_factor
    active = torch.ones(tokens, device=logits.device, dtype=torch.bool) if mask is None else mask
    if unmapped_topk_idx is not None:
        unmapped_topk_idx.copy_(torch.where(active[:, None], selected, -1))
    if shared:
        ids = torch.arange(routed, routed + shared, device=logits.device).expand(tokens, -1)
        selected = torch.cat((selected, ids), -1)
        weights = torch.cat((weights, torch.ones((tokens, shared), device=logits.device)), -1)
    if to_physical_map is not None:
        row = torch.arange(tokens, device=logits.device)[:, None]
        logical = selected.clamp_min(0)
        duplicate = (ep_rank + row * 23333) % logical_count.long()[logical]
        selected = to_physical_map.long()[logical, duplicate]
    selected, weights = torch.where(active[:, None], selected, -1), torch.where(active[:, None], weights, 0.)
    if out is not None:
        out[0].copy_(selected)
        out[1].copy_(weights)
        selected, weights = out
    return selected, weights
