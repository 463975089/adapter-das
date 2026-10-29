"""rope lightop implementation; upstream rounding/layout contracts are unchanged."""

import torch

from .lightop_usage import _count


def apply_rotary(query, cos_sin_cache, key=None, positions=None, interleaved=False,
                 conjugate=False, seqlen_offset=0):
    import lightop

    for x in (query,) if key is None else (query, key):
        if x.shape[-1] != cos_sin_cache.shape[1] or x.ndim not in (3, 4):
            raise ValueError('Unsupported LightOp rotary shape')
        length = x.shape[-3]
        if positions is None:
            pos = torch.arange(length, device=x.device)
            if x.ndim == 4:
                pos = pos.repeat(x.shape[0])
        else:
            pos = positions.reshape(-1).long()
        work = x.float().reshape(-1, x.shape[-2], x.shape[-1]).contiguous()
        cache = cos_sin_cache.float().contiguous()
        if conjugate:
            half = cache.shape[1] // 2
            cache = torch.cat((cache[:, :half], -cache[:, half:]), -1)
        lightop.rotary_embedding_fuse(pos.long() + seqlen_offset, work, None,
                                      x.shape[-1], cache, not interleaved)
        x.copy_(work.view_as(x))
        _count('rope.native')


def apply_rotary_rows(x, positions, cos, sin, rotary_dim, offset, is_neox, conjugate):
    out = x.clone(memory_format=torch.contiguous_format)
    apply_rotary(out[..., offset:offset + rotary_dim], torch.cat((cos, sin), -1),
                 positions=positions, interleaved=not is_neox, conjugate=conjugate)
    return out
