"""Torch/emu row rotary transform with the pinned BF16 stage rounding."""

import torch


def apply_rotary_rows(x, positions, cos, sin, rotary_dim, offset, is_neox, conjugate):
    half = rotary_dim // 2
    part = x[..., offset:offset + rotary_dim]
    if is_neox:
        x0, x1 = part[..., :half].float(), part[..., half:].float()
    else:
        x0, x1 = part[..., 0::2].float(), part[..., 1::2].float()
    c = cos.index_select(0, positions.long()).unsqueeze(1)
    s = sin.index_select(0, positions.long()).unsqueeze(1)
    if conjugate:
        y0 = torch.addcmul(x1 * s, x0, c)
        y1 = torch.addcmul(x1 * c, x0, -s)
    else:
        y0 = torch.addcmul(-(x1 * s), x0, c)
        y1 = torch.addcmul(x1 * c, x0, s)
    out = x.clone(memory_format=torch.contiguous_format)
    target = out[..., offset:offset + rotary_dim]
    if is_neox:
        target[..., :half] = y0.to(torch.bfloat16)
        target[..., half:] = y1.to(torch.bfloat16)
    else:
        target[..., 0::2] = y0.to(torch.bfloat16)
        target[..., 1::2] = y1.to(torch.bfloat16)
    return out
