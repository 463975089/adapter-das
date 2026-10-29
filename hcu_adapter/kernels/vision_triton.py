"""Triton implementations of the pinned vision contracts."""

import torch
import triton
import triton.language as tl

@triton.jit
def _normalize_rgb(VALUES, OUT, COUNT: tl.constexpr, MEAN, STD, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(VALUES + i, i < COUNT, other=0)
    # Explicit RN divisions match the upstream CPU LUT; reciprocal-multiply
    # scalar division can cross a BF16 midpoint on this DTK build.
    scaled = tl.div_rn(value, 255.)
    centered = scaled - MEAN
    normalized = tl.div_rn(centered, STD)
    tl.store(OUT + i, normalized, i < COUNT)


def rgb_pad_to_patches(raw, cu_raw, sizes, crops, cu_patches, mean, std, pad_color, output, *, patch_size):
    patches = output.shape[0]
    if not patches:
        return
    patch = torch.arange(patches, device=raw.device)
    image = torch.searchsorted(cu_patches[1:], patch.to(cu_patches.dtype), right=True)
    grid = crops.index_select(0, image)
    real = sizes.index_select(0, image)
    gap = grid * patch_size - real
    half = gap // 2
    padding = half + ((gap % 2 == 1) & (half % 2 == 1))
    local = patch - cu_patches.index_select(0, image)
    tile_y, tile_x = local // grid[:, 0], local % grid[:, 0]
    elem = torch.arange(3 * patch_size * patch_size, device=raw.device)
    channel = elem // (patch_size * patch_size)
    y = tile_y[:, None] * patch_size + (elem // patch_size % patch_size) - padding[:, 1, None]
    x = tile_x[:, None] * patch_size + (elem % patch_size) - padding[:, 0, None]
    valid = (y >= 0) & (y < real[:, 1, None]) & (x >= 0) & (x < real[:, 0, None])
    indices = cu_raw.index_select(0, image)[:, None] + (y.long() * real[:, 0, None] + x) * 3 + channel
    values = raw.index_select(0, indices.clamp(0, max(raw.numel() - 1, 0)).flatten()).view_as(indices).float()
    values = torch.where(valid, values, pad_color)
    # Keep the three FP32 operations separate, as the upstream CUDA kernel
    # explicitly disables FMA to match CPU ToTensor/Normalize rounding.
    values = values.contiguous()
    _normalize_rgb[(triton.cdiv(values.numel(), 512),)](values, output, values.numel(), mean, std, 512,
                                                        enable_fp_fusion=False)
