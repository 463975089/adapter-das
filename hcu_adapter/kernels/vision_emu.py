# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Torch/emu implementations of the pinned vision contracts."""

import torch

def replace_input_embeds(embeds, token_ids, images, image_token_id, offsets, pad_ids, pad_embeds):
    if images.shape[0]:
        gathered = images.index_select(0, (offsets - 1).clamp(min=0).long())
        embeds.copy_(torch.where((token_ids == image_token_id)[:, None], gathered, embeds))
    else:
        torch._assert_async((token_ids != image_token_id).all(), "Image token has no image embedding")
    for i in range(pad_ids.numel()):
        embeds.copy_(torch.where((token_ids == pad_ids[i])[:, None], pad_embeds[i], embeds))


def batch_downsample(features, image_ids, crops, cu_down, cu_source, embeds, pooled, *, ratio, do_downsample):
    n = image_ids.numel()
    if not n:
        return
    image = image_ids.long()
    grid = crops.index_select(0, image)
    local = torch.arange(n, device=features.device) - cu_down.index_select(0, image)
    width = (grid[:, 0] + ratio - 1) // ratio
    offsets = torch.arange(ratio * ratio, device=features.device)
    y = (local // width)[:, None] * ratio + offsets // ratio
    x = (local % width)[:, None] * ratio + offsets % ratio
    valid = (y < grid[:, 1, None]) & (x < grid[:, 0, None])
    indices = cu_source.index_select(0, image)[:, None] + y * grid[:, 0, None] + x
    values = features.index_select(0, indices.clamp(0, features.shape[0] - 1).flatten().long())
    values = values.view(n, ratio * ratio, features.shape[1])
    values = torch.where(valid[:, :, None], values, 0)
    if do_downsample:
        pooled.copy_(values.float().sum(1) / valid.sum(1)[:, None])
    else:
        embeds.copy_(values.permute(0, 2, 1).reshape_as(embeds))
