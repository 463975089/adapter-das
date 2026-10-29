"""deepselect lightop implementation; upstream rounding/layout contracts are unchanged."""

import torch

from .lightop_usage import _count


def topk(input, topk, sorted=False, begin=None, end=None, indices_type=torch.int64,
         sorted_index=False, hint=None, output_idx=None, output_idx_offset=None,
         idx_oob_fill_value=2147483647, value_oob_fill_value=float('-inf'),
         return_value=True, abort_when_nan_found=True):
    from .deepselect_emu import _aligned, topk as compatibility_topk
    from lightop.fuse_topk_transform import topk_transform_ragged_v2

    if (topk not in (512, 1024, 2048) or input.numel() == 0 or begin is not None or hint is not None
            or not abort_when_nan_found):
        _count('topk.compatibility')
        return compatibility_topk(input, topk, sorted, begin, end, indices_type,
                                  sorted_index, hint, output_idx, output_idx_offset,
                                  idx_oob_fill_value, value_oob_fill_value,
                                  return_value, abort_when_nan_found)
    if (input.ndim != 2 or not input.is_cuda or input.stride(-1) != 1
            or input.dtype not in (torch.float32, torch.bfloat16)
            or indices_type not in (torch.int32, torch.int64)):
        raise ValueError('Unsupported native DeepSelect input or indices dtype')
    if sorted and input.dtype != torch.float32:
        raise NotImplementedError('Pinned sorted values require FP32')
    rows, width = input.shape
    lengths = (torch.full((rows,), width, dtype=torch.int32, device=input.device)
               if end is None else end)
    if lengths.shape != (rows,) or lengths.dtype != torch.int32 or lengths.device != input.device:
        raise ValueError('DeepSelect end must be an int32 device vector')
    column = torch.arange(width, device=input.device)
    torch._assert_async(((lengths >= 0) & (lengths <= width)).all(), 'DeepSelect end out of range')
    nan = (torch.isnan(input) & (column[None] < lengths[:, None])).any(1) & (lengths > topk)
    torch._assert_async((~nan).all(), 'DeepSelect: NaN in a long input row')
    indices = torch.empty((rows, topk), device=input.device, dtype=torch.int32)
    zeros = torch.zeros(rows, device=input.device, dtype=torch.int32)
    # The native API excludes NaNs; the pinned short-row path returns every ID
    # without comparing values, so sanitize only those rows for the selection.
    scores = torch.where((lengths <= topk)[:, None], 0., input.float()).contiguous()
    topk_transform_ragged_v2(scores, lengths.contiguous(), out_offsets=zeros, out_indices=indices)
    valid = indices >= 0
    safe = indices.clamp(min=0, max=max(0, width - 1)).long()
    values = input.gather(1, safe)
    # Native short rows can be unordered; DeepSelect requires original order.
    order = torch.argsort(torch.where(valid, safe, width), dim=1, stable=True)
    safe, valid, values = (tensor.gather(1, order) for tensor in (safe, valid, values))
    if sorted:
        ranks = torch.argsort(torch.where(valid, values, -float('inf')), dim=1,
                              descending=True, stable=True)
        ranks = torch.where((lengths > topk)[:, None], ranks,
                            torch.arange(topk, device=input.device).expand(rows, -1))
        safe, valid, values = (tensor.gather(1, ranks) for tensor in (safe, valid, values))
    if output_idx_offset is not None:
        if (output_idx_offset.shape != (rows,) or output_idx_offset.dtype != torch.int32
                or output_idx_offset.device != input.device):
            raise ValueError('DeepSelect output offsets require an int32 device vector')
        safe = safe + output_idx_offset[:, None]
    mapped = torch.where(valid, safe, idx_oob_fill_value)
    values = torch.where(valid, values, value_oob_fill_value)
    if sorted_index:
        # DeepSelect interface.py / common_parts.cuh keep short-row padding
        # at the end, including a negative sentinel. Sorting the sentinel
        # itself moves -1 into the valid prefix consumed by sparse planners.
        order = torch.argsort(torch.where(valid, mapped, torch.iinfo(mapped.dtype).max),
                              dim=1, stable=True)
        mapped = mapped.gather(1, order)
        values = values.gather(1, order)
    if output_idx is None:
        output_idx = _aligned(rows, topk, indices_type, input.device)
    elif (output_idx.shape != (rows, topk) or output_idx.dtype != indices_type
          or output_idx.device != input.device or output_idx.stride(-1) != 1):
        raise ValueError('DeepSelect output indices mismatch')
    output_idx.copy_(mapped)
    output_val = None
    if return_value:
        output_val = _aligned(rows, topk, input.dtype, input.device)
        output_val.copy_(values)
    _count('topk.native')
    return output_val, output_idx
