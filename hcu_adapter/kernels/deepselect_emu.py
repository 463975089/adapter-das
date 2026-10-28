"""Pinned DeepSelect top-k contract using native HIP tensor operations."""

import torch


def _aligned(rows, columns, dtype, device):
    stride = (columns * dtype.itemsize + 31) // 32 * 32 // dtype.itemsize
    return torch.empty((rows, stride), dtype=dtype, device=device)[:, :columns]


def topk(input, topk, sorted=False, begin=None, end=None, indices_type=torch.int64,
         sorted_index=False, hint=None, output_idx=None, output_idx_offset=None,
         idx_oob_fill_value=2147483647, value_oob_fill_value=float("-inf"),
         return_value=True, abort_when_nan_found=True):
    if begin is not None or hint is not None:
        raise NotImplementedError("DeepSelect begin/hint are unsupported by the pinned API")
    if (input.ndim != 2 or not input.is_cuda or input.stride(1) != 1
            or input.dtype not in (torch.float32, torch.bfloat16)):
        raise ValueError("DeepSelect requires dense BF16/FP32 GPU rows")
    if indices_type not in (torch.int32, torch.int64) or topk <= 0:
        raise ValueError("DeepSelect requires positive K and int32/int64 indices")
    if sorted and input.dtype != torch.float32:
        raise NotImplementedError("Pinned DeepSelect sorted values support FP32 only")
    rows, columns = input.shape
    if end is None:
        length = torch.full((rows,), columns, dtype=torch.int32, device=input.device)
    else:
        if end.shape != (rows,) or end.dtype != torch.int32 or end.device != input.device:
            raise ValueError("DeepSelect end must be an int32 GPU vector")
        length = end
        torch._assert_async(((length >= 0) & (length <= columns)).all(), "DeepSelect end out of range")
    if output_idx is None:
        output_idx = _aligned(rows, topk, indices_type, input.device)
    elif (output_idx.shape != (rows, topk) or output_idx.dtype != indices_type
          or output_idx.device != input.device or output_idx.stride(-1) != 1):
        raise ValueError("DeepSelect output_idx shape/dtype/device mismatch")
    column = torch.arange(columns, device=input.device)
    valid = column[None] < length[:, None]
    long_row = length > topk
    nan = (torch.isnan(input) & valid).any(1) & long_row
    if abort_when_nan_found:
        torch._assert_async((~nan).all(), "DeepSelect: NaN in a long input row")
    # Stable sort keeps valid -inf entries ahead of padded -inf entries. Short
    # rows gather original positions, matching the pinned CUDA short-row path.
    values, indices = torch.sort(torch.where(valid, input, -float("inf")),
                                 dim=1, descending=True, stable=True)
    width = min(topk, columns)
    indices = indices[:, :width]
    short_indices = torch.arange(width, device=input.device).expand(rows, -1)
    indices = torch.where(long_row[:, None], indices, short_indices)
    values = input.gather(1, indices)
    selected_valid = torch.arange(width, device=input.device)[None] < length[:, None]
    if output_idx_offset is not None:
        if (output_idx_offset.shape != (rows,) or output_idx_offset.dtype != torch.int32
                or output_idx_offset.device != input.device):
            raise ValueError("DeepSelect output_idx_offset must be an int32 GPU vector")
        indices = indices + output_idx_offset[:, None]
    indices = torch.where(selected_valid, indices, idx_oob_fill_value)
    values = torch.where(selected_valid, values, value_oob_fill_value)
    if sorted_index:
        # Upstream's short-row shortcut emits valid IDs before padding;
        # idx_oob_fill_value is not a sortable candidate (it can be -1).
        order = torch.argsort(torch.where(selected_valid, indices, torch.iinfo(indices.dtype).max),
                              dim=1, stable=True)
        indices = indices.gather(1, order)
        values = values.gather(1, order)
    output_idx.fill_(idx_oob_fill_value)
    output_idx[:, :width].copy_(indices)
    if not abort_when_nan_found:
        output_idx[:, 0].copy_(torch.where(nan, 0x3F3F3F3F, output_idx[:, 0]))
    output_val = None
    if return_value:
        output_val = _aligned(rows, topk, input.dtype, input.device)
        output_val.fill_(value_oob_fill_value)
        output_val[:, :width].copy_(values)
    return output_val, output_idx
