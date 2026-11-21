"""Byte-exact MLA HiCache transfers over device and pinned-host addresses."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_rows(
    src, dst, src_indices, dst_indices,
    ROW_BYTES: tl.constexpr, SRC_TABLE: tl.constexpr, DST_TABLE: tl.constexpr,
    SRC_STRIDE: tl.constexpr, DST_STRIDE: tl.constexpr,
    SRC_OFFSET: tl.constexpr, DST_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    layer = tl.program_id(1)
    tile = tl.program_id(2)
    s = tl.load(src + layer) if SRC_TABLE else src
    d = tl.load(dst + layer) if DST_TABLE else dst
    s = s.to(tl.pointer_type(tl.uint8))
    d = d.to(tl.pointer_type(tl.uint8))
    si = tl.load(src_indices + row).to(tl.int64)
    di = tl.load(dst_indices + row).to(tl.int64)
    offset = tile * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(
        s + si * SRC_STRIDE + SRC_OFFSET * layer + offset,
        mask=offset < ROW_BYTES, other=0,
    )
    tl.store(
        d + di * DST_STRIDE + DST_OFFSET * layer + offset,
        value, mask=offset < ROW_BYTES,
    )


def copy_rows(
    src, dst, src_indices, dst_indices, row_bytes, num_layers=1,
    *, src_table=False, dst_table=False, src_stride=None, dst_stride=None,
    src_offset=0, dst_offset=0,
):
    if src_indices.device.type != "cuda" or dst_indices.device != src_indices.device:
        raise ValueError("HiCache kernel indices must share a CUDA device")
    if src_indices.numel() != dst_indices.numel():
        raise ValueError("HiCache source and destination row counts differ")
    src_indices = src_indices.contiguous()
    dst_indices = dst_indices.contiguous()
    for pointers, is_table in ((src, src_table), (dst, dst_table)):
        if is_table and (
            pointers.device != src_indices.device
            or pointers.dtype not in (torch.int64, torch.uint64)
            or not pointers.is_contiguous()
            or pointers.numel() < num_layers
        ):
            raise ValueError("HiCache pointer tables must be contiguous device int64")
    if row_bytes < 0 or num_layers < 0:
        raise ValueError("HiCache row bytes and layer count must be nonnegative")
    if not src_indices.numel() or not row_bytes or not num_layers:
        return
    with torch.cuda.device(src_indices.device):
        _copy_rows[(src_indices.numel(), num_layers, triton.cdiv(row_bytes, 4096))](
            src, dst, src_indices, dst_indices,
            row_bytes, src_table, dst_table,
            row_bytes if src_stride is None else src_stride,
            row_bytes if dst_stride is None else dst_stride,
            src_offset, dst_offset, 4096,
        )
