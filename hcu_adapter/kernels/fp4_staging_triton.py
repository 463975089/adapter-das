# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Copy variable FP4 page planes directly into canonical 64-record pages.

The native cache is page-local payload then UE8M0 scale bytes, rather than
interleaved 68-byte records. Invalid logical positions reproduce the existing
native gather's slot-zero bytes. No quantization or score arithmetic changes.
"""
from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _stage_fp4_pages(Cache, Table, Lengths, Pages, OutTable,
                     CACHE_STRIDE: tl.constexpr, TABLE_STRIDE: tl.constexpr,
                     TABLE_INNER: tl.constexpr, PAGE_ROW_STRIDE: tl.constexpr,
                     PAGE_STRIDE: tl.constexpr, OUT_ROW_STRIDE: tl.constexpr,
                     OUT_INNER: tl.constexpr, RECORDS: tl.constexpr,
                     PAGES_PER_ROW: tl.constexpr, BLOCK: tl.constexpr):
    # Long-context pools exceed 2 GiB even when each row fits in int32.
    row = tl.program_id(0).to(tl.int64)
    page = tl.program_id(1).to(tl.int64)
    offset = tl.arange(0, BLOCK)
    payload = offset < 4096
    record = tl.where(payload, offset // 64, (offset - 4096) // 4)
    byte = tl.where(payload, offset % 64, (offset - 4096) % 4)
    logical = page * 64 + record
    length = tl.load(Lengths + row)
    valid = (logical < length) & (offset < 4352)
    physical_page = tl.load(Table + row * TABLE_STRIDE + logical // RECORDS * TABLE_INNER,
                            mask=valid, other=0).to(tl.int64)
    physical_record = tl.where(valid, logical % RECORDS, 0)
    inside = tl.where(payload, physical_record * 64 + byte,
                      RECORDS * 64 + physical_record * 4 + byte)
    values = tl.load(Cache + physical_page * CACHE_STRIDE + inside,
                     mask=offset < 4352, other=0)
    tl.store(Pages + row * PAGE_ROW_STRIDE + page * PAGE_STRIDE + offset,
              values, mask=offset < 4352)
    tl.store(OutTable + row * OUT_ROW_STRIDE + page * OUT_INNER,
              row * PAGES_PER_ROW + page)


@triton.jit
def _stage_fp4_live_pages(Cache, Table, Lengths, Pages, OutTable,
                          CACHE_STRIDE: tl.constexpr, TABLE_STRIDE: tl.constexpr,
                          TABLE_INNER: tl.constexpr, PAGE_ROW_STRIDE: tl.constexpr,
                          PAGE_STRIDE: tl.constexpr, OUT_ROW_STRIDE: tl.constexpr,
                          OUT_INNER: tl.constexpr, RECORDS: tl.constexpr,
                          PAGES_PER_ROW: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    length = tl.minimum(tl.load(Lengths + row), PAGES_PER_ROW * 64)
    # The allocation retains its maximum pitch, but replay visits only pages
    # named by the current device lengths. No host read or captured length.
    for page in range(tl.program_id(1), tl.cdiv(length, 64), tl.num_programs(1)):
        page64 = page.to(tl.int64)
        offset = tl.arange(0, BLOCK)
        payload = offset < 4096
        record = tl.where(payload, offset // 64, (offset - 4096) // 4)
        byte = tl.where(payload, offset % 64, (offset - 4096) % 4)
        logical = page64 * 64 + record
        valid = (logical < length) & (offset < 4352)
        physical_page = tl.load(
            Table + row * TABLE_STRIDE + logical // RECORDS * TABLE_INNER,
            mask=valid, other=0).to(tl.int64)
        physical_record = tl.where(valid, logical % RECORDS, 0)
        inside = tl.where(payload, physical_record * 64 + byte,
                          RECORDS * 64 + physical_record * 4 + byte)
        values = tl.load(Cache + physical_page * CACHE_STRIDE + inside,
                         mask=offset < 4352, other=0)
        tl.store(Pages + row * PAGE_ROW_STRIDE + page64 * PAGE_STRIDE + offset,
                  values, mask=offset < 4352)
        tl.store(OutTable + row * OUT_ROW_STRIDE + page64 * OUT_INNER,
                  row * PAGES_PER_ROW + page)


@lru_cache(maxsize=None)
def _supported_device(index):
    props = torch.cuda.get_device_properties(index)
    return (getattr(props, 'gcnArchName', '').split(':')[0] == 'gfx938'
            and props.multi_processor_count == 64)


def supports_staging_device(device):
    """Share the device decision with startup workspace accounting."""
    device = torch.device(device)
    if device.type != 'cuda' or torch.version.hip is None:
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _supported_device(index)


def stage_decode_pages(cache, table, lengths, pages, out_table, *, live_only=False):
    """Fill caller-owned pages/table; return False outside the tested ABI."""
    tensors = (cache, table, lengths, pages, out_table)
    if (not cache.is_cuda or torch.version.hip is None
            or any(t.device != cache.device for t in tensors)
            or cache.ndim != 3 or cache.dtype != torch.uint8
            or cache.shape[0] == 0 or cache.shape[1] not in (32, 128)
            or cache.shape[2] != 68 or cache.stride(1) != 68 or cache.stride(2) != 1
            or cache.stride(0) < cache.shape[1] * 68
            or table.ndim != 2 or table.dtype != torch.int32
            or lengths.ndim != 1 or lengths.dtype != torch.int32 or not lengths.is_contiguous()
            or pages.ndim != 3 or pages.dtype != torch.uint8 or pages.shape[2] != 4352
            or pages.stride(2) != 1 or out_table.ndim != 2 or out_table.dtype != torch.int32
            or table.shape[0] != lengths.shape[0] or pages.shape[0] != lengths.shape[0]
            or out_table.shape != pages.shape[:2]
            or table.shape[1] * cache.shape[1] < pages.shape[1] * 64):
        return False
    if not supports_staging_device(cache.device):
        return False
    if pages.shape[0] and pages.shape[1]:
        if live_only:
            sms = torch.cuda.get_device_properties(cache.device).multi_processor_count
            grid = (pages.shape[0], min(pages.shape[1], triton.cdiv(sms, pages.shape[0])))
            kernel = _stage_fp4_live_pages
        else:
            grid, kernel = (pages.shape[0], pages.shape[1]), _stage_fp4_pages
        kernel[grid](
            cache, table, lengths, pages, out_table,
            cache.stride(0), table.stride(0), table.stride(1),
            pages.stride(0), pages.stride(1), out_table.stride(0), out_table.stride(1),
            cache.shape[1], pages.shape[1], 8192, num_warps=4)
    return True
