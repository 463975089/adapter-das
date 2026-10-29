"""Packed Indexer bridges to the LightOp kernels observed in SGLang traces.

Q/K stay MXFP4 here. Only byte layout and logical-to-physical indices change.
The default boundary is BF16; the independent plugin may request FP32 logits.
Invalid lanes remain zero in either policy.
"""

import torch
import triton
import triton.language as tl


# Token/page counts vary at runtime; do not specialize the packing kernel on them.
@triton.jit(do_not_specialize=["N", "NP"])
def _pack_shifted_pages(K, SF, OUT, N, NP,
                        KS: tl.constexpr, SS: tl.constexpr,
                        PAGE: tl.constexpr, B: tl.constexpr):
    page = tl.program_id(0)
    bank = tl.program_id(1)
    x = tl.arange(0, B)
    # SOA page: PAGE * 64 payload bytes followed by PAGE * four scale bytes.
    payload = x < PAGE * 64
    slot = tl.where(payload, x // 64, (x - PAGE * 64) // 4)
    col = tl.where(payload, x % 64, (x - PAGE * 64) % 4)
    token = page * PAGE + slot + bank
    valid = (x < PAGE * 68) & (token < N)
    q = tl.load(K + token * KS + col, valid & payload, 0)
    sf = tl.load(SF + token * SS + col, valid & ~payload, 127)
    tl.store(OUT + (bank * NP + page) * PAGE * 68 + x,
             tl.where(payload, q, sf), x < PAGE * 68)


def pack_linear_kv(kv, block, banks):
    """O(banks * K) byte packing, with no per-query dequantized K expansion."""
    data, scales = kv
    n = data.shape[0]
    if (data.dtype != torch.int8 or data.shape[-1] != 64
            or data.numel() != n * 64 or scales.dtype != torch.int32
            or scales.numel() != n or block not in (8, 16)):
        raise ValueError('Expected packed D128 MXFP4 K and one int32 scale per token')
    data = data.reshape(n, 64).contiguous().view(torch.uint8)
    scales = scales.reshape(n).contiguous().view(torch.uint8)
    pages = triton.cdiv(n, block)
    cache = torch.empty((banks * pages, block, 1, 68), device=data.device,
                        dtype=torch.uint8)
    _pack_shifted_pages[(pages, banks)](
        data, scales, cache, n, pages, data.stride(0), 4, block,
        triton.next_power_of_2(block * 68))
    return cache, pages


def sparse_linear(q, kv, weights, metadata, capacity, block, *, output_dtype=torch.bfloat16):
    from lightop.attention import (
        get_paged_sparse_mqa_logits_metadata,
        sparse_fp8_fp4_paged_mqa_logits,
    )

    ks, ke = metadata['ks'].int(), metadata['ke'].int()
    unaligned = metadata['unaligned']
    cache, pages = pack_linear_kv(kv, block, block if unaligned else 1)
    base = ks // block
    bank = ks % block if unaligned else torch.zeros_like(ks)
    start = ks if unaligned else base * block
    lens = (ke - start).clamp_min(0).contiguous()
    blocks = (metadata['blocks'][:, :capacity] - base[:, None]).int().contiguous()
    table = (bank[:, None] * pages + base[:, None]
             + torch.arange(pages, device=ks.device, dtype=torch.int32)[None])
    # Group only rows with the same request start. Adjacent rows in each
    # request share a table; unique request starts also identify packed banks.
    indices = ks.contiguous()
    native = get_paged_sparse_mqa_logits_metadata(
        lens, table, indices, block, blocks, q[0].dtype, block)
    output = sparse_fp8_fp4_paged_mqa_logits(
        q, cache, weights.float().contiguous(), native, capacity, block,
        context_lens=lens, sparse_kv_block_indices=blocks, is_prefill=True)
    positions = (blocks[:, :, None] * block
                 + torch.arange(block, device=ks.device)).flatten(1) + start[:, None]
    valid = (positions >= ks[:, None]) & (positions < ke[:, None])
    return torch.where(valid, output, 0).to(output_dtype)


def paged_dense(q, cache, weights, lens, table, width, *, output_dtype=torch.bfloat16):
    from lightop.attention import (
        get_paged_mqa_logits_fp4_metadata,
        paged_mqa_logits_fp4,
    )

    lens = lens.reshape(-1).int().contiguous()
    sms = torch.cuda.get_device_properties(q[0].device).multi_processor_count
    schedule = get_paged_mqa_logits_fp4_metadata(lens, 64, sms)
    result = paged_mqa_logits_fp4(
        q[0], q[1], cache, weights.float().contiguous(), lens, table,
        schedule, width)
    valid = torch.arange(width, device=cache.device)[None] < lens[:, None]
    return torch.where(valid, result, 0).to(output_dtype)


def paged_sparse(q, cache, weights, metadata, capacity, block, *, output_dtype=torch.bfloat16):
    from lightop.attention import (
        get_paged_sparse_mqa_logits_metadata,
        sparse_fp8_fp4_paged_mqa_logits,
    )

    lens = metadata['lens'].reshape(-1).int().contiguous()
    blocks = metadata['blocks'][:, :capacity].int().contiguous()
    native = get_paged_sparse_mqa_logits_metadata(
        lens, metadata['block_table'], metadata['indices'], cache.shape[1],
        blocks, q[0].dtype, block)
    result = sparse_fp8_fp4_paged_mqa_logits(
        q, cache, weights.float().contiguous(), native, capacity, block,
        context_lens=lens, sparse_kv_block_indices=blocks)
    position = (blocks[:, :, None] * block
                + torch.arange(block, device=cache.device)).flatten(1)
    valid = (position >= 0) & (position < lens[:, None])
    return torch.where(valid, result, 0).to(output_dtype)
