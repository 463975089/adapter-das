# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl


@triton.jit
def _layout(Cu, Sq, row, valid, N: tl.constexpr):
    lo = tl.full(row.shape, 0, tl.int32)
    hi = tl.full(row.shape, N, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(Cu + mid, valid & (mid < N), other=2147483647)
        right = end <= row
        active = lo < hi
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    seq = tl.minimum(lo, N - 1)
    end = tl.load(Cu + seq, valid, other=0)
    count = tl.load(Sq + seq, valid, other=0)
    return seq, row - (end - count)


@triton.jit(do_not_specialize=["NB"])
def _metadata(Sq, Sk, Cu, Blocks, Offsets, State, Pad, Pos, Valid, Gate, Fix,
              Context, Klen, CaLen, SwaLen, SwaSlot, Query, StateOut, PosC, CaSlot,
              N: tl.constexpr, T: tl.constexpr, NB, MAX_BLOCKS: tl.constexpr,
              WINDOW: tl.constexpr, RING: tl.constexpr, PAGE: tl.constexpr, FORCE: tl.constexpr,
              CA: tl.constexpr, DENSE: tl.constexpr, SPARSE: tl.constexpr,
              KS: tl.constexpr, CS: tl.constexpr, PS: tl.constexpr, SS: tl.constexpr,
              B: tl.constexpr):
    row = tl.program_id(0) * B + tl.arange(0, B)
    valid = row < T
    seq, j = _layout(Cu, Sq, row, valid, N)
    sk = tl.load(Sk + seq, valid, other=0).to(tl.int64)
    sq = tl.load(Sq + seq, valid, other=0).to(tl.int64)
    state = tl.load(State + seq, valid, other=0).to(tl.int64)
    pad = tl.load(Pad + seq, valid, other=1)
    off = tl.load(Offsets + seq, valid, other=0).to(tl.int64)
    pos = sk - sq + j
    tl.store(Pos + row, pos, valid)
    tl.store(Valid + row, ~pad.to(tl.int1), valid)
    tl.store(Gate + row, FORCE, valid)
    tl.store(Fix + row, False, valid)
    tl.store(Context + row, pos + 1, valid)
    for i in tl.static_range(len(SPARSE)):
        tl.store(Klen + i * KS + row, (pos + 1) // SPARSE[i], valid)
    for i in tl.static_range(len(DENSE)):
        tl.store(CaLen + i * CS + row * 2, row.to(tl.int64) * MAX_BLOCKS * (PAGE // DENSE[i]), valid)
        tl.store(CaLen + i * CS + row * 2 + 1, (pos + 1) // DENSE[i], valid)
    length = tl.minimum(pos + 1, WINDOW)
    start = pos - length + 1
    tl.store(SwaLen + row * 2, state * RING + start % RING, valid)
    tl.store(SwaLen + row * 2 + 1, length, valid)
    tl.store(SwaSlot + row, state * RING + pos % RING, valid)
    tl.store(Query + row, tl.where(pad, 998244353, seq), valid)
    tl.store(StateOut + row, state, valid)
    for i in tl.static_range(len(CA)):
        ratio = CA[i]
        cbs = PAGE // ratio
        index = pos // ratio
        boundary = (pos + 1) % ratio == 0
        tl.store(PosC + i * PS + row, tl.where(boundary, index * ratio, 0), valid)
        address = tl.minimum(tl.maximum(off + index // cbs, 0), NB - 1)
        block = tl.load(Blocks + address, valid, other=0).to(tl.int64)
        slot = block * cbs + index % cbs
        tl.store(CaSlot + i * SS + row, tl.where(boundary, slot, -1), valid)


def prepare_v4_decode_metadata(seqlens_q, seqlens_k, cu_seqlens_q, block_tables_in, block_tables_offsets,
                               state_block_idx_in, is_placeholder, positions, valid_token_mask, gate_force_random,
                               fix_routing_mask, context_lens, klen, kv_ca_index_offset_len,
                               kv_swa_index_offset_len, slot_mapping_store_swa, query_indices, state_block_idx_out,
                               positions_c, slot_mapping_store_ca, *, max_num_blocks, window_size, swa_block_size,
                               block_size, force_load_balance, ca_ratios, dense_ratios, sparse_ratios):
    total = positions.numel()
    if total:
        _metadata[(triton.cdiv(total, 128),)](
            seqlens_q, seqlens_k, cu_seqlens_q, block_tables_in, block_tables_offsets,
            state_block_idx_in, is_placeholder, positions, valid_token_mask, gate_force_random,
            fix_routing_mask, context_lens, klen, kv_ca_index_offset_len, kv_swa_index_offset_len,
            slot_mapping_store_swa, query_indices, state_block_idx_out, positions_c, slot_mapping_store_ca,
            seqlens_q.numel(), total, block_tables_in.numel(), max_num_blocks, window_size,
            swa_block_size, block_size, force_load_balance, tuple(ca_ratios), tuple(dense_ratios),
            tuple(sparse_ratios), klen.stride(0), kv_ca_index_offset_len.stride(0),
            positions_c.stride(0), slot_mapping_store_ca.stride(0), 128)


@triton.jit(do_not_specialize=["NB"])
def _blocks(Sq, Cu, Blocks, Offsets, Out, N: tl.constexpr, T: tl.constexpr,
            NB, WIDTH: tl.constexpr, STRIDE: tl.constexpr, B: tl.constexpr):
    index = tl.program_id(0) * B + tl.arange(0, B)
    row, col = index // WIDTH, index % WIDTH
    seq, _ = _layout(Cu, Sq, row, row < T, N)
    start = tl.load(Offsets + seq, row < T, other=0)
    end = tl.load(Offsets + seq + 1, row < T, other=0)
    valid = (row < T) & (col < end - start)
    value = tl.load(Blocks + tl.minimum(start + col, NB - 1), valid, other=0)
    tl.store(Out + row * STRIDE + col, value, valid)


def copy_block_tables(seqlens_q, cu_seqlens_q, block_tables_in, block_tables_offsets, block_tables_out, *, max_num_blocks):
    total = block_tables_out.shape[0]
    if total and max_num_blocks:
        _blocks[(triton.cdiv(total * max_num_blocks, 256),)](
            seqlens_q, cu_seqlens_q, block_tables_in, block_tables_offsets, block_tables_out,
            seqlens_q.numel(), total, block_tables_in.numel(), max_num_blocks, block_tables_out.stride(0), 256)


@triton.jit
def _ids(Fresh, Sample, Predict, Remap, Sq, Cu, Out,
         N: tl.constexpr, T: tl.constexpr, NS: tl.constexpr, NP: tl.constexpr,
         NEXT: tl.constexpr, FS: tl.constexpr, PS: tl.constexpr, MAXQ: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0) * B + tl.arange(0, B)
    valid = row < T
    seq, j = _layout(Cu, Sq, row, valid, N)
    pidx = tl.load(Remap + seq, valid, other=-1)
    if NS:
        sampled = tl.load(Sample + tl.minimum(tl.maximum(pidx, 0), NS - 1), valid, other=0)
    else:
        sampled = tl.full((B,), 0, tl.int32)
    if NP and NEXT:
        predicted = tl.load(Predict + tl.minimum(tl.maximum(pidx, 0), NP - 1) * PS + tl.minimum(tl.maximum(j - 1, 0), NEXT - 1), valid, other=0)
    else:
        predicted = tl.full((B,), 0, tl.int32)
    fresh = tl.load(Fresh + seq * FS + tl.minimum(j, MAXQ - 1), valid, other=0)
    tl.store(Out + row, tl.where(pidx >= 0, tl.where(j == 0, sampled, predicted), fresh), valid)


def gather_input_ids(fresh_tokens, prev_sampled, prev_predict, remap, seqlens_q, cu_seqlens_q, out_input_ids, *, max_seqlen_q):
    total = out_input_ids.numel()
    if total:
        np = prev_predict.shape[0]
        width = prev_predict.shape[1] if prev_predict.ndim == 2 else 0
        _ids[(triton.cdiv(total, 128),)](
            fresh_tokens, prev_sampled, prev_predict, remap, seqlens_q, cu_seqlens_q, out_input_ids,
            seqlens_q.numel(), total, prev_sampled.numel(), np, width, fresh_tokens.stride(0),
            prev_predict.stride(0), max_seqlen_q, 128)


@triton.jit
def _engram(Win, Fresh, Sample, Predict, Accepted, Remap, Lut, Sq, Cu, Out,
            N: tl.constexpr, T: tl.constexpr, NS: tl.constexpr, NP: tl.constexpr, NA: tl.constexpr,
            PW: tl.constexpr, PS: tl.constexpr, FW: tl.constexpr, FS: tl.constexpr,
            WW: tl.constexpr, WS: tl.constexpr, OS: tl.constexpr, NL: tl.constexpr,
            NEXT: tl.constexpr, GRAM: tl.constexpr, PAD: tl.constexpr, SENTINEL: tl.constexpr,
            B: tl.constexpr):
    row = tl.program_id(0) * B + tl.arange(0, B)
    valid = row < T
    seq, j = _layout(Cu, Sq, row, valid, N)
    pidx = tl.load(Remap + seq, valid, other=-1)
    safe = tl.maximum(pidx, 0)
    if NA:
        accepted = tl.load(Accepted + tl.minimum(safe, NA - 1), valid, other=0)
    else:
        accepted = tl.full((B,), 0, tl.int32)
    base = NEXT + GRAM - 1 - tl.where(pidx >= 0, NEXT - accepted, NEXT)
    if NS:
        sampled = tl.load(Sample + tl.minimum(safe, NS - 1), valid, other=0)
    else:
        sampled = tl.full((B,), 0, tl.int32)
    hit = tl.full((B,), False, tl.int1)
    for r in tl.static_range(GRAM):
        c = j - r
        if NP and PW:
            predicted = tl.load(Predict + tl.minimum(safe, NP - 1) * PS + tl.minimum(tl.maximum(c - 1, 0), PW - 1), valid, other=0)
        else:
            predicted = tl.full((B,), 0, tl.int32)
        raw = tl.where(c == 0, sampled, predicted)
        previous = tl.load(Lut + tl.minimum(tl.maximum(raw, 0), NL - 1), valid, other=0)
        fresh = tl.load(Fresh + seq * FS + tl.minimum(tl.maximum(c, 0), FW - 1), valid, other=0)
        window = tl.load(Win + seq * WS + tl.minimum(tl.maximum(base + c, 0), WW - 1), valid, other=0)
        token = tl.where(c >= 0, tl.where(pidx >= 0, previous, fresh), window)
        hit = hit | (token == SENTINEL)
        tl.store(Out + row * OS + r, tl.where(hit, PAD, token), valid)


def gather_engram_tokens(win_compressed, curr_fresh, prev_sampled, prev_predict, prev_num_accepted, remap,
                         lookup_table, seqlens_q, cu_seqlens_q, out, *, max_seqlen_q, max_ngram_size,
                         engram_compressed_pad_token, image_sentinel):
    total = out.shape[0]
    if total:
        _engram[(triton.cdiv(total, 128),)](
            win_compressed, curr_fresh, prev_sampled, prev_predict, prev_num_accepted, remap,
            lookup_table, seqlens_q, cu_seqlens_q, out, seqlens_q.numel(), total,
            prev_sampled.numel(), prev_predict.shape[0], prev_num_accepted.numel(),
            prev_predict.shape[1] if prev_predict.ndim == 2 else 0, prev_predict.stride(0),
            curr_fresh.shape[1], curr_fresh.stride(0), win_compressed.shape[1], win_compressed.stride(0),
            out.stride(0), lookup_table.numel(), max_seqlen_q - 1, max_ngram_size,
            engram_compressed_pad_token, image_sentinel, 128)


@triton.jit
def _lengths(Sk, Sq, Remap, Accepted, N: tl.constexpr, NA: tl.constexpr, NEXT: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < N
    pidx = tl.load(Remap + i, valid, other=-1)
    if NA:
        accepted = tl.load(Accepted + tl.minimum(tl.maximum(pidx, 0), NA - 1), valid, other=0)
    else:
        accepted = tl.full((B,), 0, tl.int32)
    sk = tl.load(Sk + i, valid, other=0)
    sq = tl.load(Sq + i, valid, other=0)
    tl.store(Sk + i, sk + sq - tl.where(pidx >= 0, NEXT - accepted, 0), valid)


def compute_seqlens_k(seqlens_k, seqlens_q, remap, prev_num_accepted, *, nextn):
    if seqlens_k.numel():
        _lengths[(triton.cdiv(seqlens_k.numel(), 128),)](
            seqlens_k, seqlens_q, remap, prev_num_accepted, seqlens_k.numel(), prev_num_accepted.numel(), nextn, 128)
