"""torch paged-MQA logits for the v0.5.21 DSA indexer (is_cuda semantic branch).

Upstream call site (dsa_indexer.py, ``deep_gemm.fp8_paged_mqa_logits`` arm):
    q      (N, H, 128) fp8e4m3 — per-head q scales are pre-folded by the
           caller into ``weights`` (N_CTX heads x 1 fp32); the per-dim
           quant is hadamard-flattened so a straight dot is the contract.
    kv     flat cache viewed (T, block_kv=64, 1, 132): 128 fp8 bytes + fp32
           scale per token slot.
    ctx    int32 context lens, 2D (N, 1) after the (N_total,1) normalisation.
    bt     int32 page table (N, L) — page indices; token t = page*64 + slot.
    sched  get_paged_mqa_logits_metadata(ctx, 64, num_sms) opaquely — not
           interpreted by the kernel contract, only scheduling.
    out    fp32 logits (N, max_len); clean_logits=False leaves stale values
           outside each row's context (caller reads only valid windows and
           later top-k masks), True zeroes them.

The is_hip arm routes to aiter, which HCU never takes under the is_cuda
platform semantic; this emulation keeps identical numerics paths to the
proven HCU emu (fp8->bf16 exact dequant, bf16 GEMM widening in fp32, fp32
per-token scale fold), tiling over (rows, tokens) to stay capture-safe.
"""

import os

import torch

_HEAD_PAD = 132


def _fp8_view_dtype():
    try:
        from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

        return torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
    except Exception:
        return torch.float8_e4m3fn


def _as_view(t, dtype):
    if t.dtype in (torch.uint8, torch.int8):
        return t.contiguous().view(dtype)
    return t


def fp8_paged_mqa_logits(
    q_fp8,
    kv_cache_fp8,
    weights,
    context_lens,
    block_table,
    schedule_metadata=None,
    max_context_len=None,
    clean_logits=False,
    **_ignored,
):
    if max_context_len is None:
        max_context_len = int(block_table.shape[1]) * 64
    N = q_fp8.shape[0] if q_fp8.dim() == 3 else q_fp8.shape[0] * q_fp8.shape[1]
    if q_fp8.dim() == 4:
        q_flat = q_fp8.reshape(-1, q_fp8.shape[-2], q_fp8.shape[-1])
        H, D = q_flat.shape[-2], q_fp8.shape[-1]
        if q_fp8.dtype in (torch.uint8, torch.int8):
            q_flat = q_flat.contiguous().view(*q_flat.shape[:-1], D)
        # (B, next_n, H, D) -> rows are B*next_n; decode passes next_n=1 views
        q_b = _as_view(q_flat, _fp8_view_dtype()).to(torch.bfloat16).contiguous()
        N, H, D = q_b.shape
    else:
        q_b = _as_view(q_fp8, _fp8_view_dtype()).to(torch.bfloat16).contiguous()
        N, H, D = q_b.shape

    kv = kv_cache_fp8.reshape(-1, kv_cache_fp8.shape[-1]).contiguous()
    if kv.shape[-1] == 132:
        k_b = _as_view(kv[:, :128], _fp8_view_dtype()).to(torch.bfloat16).contiguous()
        k_sf = kv[:, 128:].view(torch.float32).reshape(-1)
    else:
        # bf16/fp16 packed cache (P, S, 1, D): no per-token scale,
        # reshape any trailing head-dim padding back to tokens x D.
        kb16 = kv_cache_fp8.contiguous()
        k_b = kb16.to(torch.bfloat16).reshape(-1, kb16.shape[-1])
        k_sf = torch.ones(k_b.shape[0], device=kb16.device)

    if weights.dim() == 3:
        weights = weights.reshape(weights.shape[0], -1)
    w = weights.reshape(q_b.shape[0], H).to(torch.float32)
    ctx = context_lens.to(torch.int64).reshape(-1)
    bt = block_table.to(torch.int64).contiguous()
    PS = kv_cache_fp8.shape[1] if kv_cache_fp8.dim() == 4 else 1

    device = q_b.device
    logits = torch.zeros((N, int(max_context_len)), dtype=torch.float32, device=device)

    qt = int(os.environ.get("HCU_DSA_MQA_QTILE", "64"))
    tt = int(os.environ.get("HCU_DSA_MQA_TTILE", "2048"))
    ar = torch.arange(tt, device=device)
    for i0 in range(0, N, qt):
        i1 = min(N, i0 + qt)
        qb = q_b[i0:i1]  # (Q, H, D)
        bt_rows = bt[i0:i1]
        seq_rows = ctx[i0:i1]
        for t0 in range(0, int(max_context_len), tt):
            t1 = min(int(max_context_len), t0 + tt)
            pages = bt_rows[:, t0 // PS : (t1 + PS - 1) // PS]  # (Q, P)
            toks = pages.unsqueeze(2) * PS + torch.arange(PS, device=device)
            toks = toks.reshape(pages.shape[0], -1)[:, : t1 - t0]  # (Q, T)
            kg = k_b[toks]  # (Q, T, D)
            sc = k_sf[toks]  # (Q, T)
            s = torch.matmul(qb, kg.transpose(1, 2)).float()  # (Q, H, T)
            row = (s * w[i0:i1].unsqueeze(-1)).sum(dim=1) * sc  # (Q, T)
            valid = ar[: t1 - t0].unsqueeze(0) + t0 < seq_rows.unsqueeze(1)
            logits[i0:i1, t0:t1] = torch.where(valid, row, 0.0)
    return logits


def get_num_sms():
    try:
        return torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    except Exception:
        return 120


def get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms):
    """Opaque scheduling blob upstream only feeds back to the same kernel.

    Our torch path ignores it; return a small int tensor with the inputs
    threaded through so callers that assert on shape have something sane.
    """
    rows = context_lens.reshape(-1)
    meta = torch.zeros(
        (rows.numel() + 1, 2), dtype=torch.int32, device=rows.device
    )
    return meta


fp8_paged_mqa_logits.__name__ = "fp8_paged_mqa_logits"
