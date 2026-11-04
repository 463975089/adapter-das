# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""fa3 varlen prefill face (packed q + ragged dense K/V) — triton kernel
(tier-③ triton-first, HCU_FA_IMPL=triton).

sglang's fa3 backend routes extend/target-verify (and the kvcache-facade
varlen mode) through ``flash_attn_varlen_func``: packed q ``(M, H, D)``,
packed ragged k/v ``(N, Hk, D)``, ``cu_seqlens_q/k``, per-sequence
bottom-right causal — query row j of sequence b sits at logical position
``Lk_b - Lq_b + j`` (local kv frame, matching the emu's per-sequence
chunk math exactly). The paged faces of the same backend live in
``paged_attn_triton.py``; this module covers the one face still on the
torch emu (python loop + host syncs, unusable under prefill graph
capture).

HCU constraints honored (decode-kernel finds #1-#5, see models.md):
- fp32 accumulation end-to-end; ``tl.dot(..., input_precision="ieee")``
- no ``tl.minimum`` (broken on the HCU triton backend) — compare+where
- token ids widened to int64 before address math (a real prefill concat
  reaches 100k+ tokens × Hk × D and breaks int32 long before the token
  count runs out)
- fully-masked rows/chunks guarded by ``m_safe`` so exp never sees -inf
  offsets; the degenerate empty-chunk case folds into a zero-trip loop
  via ``hi`` clamped to 0 (a scalar ``if: return`` on a device value is
  avoided deliberately)
- masks + tl reductions only; no warp-size assumptions

Launch-config census (BW1101, tools/_probe_fa_varlen_sweep{,2}.py): the
shipped defaults BM=64 / BN=64 / W=4 are the optimum (or the only fast
arm) on every shape probed — serving bs8×Lq512×4K ctx 15.3ms vs emu 57ms,
wide-ctx 2.8ms. W=2 anywhere and BM<64 run 3-10× cliffs (HCU triton
waveform pathology — do not "fix" the defaults without a BW1102 rerun);
BN≥256 exhausts shared memory at D=128. Known slow arm: fp8 KV on narrow
extend windows (~10ms vs emu 1.8ms — suspected e4m3 software-cvt path;
numerics pass, direction awaits BW1102).

Graph-safety contract of the wrapper: the grid derives from
``max_seqlen_q`` (shape-static) and ``B = cu_seqlens_q.numel()-1`` only —
never from tensor *content*. When ``max_seqlen_q`` is omitted the wrapper
reads it on the host (a sync): allowed eager-only, refused under capture
with a pointed error (the emu has the same lazy-host contract minus the
refusal).

Contract limits (wrapper asserts / routing falls back to the emu):
- ``D_v == D`` and D a power of two ≤ 256 (absorbed-MLA D=576 faces stay
  on the emu); q dtype fp16/bf16 (fp8-q is a kvcache-face thing);
  ``softcap``/``qv``/``sinks``/``q_descale`` unsupported (emu keeps them);
  fp8 **KV** (e4m3 pool bytes + per-head ``k_descale``/``v_descale``)
  is supported.
- a zero-Lk sequence raising in the emu writes ``o=0, lse=-inf`` here
  (a kernel cannot host-raise under capture); pinned by the sentry test.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fa_varlen_kernel(
    q_ptr,           # [M, H, D] packed
    k_ptr,           # [N, HKV, D] packed ragged (dense, bf16/fp16/e4m3)
    v_ptr,           # [N, HKV, D]
    cu_q_ptr,        # [B+1] int32
    cu_k_ptr,        # [B+1] int32
    kd_ptr,          # [HKV] fp32 or unused
    vd_ptr,          # [HKV] fp32 or unused
    o_ptr,           # [M, H, D] (dtype = q's compute dtype)
    lse_ptr,         # [M, H] fp32
    scale,
    lef_win,         # window half left: >=0 effective, 1<<30 = unbounded
    rig_win,         # window half right: >=0 effective, 1<<30 = unbounded
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,         # H // HKV
    CAUSAL: tl.constexpr,
    HAS_DS: tl.constexpr,    # descale pointers valid
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    kv_base = tl.load(cu_q_ptr + pid_b)          # q token base of sequence
    kb_base = tl.load(cu_k_ptr + pid_b)          # k token base of sequence
    Lq = tl.load(cu_q_ptr + pid_b + 1) - kv_base
    Lk = tl.load(cu_k_ptr + pid_b + 1) - kb_base

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = offs_m < Lq
    # dead q-chunk (beyond this sequence's window inside the maxL grid):
    # clamp the loop span to zero trips instead of a scalar return — the
    # stores below are row_ok-masked either way.
    m_hi = pid_m * BLOCK_M + BLOCK_M
    m_hi = tl.where(m_hi <= Lq, m_hi, Lq)        # no tl.minimum (HCU)

    kv_h = pid_h // G
    d = tl.arange(0, BLOCK_D)
    d_ok = d < D

    q_base = (kv_base.to(tl.int64) + offs_m.to(tl.int64)) * H + pid_h
    q = tl.load(q_ptr + q_base[:, None] * D + d[None, :],
                mask=row_ok[:, None] & d_ok[None, :], other=0.0)
    q = q.to(tl.float32)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # bottom-right causal: local row j sees keys j_local <= Lk - Lq + j;
    # upper loop bound from the chunk's widest live row (+ right window).
    # Dead q-chunks (beyond Lq inside the maxL grid) collapse to zero
    # trips — stores are row_ok-masked anyway.
    bound = Lk - Lq + offs_m                     # per-row key bound
    hi = Lk - Lq + m_hi if CAUSAL else Lk
    hi = tl.where(hi < Lk, hi, Lk)               # clamp: no tl.minimum
    hi = tl.where(pid_m * BLOCK_M < Lq, hi, 0)
    if HAS_DS:
        kd = tl.load(kd_ptr + kv_h)
        vd_s = tl.load(vd_ptr + kv_h)
    for t0 in range(0, hi, BLOCK_N):
        offs_n = t0 + tl.arange(0, BLOCK_N)
        n_ok = offs_n < Lk
        tok = (kb_base.to(tl.int64) + offs_n.to(tl.int64)) * HKV + kv_h
        base = (tok * D)[:, None] + d[None, :]   # [N,1]+[1,D] broadcast
        k = tl.load(k_ptr + base, mask=n_ok[:, None] & d_ok[None, :],
                    other=0.0).to(tl.float32)
        if HAS_DS:
            k = k * kd
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        keep = n_ok[None, :]
        if CAUSAL:
            keep = keep & (offs_n[None, :] <= bound[:, None])
        qpos = (Lk - Lq) + offs_m                # row's logical position
        rel = offs_n[None, :] - qpos[:, None]
        keep = keep & (rel >= -lef_win) & (rel <= rig_win)
        s = tl.where(keep, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        # fully-masked rows (zero-Lq tails, zero-Lk sequences) must not
        # poison exp — the reference kernel's m_safe guard
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])
        v = tl.load(v_ptr + base, mask=n_ok[:, None] & d_ok[None, :],
                    other=0.0).to(tl.float32)
        if HAS_DS:
            v = v * vd_s
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_new

    denom = tl.maximum(l_i, 1e-10)
    o = acc / denom[:, None]
    tl.store(o_ptr + q_base[:, None] * D + d[None, :], o,
             mask=row_ok[:, None] & d_ok[None, :])
    tl.store(lse_ptr + q_base, m_i + tl.log(denom), mask=row_ok)


def flash_attn_varlen_triton(
    q,               # [M, H, D] packed (bf16/fp16)
    k,               # [N, Hkv, D] packed ragged
    v,               # [N, Hkv, D]
    cu_seqlens_q,    # [B+1] int32
    cu_seqlens_k,    # [B+1] int32
    max_seqlen_q=None,   # static window span; None = host-read (eager only)
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
    k_descale=None,      # per-KV-head fp32 scales (e4m3 pool), optional
    v_descale=None,
    return_softmax_lse=False,
    out=None,
    max_seqlen_k=None,   # accepted for contract alignment (unused)
    BLOCK_M=64,
    BLOCK_N=64,
    num_warps=4,
):
    """Graph-safe varlen prefill attention, bottom-right causal per
    sequence. Returns o [M, H, D] in q's dtype (optionally (o, lse)
    [M, H] fp32). See the module docstring for contract limits."""
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            "fa_varlen_triton: q/k/v must be 3D packed, got "
            f"{q.dim()}/{k.dim()}/{v.dim()}")
    M, H, D = q.shape
    N, Hk, Dv = v.shape
    if k.shape != (N, Hk, D):
        raise NotImplementedError(
            f"fa_varlen_triton: needs D_v == D (got D={D}, Dv={Dv}) -- MLA "
            "absorbed faces stay on the emu")
    if D > 256 or (D & (D - 1)) != 0:
        raise NotImplementedError(
            f"fa_varlen_triton: D={D} not a power of two <= 256")
    if H % Hk:
        raise ValueError(f"fa_varlen_triton: GQA {H} not divisible by {Hk}")
    if q.dtype.itemsize == 1:
        raise NotImplementedError(
            "fa_varlen_triton: fp8-q is not a varlen-face contract")

    if M == 0:
        o = q.new_empty((0, H, D), dtype=q.dtype)
        if out is not None:
            out.copy_(o)
            o = out
        if return_softmax_lse:
            return o, q.new_empty((0, H), dtype=torch.float32)
        return o

    capturing = torch.cuda.is_current_stream_capturing()
    if max_seqlen_q is not None:
        maxL = max(int(max_seqlen_q), 1)
    elif capturing:
        raise RuntimeError(
            "fa_varlen_triton: max_seqlen_q required during graph capture "
            "(host-read of cu_seqlens would abort capture)")
    else:
        maxL = max(int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max()), 1)

    left, right = window_size if window_size is not None else (-1, -1)
    lef = int(left) if left is not None and int(left) >= 0 else 1 << 30
    rig = int(right) if right is not None and int(right) >= 0 else 1 << 30

    scale = softmax_scale if softmax_scale is not None else D ** -0.5
    kd = None
    vd_s = None
    has_ds = k_descale is not None or v_descale is not None
    if has_ds:
        kd = torch.ones(Hk, dtype=torch.float32, device=q.device
                        ) if k_descale is None else k_descale.reshape(
                            -1).to(torch.float32).contiguous()
        vd_s = torch.ones(Hk, dtype=torch.float32, device=q.device
                          ) if v_descale is None else v_descale.reshape(
                              -1).to(torch.float32).contiguous()

    o16 = q.dtype if q.dtype.itemsize >= 2 else torch.bfloat16
    o = torch.empty(M, H, D, dtype=o16, device=q.device)
    lse = torch.empty(M, H, dtype=torch.float32, device=q.device)

    _fa_varlen_kernel[(cu_seqlens_q.shape[0] - 1, H,
                       triton.cdiv(maxL, BLOCK_M))](
        q.contiguous(), k.contiguous(), v.contiguous(),
        cu_seqlens_q.to(torch.int32).contiguous(),
        cu_seqlens_k.to(torch.int32).contiguous(),
        kd if has_ds else q,   # placeholder pointers when unused
        vd_s if has_ds else q,
        o, lse,
        scale, lef, rig,
        H=H, D=D, BLOCK_D=D, HKV=Hk, G=H // Hk,
        CAUSAL=bool(causal), HAS_DS=has_ds,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    if out is not None:
        out.copy_(o)
        o = out
    if return_softmax_lse:
        return o, lse
    return o
