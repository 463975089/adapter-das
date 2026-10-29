"""DeepGEMM-contract GEMMs on HCU (no deepinfer imports).

FP8 E4M3 values times power-of-two UE8M0 scales are exactly representable in
BF16, so dequantize-then-GEMM with FP32 accumulation computes the same products
as the SM100 block-scaled MMA; only the FP32 summation order differs.
"""

import torch


def cublaslt_gemm_nt(a, b, d, c=None):
    """Pinned dense NT GEMM: FP32 accumulate, then cast once into N-major D."""
    if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]
            or d.shape != (a.shape[0], b.shape[0]) or d.stride(-1) != 1):
        raise ValueError("cublaslt_gemm_nt expects [M,K] @ [N,K].T into N-major [M,N]")
    if a.dtype != b.dtype or a.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("HCU dense NT GEMM supports matching BF16 or FP32 operands")
    if d.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("HCU dense NT GEMM output must be BF16 or FP32")
    if not a.is_cuda or a.device != b.device or a.device != d.device:
        raise ValueError("Dense GEMM tensors must share one HCU device")
    if c is not None and (c.shape != d.shape or c.device != d.device or c.dtype != d.dtype):
        raise ValueError("Dense GEMM accumulator must match output shape, device and dtype")
    if a.shape[0] == 0 or b.shape[0] == 0:
        return
    if a.shape[1] == 0:
        d.copy_(c) if c is not None else d.zero_()
        return
    if a.dtype == torch.bfloat16 and a.is_contiguous() and b.is_contiguous() and a.shape[1] % 32 == 0:
        from .gemm_lightop import bf16_gemm_nt

        return bf16_gemm_nt(a, b, d, c)
    result = a.float() @ b.float().T
    if c is not None:
        result = result + c.float()
    d.copy_(result)


def ue8m0_to_float(exponents):
    """UE8M0 exponent bytes (int tensor) -> float32 2**(e - 127); e=0 is the 2**-127 subnormal."""
    exponents = exponents.to(torch.int32)
    bits = torch.where(exponents == 0, torch.full_like(exponents, 1 << 22), exponents << 23)
    return bits.view(torch.float32)


def unpack_ue8m0_sf(sf, num_blocks):
    """Packed int32 [..., mn, packed_k] (4 LSB-first exponents each) -> float32 [..., mn, num_blocks]."""
    # Built on device: a host tensor would be an H2D copy inside graph capture.
    shifts = torch.arange(0, 32, 8, dtype=torch.int32, device=sf.device)
    exponents = (sf.unsqueeze(-1) >> shifts) & 255
    return ue8m0_to_float(exponents.flatten(-2)[..., :num_blocks])


def _float_scales(sf, mn, k, gran_mn, gran_k):
    blocks = (k + gran_k - 1) // gran_k
    if sf.dtype == torch.int32:
        if gran_mn != 1:
            raise ValueError("Packed UE8M0 scales require gran_mn=1")
        return unpack_ue8m0_sf(sf, blocks)
    if sf.dtype != torch.float32:
        raise ValueError("Scales must be packed int32 UE8M0 or float32")
    return sf.repeat_interleave(gran_mn, dim=-2)[..., :mn, :]


def dequantize_fp8(x, sf, gran_mn, gran_k, dtype):
    """FP8 E4M3 [..., mn, k] with block scales -> dtype, exact for BF16/FP32."""
    if x.dtype != torch.float8_e4m3fn:
        raise NotImplementedError("HCU DeepGEMM contract currently supports FP8 E4M3 operands only")
    mn, k = x.shape[-2:]
    scales = _float_scales(sf, mn, k, gran_mn, gran_k).to(dtype)
    if k % gran_k == 0:
        # Power-of-two scaling is exact in BF16/FP32; broadcast instead of expanding.
        blocks = x.to(dtype).unflatten(-1, (k // gran_k, gran_k))
        return (blocks * scales.unsqueeze(-1)).flatten(-2)
    return x.to(dtype) * scales.repeat_interleave(gran_k, dim=-1)[..., :k]


def _recipes(recipe, recipe_a, recipe_b):
    if recipe is not None and recipe_a is None and recipe_b is None:
        gran_m, gran_n, gran_k = recipe
        return (gran_m, gran_k), (gran_n, gran_k)
    if recipe is None and recipe_a is not None and recipe_b is not None:
        return tuple(recipe_a), tuple(recipe_b)
    raise ValueError("Pass exactly one of recipe or the recipe_a + recipe_b pair")


def fp8_fp4_gemm_nt(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None,
                    compiled_dims="nk", disable_ue8m0_cast=False, alpha=None, epilogue=None):
    """D = A @ B.T (+ C), A = [M, K], B = [N, K] FP8 with block scales; D BF16/FP32 N-major."""
    if alpha is not None or epilogue is not None:
        raise NotImplementedError("HCU fp8_fp4_gemm_nt does not implement alpha/custom epilogues")
    (a_x, a_sf), (b_x, b_sf) = a, b
    (gran_m, gran_k_a), (gran_n, gran_k_b) = _recipes(recipe, recipe_a, recipe_b)
    m, k = a_x.shape
    n, k_b = b_x.shape
    if k != k_b or d.shape != (m, n) or d.dtype not in (torch.bfloat16, torch.float32) or d.stride(-1) != 1:
        raise ValueError("fp8_fp4_gemm_nt expects [M,K] @ [N,K].T into N-major BF16/FP32 [M,N]")
    if m == 0 or n == 0:
        return
    if k == 0:
        if c is not None:
            d.copy_(c)
        else:
            d.zero_()
        return
    exact = torch.float32 if (c is not None or d.dtype == torch.float32) else torch.bfloat16
    a_deq = dequantize_fp8(a_x, a_sf, gran_m, gran_k_a, exact)
    b_deq = dequantize_fp8(b_x, b_sf, gran_n, gran_k_b, exact)
    result = torch.matmul(a_deq, b_deq.T)
    if c is not None:
        result = result + c.float()
    d.copy_(result)


def fp8_einsum(expr, a, b, d, c=None, recipe=(1, 128, 128), epilogue=None):
    """Pinned DeepGEMM fp8_einsum for 'bhr,hdr->bhd' (the only V4.1 expression).

    a = (x [b, h, r] E4M3, sf [b, h, r/(gran*4)] packed); b = (w [h, d, r] E4M3, sf);
    d is BF16/FP32 [b, h, d] or an FP8 pair (d [b, h, d] E4M3, sfd [b, h*d/128] int32,
    MN-major) quantized like a per-token per-32 cast of the BF16-rounded result.
    """
    if expr != "bhr,hdr->bhd":
        raise NotImplementedError("HCU fp8_einsum implements 'bhr,hdr->bhd' only")
    if epilogue is not None:
        raise NotImplementedError("HCU fp8_einsum does not implement custom epilogues")
    gran_m, gran_n, gran_k = recipe
    (a_x, a_sf), (b_x, b_sf) = a, b
    batch, heads, _ = a_x.shape
    out = d[0] if isinstance(d, tuple) else d
    exact = torch.float32 if (c is not None or out.dtype == torch.float32) else torch.bfloat16
    a_deq = dequantize_fp8(a_x.transpose(0, 1), a_sf.transpose(0, 1), gran_m, gran_k, exact)  # [h, b, r]
    b_deq = dequantize_fp8(b_x, b_sf, gran_n, gran_k, exact)  # [h, d, r]
    result = torch.bmm(a_deq, b_deq.transpose(1, 2)).transpose(0, 1)  # [b, h, d]
    if c is not None:
        result = result + c.float()
    if not isinstance(d, tuple):
        out.copy_(result)
        return
    from .quant_emu import per_token_cast

    flat = result.to(torch.bfloat16).reshape(batch, -1)
    # check_sf_layout(sfd, m, batch*n, 1, 32, ..., MN-major): the TMA-aligned column-major packing.
    q, sf = per_token_cast(flat, "e4m3", 32, use_tma_aligned_col_major_sf=True, round_sf=True,
                           use_packed_ue8m0=True)
    d[0].copy_(q.view_as(d[0]))
    d[1].copy_(sf)


def dequantize_mxfp4(w, sf, gran_k, dtype):
    """Packed E2M1 int8 [..., n, k/2] (low nibble = even k) with UE8M0 scales -> dtype.

    sf is packed int32 [..., n, k/(gran_k*4)] or float32 [..., n, k/gran_k]. E2M1
    values times power-of-two scales are exact in BF16.
    """
    from .quant_emu import unpack_e2m1

    n, k = w.shape[-2], w.shape[-1] * 2
    scales = _float_scales(sf, n, k, 1, gran_k).to(dtype)
    values = unpack_e2m1(w).to(dtype)  # [..., n, k]
    return (values.unflatten(-1, (k // gran_k, gran_k)) * scales.unsqueeze(-1)).flatten(-2)


def m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, grouped_layout, recipe=None, recipe_a=None,
                                         recipe_b=None, compiled_dims="nk", disable_ue8m0_cast=False,
                                         use_psum_layout=False, ensure_zero_padding=True,
                                         expected_m_for_psum_layout=None, epilogue=None):
    """D[m] = A[m] @ B[g(m)].T for M-grouped contiguous rows (pinned DeepGEMM contract).

    A: FP8 E4M3 [M, K] with per-token scales; B: [G, N, K] FP8 or packed FP4 int8
    [G, N, K/2] with UE8M0 scales. grouped_layout: PSUM end offsets [G] (group g
    rows are [align(end[g-1], mk_align), end[g])) or per-row group ids [M] (-1 =
    padding). Padding rows of D are zeroed. Graph-safe: every group is computed
    densely over all rows and selected by mask, so no host reads are needed.
    """
    from .runtime_config import get_mk_alignment_for_contiguous_layout

    if epilogue is not None or disable_ue8m0_cast:
        raise NotImplementedError("HCU grouped GEMM supports the default UE8M0 path without epilogues")
    (a_x, a_sf), (b_x, b_sf) = a, b
    (gran_m, gran_k_a), (gran_n, gran_k_b) = _recipes(recipe, recipe_a, recipe_b)
    m = a_x.shape[0]
    groups = b_x.shape[0]
    if m == 0:
        return
    rows = torch.arange(m, device=a_x.device)
    if use_psum_layout:
        ends = grouped_layout.long()
        align = get_mk_alignment_for_contiguous_layout()
        starts = torch.cat((ends.new_zeros(1), (ends[:-1] + align - 1) // align * align))
        member = (rows[None, :] >= starts[:, None]) & (rows[None, :] < ends[:, None])  # [G, M]
    else:
        member = grouped_layout.long()[None, :] == torch.arange(groups, device=a_x.device)[:, None]
    a_deq = dequantize_fp8(a_x, a_sf, gran_m, gran_k_a, torch.bfloat16)
    result = torch.zeros(d.shape, device=d.device, dtype=torch.float32)
    for g in range(groups):
        if b_x.dtype == torch.int8:
            w = dequantize_mxfp4(b_x[g], b_sf[g], gran_k_b, torch.bfloat16)
        else:
            w = dequantize_fp8(b_x[g], b_sf[g], gran_n, gran_k_b, torch.bfloat16)
        out = torch.matmul(a_deq, w.T).float()
        result = torch.where(member[g][:, None], out, result)
    d.copy_(result)
