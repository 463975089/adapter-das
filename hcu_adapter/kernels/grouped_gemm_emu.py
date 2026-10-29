"""Torch implementation of the existing grouped block-scaled GEMM contract."""

import torch

from .gemm_emu import _recipes, dequantize_fp8, dequantize_mxfp4


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
