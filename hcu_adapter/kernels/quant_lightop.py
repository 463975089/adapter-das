"""quant lightop implementation; upstream rounding/layout contracts are unchanged."""

import torch

from .lightop_usage import _count


def per_token_cast(x, fmt, num_per_channels, x_block_size=None,
                   use_tma_aligned_col_major_sf=False, round_sf=False,
                   use_packed_ue8m0=False, use_e4m3_sf=False,
                   stochastic_cast=False, out=None):
    from .quant_emu import per_token_cast as compatibility_cast

    if (fmt != 'e4m3' or num_per_channels not in (32, 128) or not round_sf
            or not use_packed_ue8m0 or use_e4m3_sf or stochastic_cast
            or x_block_size is not None or isinstance(x, tuple)):
        _count('quant.compatibility')
        return compatibility_cast(x, fmt, num_per_channels, x_block_size,
                                  use_tma_aligned_col_major_sf, round_sf,
                                  use_packed_ue8m0, use_e4m3_sf, stochastic_cast, out)
    from lightop.quant import per_token_group_quant_fp8

    if (x.ndim != 2 or not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float32)
            or x.shape[1] % 128 or x.shape[1] <= 0 or x.stride(-1) != 1):
        raise ValueError('LightOp FP8 requires dense BF16/FP32 rows, K divisible by 128')
    m, k = x.shape
    groups = k // num_per_channels
    width = (groups + 3) // 4
    if out is None:
        q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
        sf = (torch.empty_strided((m, width), (1, (m + 3) // 4 * 4), dtype=torch.int32, device=x.device)
              if use_tma_aligned_col_major_sf else torch.empty((m, width), dtype=torch.int32, device=x.device))
    else:
        if use_tma_aligned_col_major_sf:
            raise NotImplementedError('Pinned cast does not accept column-major output scale buffers')
        q, sf = out
        if (q.shape != (m, k) or q.dtype != torch.float8_e4m3fn or not q.is_contiguous()
                or sf.shape != (m, width) or sf.dtype != torch.int32
                or sf.stride(-1) != 1 or q.device != x.device or sf.device != x.device):
            raise ValueError('LightOp cast output buffers do not match the pinned ABI')
    if m:
        scales = torch.empty((m, groups), device=x.device, dtype=torch.float32)
        per_token_group_quant_fp8(x.contiguous(), num_per_channels, eps=1e-4,
                                 out_q=q, out_scale=scales, use_ue8m0=True)
        exponent = (scales.view(torch.int32) >> 23).to(torch.uint8)
        if use_tma_aligned_col_major_sf:
            raw = sf.T.view(torch.uint8)
            for lane in range(4):
                values = exponent[:, lane::4]
                raw[:values.shape[1], lane::4].copy_(values.T)
        else:
            sf.view(torch.uint8)[:, :groups].copy_(exponent)
        _count('quant.native')
    return q, sf
