# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Backend facade（门面）: Validation and backend selection for norm; no numerical implementation."""

from .norm_emu import norm as _torch_norm


def norm(x, out, weight, residual, residual_out, sf, extra_sf, origin, *,
         backend, has_residual, eps, out_scale, do_fp8_cast,
         num_per_channels, col_packed, round_sf, use_packed_ue8m0,
         return_origin_out, shared_sf_block_m):
    import torch

    tensors = [x, out, weight]
    if has_residual:
        tensors.extend((residual, residual_out))
    if do_fp8_cast:
        tensors.append(sf)
    if return_origin_out:
        tensors.append(origin)
    if shared_sf_block_m:
        tensors.append(extra_sf)
    if not x.is_cuda or any(t.device != x.device for t in tensors):
        raise ValueError("norm tensors must share one HCU device")
    if x.ndim != 3 or out.shape != x.shape or weight.shape != (x.shape[-1],):
        raise ValueError("Expected x/out[B,M,D] and weight[D]")
    if x.dtype not in (torch.bfloat16, torch.float32) or weight.dtype != x.dtype:
        raise TypeError("norm requires matching BF16/FP32 input and weight")
    if out.dtype != (torch.float8_e4m3fn if do_fp8_cast else x.dtype):
        raise TypeError("norm output dtype does not match the requested format")
    if any(t.stride(-1) != 1 for t in tensors[:3]) or not weight.is_contiguous():
        raise ValueError("norm input/output rows and weight must be dense")
    if has_residual and any(t.shape != x.shape or t.dtype != x.dtype or
                            t.stride(-1) != 1 for t in (residual, residual_out)):
        raise ValueError("Residual tensors must match input shape/dtype with dense rows")
    if return_origin_out and (origin.shape != x.shape or origin.dtype != x.dtype or
                              origin.stride(-1) != 1):
        raise ValueError("Origin output must match input shape/dtype with dense rows")
    if do_fp8_cast:
        groups = x.shape[-1] // num_per_channels
        if col_packed:
            expected = (*x.shape[:-1], (groups + 3) // 4, 4)
        else:
            expected = (*x.shape[:-1], groups)
        if sf.shape != expected or sf.dtype != (torch.uint8 if use_packed_ue8m0 else torch.float32):
            raise ValueError("Scale tensor does not match the requested byte/float layout")
        if sf.stride(-1) != 1 and use_packed_ue8m0:
            raise ValueError("Packed scale bytes must be adjacent inside each pack")
    if shared_sf_block_m:
        m = x.shape[1]
        needed = ((m + shared_sf_block_m - 1) // shared_sf_block_m *
                  ((shared_sf_block_m + 127) // 128 * 128))
        if (x.shape[0] != 1 or x.shape[-1] % 128 or extra_sf.ndim != 3 or
                extra_sf.shape[0] < needed or extra_sf.shape[1:] != (x.shape[-1] // 128, 4) or
                extra_sf.dtype != torch.uint8):
            raise ValueError("Invalid mega-MoE shared scale buffer")
    if not x.numel():
        return
    options = dict(has_residual=has_residual, eps=eps, out_scale=out_scale,
                   do_fp8_cast=do_fp8_cast, num_per_channels=num_per_channels,
                   col_packed=col_packed, round_sf=round_sf,
                   use_packed_ue8m0=use_packed_ue8m0,
                   return_origin_out=return_origin_out, shared_sf_block_m=shared_sf_block_m)
    if backend == "triton":
        from . import norm_triton as triton_elementwise

        triton_elementwise.norm(x, out, weight, residual, residual_out, sf, extra_sf, origin, **options)
    elif backend == "torch":
        _torch_norm(x, out, weight, residual, residual_out, sf, extra_sf, origin, **options)
    elif backend == "lightop":
        from .norm_lightop import norm as lightop_norm

        lightop_norm(x, out, weight, residual, residual_out, sf, extra_sf, origin, **options)
    else:
        raise ValueError("Unknown norm backend: " + backend)
