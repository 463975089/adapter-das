# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Multi-backend staged execution facade（门面） for attention-row transform + paged KV store, mirroring the pinned CUDA kernel.

Stage semantics follow tilelang_ops/cuda/fused_transform.py @774ce24: each
stage rounds to BF16; the norm stage reuses the norm_tl provider so the fused
and unfused paths share one reduction order; RoPE is FP32
fma(x0, cos, -(x1 * sin)); the cast quantizes the BF16 stage output with
hcu_adapter.kernels.quant_emu (UE8M0 multiplier in BF16, E4M3-scale path in FP32). No
deepinfer imports.
"""

import torch

from .quant_emu import dequantize_groups, quantize_groups
from . import kv_cache_emu


def _norm(x, weight, eps, backend):
    from .norm import norm

    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    # Unused operands get real tensors as placeholders; the kernel never reads them.
    norm(x, out, weight, x, out, x, x, out, backend=backend, has_residual=False, eps=eps,
         out_scale=1.0, do_fp8_cast=False, num_per_channels=0, col_packed=False,
         round_sf=False, use_packed_ue8m0=False, return_origin_out=False, shared_sf_block_m=0)
    return out


def _rope(x, positions, cos, sin, rotary_dim, offset, is_neox, conjugate, backend):
    if backend == 'lightop':
        from .rope_lightop import apply_rotary_rows
    else:
        from .rope_emu import apply_rotary_rows
    return apply_rotary_rows(x, positions, cos, sin, rotary_dim, offset, is_neox, conjugate)


def fused_transform(x, weight, positions, cos, sin, out_bf16, out_quant, out_sf,
                    cache_quant, cache_sf, slot_mapping, *, backend, do_norm, norm_eps, do_rope,
                    rotary_dim, rope_offset, is_neox, conjugate, cast_fmt, num_per_channels,
                    sf_format, return_bf16, bf16_stage, return_quant, store_cache, block_size):
    work = x
    if do_norm:
        work = _norm(work, weight, norm_eps, backend)
    if return_bf16 and bf16_stage == "norm":
        out_bf16.copy_(work)
    if do_rope:
        work = _rope(work, positions, cos, sin, rotary_dim, rope_offset, is_neox, conjugate, backend)
    if return_bf16 and bf16_stage == "transform":
        out_bf16.copy_(work)
    if not cast_fmt:
        return
    if (backend == 'lightop' and cast_fmt == 'e4m3' and sf_format == 'ue8m0'
            and num_per_channels in (32, 128) and work.shape[-1] % 128 == 0):
        from .quant_lightop import quantize_transform

        quant, sf, sf_float = quantize_transform(work, num_per_channels)
    else:
        quant, sf, sf_float = quantize_groups(work, cast_fmt, num_per_channels, sf_format)
    if return_bf16 and bf16_stage == "cast_back":
        out_bf16.copy_(dequantize_groups(quant, sf_float, cast_fmt, num_per_channels))
    if return_quant:
        out_quant.copy_(quant)
        out_sf.copy_(sf)
    if store_cache:
        kv_cache_emu._store(cache_quant, slot_mapping, quant[:, 0], block_size)
        kv_cache_emu._store(cache_sf, slot_mapping, sf[:, 0, :cache_sf.shape[-1]], block_size)


def fused_transform_lightop(*args, **kwargs):
    """Native norm/RoPE/FP8 stages with the original paged-store layout."""
    kwargs['backend'] = 'lightop'
    return fused_transform(*args, **kwargs)
