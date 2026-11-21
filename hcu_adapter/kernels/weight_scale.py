# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Native UE8M0 weight-scale layout, matching tile-kernels 2.1.0 GPU ABI.

This is a loading-time candidate, not a GEMM implementation. Four exponent
bytes occupy one int32, with the first byte in the least significant bits.
"""

import torch
import torch.nn.functional as F


def _expanded_packed(exponents, block_size):
    columns = exponents.shape[-1]
    padded = F.pad(exponents, (0, (-columns) % 4))
    packed = padded[..., 0::4].clone()
    for byte in range(1, 4):
        packed.bitwise_or_(padded[..., byte::4] << (8 * byte))
    return packed.repeat_interleave(block_size, dim=-2)


def transform_sf_into_required_layout(sf, mn, k, recipe, num_groups=None,
                                      is_sfa=None, disable_ue8m0_cast=False,
                                      psum_layout=None):
    """Produce the pinned DeepGEMM SM100 packed scale ABI on HIP.

    PSUM gap layouts and the SM90 floating-scale mode are not implemented.
    Packed inputs are validated and returned without another conversion.
    """
    if disable_ue8m0_cast or psum_layout is not None:
        raise NotImplementedError("HCU scale transform requires packed UE8M0 without PSUM gaps")
    if len(recipe) == 3 and is_sfa is not None:
        gran_mn, gran_k = recipe[0 if is_sfa else 1], recipe[2]
    elif len(recipe) == 2 and is_sfa is None:
        gran_mn, gran_k = recipe
    else:
        raise ValueError("Three-part recipes require is_sfa; two-part recipes forbid it")
    if gran_mn <= 0 or gran_k not in (32, 128) or mn <= 0 or k <= 0:
        raise ValueError("Scale dimensions must be positive; K granularity must be 32 or 128")
    if sf.device.type != "cuda":
        raise ValueError("HCU scale transform requires a CUDA/HIP device")
    groups = () if num_groups is None else (num_groups,)
    packed_k = (k + 4 * gran_k - 1) // (4 * gran_k)
    aligned_mn = (mn + 3) // 4 * 4
    shape = (*groups, mn, packed_k)
    stride = ((aligned_mn * packed_k,) if groups else ()) + (1, aligned_mn)
    if sf.dtype == torch.int32:
        if gran_mn != 1 or tuple(sf.shape) != shape or sf.stride() != stride:
            raise ValueError("Packed scales require gran_mn=1 and aligned MN-major shape/stride")
        return sf
    expected = (*groups, (mn + gran_mn - 1) // gran_mn, (k + gran_k - 1) // gran_k)
    if sf.dtype != torch.float32 or tuple(sf.shape) != expected:
        raise ValueError("Unpacked scales must be FP32 with the recipe's group/MN/K shape")
    bits = sf.view(torch.int32)
    torch._assert_async(torch.all((bits & -2139095041) == 0),
                        "Scale transform requires nonnegative power-of-two bit patterns")
    packed = _expanded_packed((bits >> 23) & 255, gran_mn)[..., :mn, :]
    output = torch.empty_strided(shape, stride, dtype=torch.int32, device=sf.device)
    output.copy_(packed)
    return output


def batched_transpose_weight_sf(sf, block_size, transpose_sf1s, outs):
    """Pack both weight orientations, preserving grouped strides and out buffers.

    All value checks stay on the device; no scale values are read by the host.
    Non-power-of-two/negative scales fail asynchronously, like the CUDA kernel.
    """
    if block_size not in (32, 128):
        raise ValueError("Weight scale block_size must be 32 or 128")
    if not len(sf) == len(transpose_sf1s) == len(outs):
        raise ValueError("Weight scales, transpose flags and outs must have equal lengths")
    if not sf:
        return []
    device = sf[0].device
    if device.type != "cuda":
        raise ValueError("HCU weight scales require a CUDA/HIP device")
    layouts = []
    pool_size = 0
    for scale, transpose, out in zip(sf, transpose_sf1s, outs):
        if (scale.dtype != torch.float32 or scale.device != device
                or scale.ndim not in (2, 3) or not scale.is_contiguous()):
            raise ValueError("Weight scales must be contiguous FP32 tensors on one device")
        groups = tuple(scale.shape[:-2])
        sn, sk = scale.shape[-2:]
        n, k = sn * block_size, sk * block_size
        if min(scale.shape) <= 0 or n % 128 or k % 128:
            raise ValueError("Weight dimensions must be positive multiples of 128")
        pn, pk = (sn + 3) // 4, (sk + 3) // 4
        specifications = (
            ((*groups, n, pk), ((n * pk,) if groups else ()) + (1, n)),
            ((*groups, pn, k) if transpose else (*groups, k, pn),
             ((k * pn,) if groups else ()) + ((k, 1) if transpose else (1, k))),
        )
        size = (groups[0] if groups else 1) * (n * pk + k * pn)
        layouts.append((specifications, pool_size))
        if out is None:
            pool_size += size
        else:
            if len(out) != 2:
                raise ValueError("Each out must contain two scale tensors")
            for tensor, (shape, stride) in zip(out, specifications):
                if (tensor.dtype != torch.int32 or tensor.device != device
                        or tuple(tensor.shape) != shape or tensor.stride() != stride
                        or tensor.data_ptr() % 32):
                    raise ValueError("Out scale dtype/device/shape/stride/alignment mismatch")
    pool = torch.empty(pool_size, dtype=torch.int32, device=device)
    result = []
    for scale, transpose, out, (specifications, offset) in zip(
            sf, transpose_sf1s, outs, layouts):
        if out is None:
            tensors = []
            for shape, stride in specifications:
                tensors.append(pool.as_strided(shape, stride, offset))
                offset += tensors[-1].numel()
            out = tuple(tensors)
        bits = scale.view(torch.int32)
        torch._assert_async(torch.all((bits & -2139095041) == 0),
                            "Weight scales must have nonnegative power-of-two bit patterns")
        exponents = (bits >> 23) & 255
        out[0].copy_(_expanded_packed(exponents, block_size))
        packed_transpose = _expanded_packed(exponents.transpose(-1, -2), block_size)
        out[1].copy_(packed_transpose.transpose(-1, -2) if transpose else packed_transpose)
        result.append(out)
    return result
