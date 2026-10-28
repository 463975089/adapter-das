"""Framework-free providers for the pinned gather and RMSNorm kernel contracts.

Torch is an explicit execution backend, not a binding to upstream test refs.
Both providers write caller-owned outputs and avoid data-dependent host reads.
"""


def gather(x, index, out, *, backend):
    import torch

    if not x.is_cuda or x.device != index.device or x.device != out.device:
        raise ValueError("gather tensors must share one HCU device")
    if x.ndim != 2 or out.shape != (index.numel(), x.shape[1]):
        raise ValueError("Expected byte matrices x[N,D] and out[M,D]")
    if x.dtype != torch.int8 or out.dtype != torch.int8:
        raise TypeError("The upstream gather factory operates on int8 byte views")
    if index.ndim != 1 or index.dtype not in (torch.int32, torch.int64):
        raise TypeError("Expected one-dimensional int32/int64 indices")
    if not index.is_contiguous() or not out.is_contiguous() or x.stride(1) != 1:
        raise ValueError("Expected dense rows, contiguous indices and output")
    if not out.numel():
        return
    if backend == "triton":
        from . import elementwise_triton as triton_elementwise

        triton_elementwise.gather(x, index, out)
    elif backend == "torch":
        _torch_gather(x, index, out)
    else:
        raise ValueError("Unknown gather backend: " + backend)


def _torch_gather(x, index, out):
    import torch

    if not x.shape[0]:
        out.zero_()
        return
    selected = torch.index_select(x, 0, index.clamp(0, x.shape[0] - 1))
    valid = (index >= 0) & (index < x.shape[0])
    out.copy_(torch.where(valid[:, None], selected, 0))


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
        from . import elementwise_triton as triton_elementwise

        triton_elementwise.norm(x, out, weight, residual, residual_out, sf, extra_sf, origin, **options)
    elif backend == "torch":
        _torch_norm(x, out, weight, residual, residual_out, sf, extra_sf, origin, **options)
    elif backend == "lightop":
        from .lightop_ops import norm as lightop_norm

        lightop_norm(x, out, weight, residual, residual_out, sf, extra_sf, origin, **options)
    else:
        raise ValueError("Unknown norm backend: " + backend)


def _torch_norm(x, out, weight, residual, residual_out, sf, extra_sf, origin, *,
                has_residual, eps, out_scale, do_fp8_cast, num_per_channels,
                col_packed, round_sf, use_packed_ue8m0,
                return_origin_out, shared_sf_block_m):
    import torch

    value = x + residual if has_residual else x
    if has_residual:
        residual_out.copy_(value)
    value = value.float()
    normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    result = normalized * (weight.float() * out_scale)
    if return_origin_out:
        origin.copy_(result)
    if not do_fp8_cast:
        out.copy_(result)
        return
    # Upstream casts the norm result back to input dtype BEFORE measuring amax.
    q = result.to(x.dtype).float().reshape(*x.shape[:-1], -1, num_per_channels)
    amax = q.abs().amax(-1).clamp_min(1e-4)
    if round_sf:
        scale_bits = (amax / 448.0).view(torch.int32)
        exponent_byte = ((scale_bits - 1) >> 23) + 1
        multiplier = ((254 - exponent_byte) << 23).view(torch.float32)
        scale = (exponent_byte << 23).view(torch.float32)
        packed = exponent_byte.to(torch.uint8)
    else:
        multiplier = 448.0 / amax
        scale = amax / 448.0
    out.copy_((q * multiplier[..., None]).reshape(x.shape).to(out.dtype))
    if use_packed_ue8m0:
        if col_packed:
            # The final incomplete int32 pack has unspecified padding bytes.
            for lane in range(4):
                count = packed[..., lane::4].shape[-1]
                sf[..., :count, lane].copy_(packed[..., lane::4])
        else:
            sf.copy_(packed)
    else:
        sf.copy_(scale)
    if shared_sf_block_m:
        row = torch.arange(x.shape[1], device=x.device)
        local = row % shared_sf_block_m
        mapped = (row // shared_sf_block_m * ((shared_sf_block_m + 127) // 128 * 128) +
                  local // 128 * 128 + local % 32 * 4 + local % 128 // 32)
        extra_sf.index_copy_(0, mapped, packed.reshape(x.shape[1], -1, 4))
