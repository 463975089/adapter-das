"""Native inference candidates for the pinned tile-kernels mHC contracts."""

import torch


def tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=None):
    """DeepGEMM split-K contract using full-FP32 torch GEMMs on HCU."""
    if a.ndim != 2 or b.ndim != 2 or d.ndim not in (2, 3) or sqr_sum.ndim not in (1, 2):
        raise ValueError("mHC prenorm GEMM rank mismatch")
    _check((a, torch.bfloat16), (b, torch.float32), (d, torch.float32), (sqr_sum, torch.float32))
    tokens, k = a.shape
    width, k_b = b.shape
    splits = 1 if num_splits is None else num_splits
    if k != k_b or k <= 0 or k % 64 or not 0 < width <= 128 or width % 8 or splits < 1:
        raise ValueError("mHC prenorm GEMM requires K%64=0, 0<N<=128, N%8=0 and positive splits")
    if any(t.stride(-1) != 1 for t in (a, b, d)) or not sqr_sum.is_contiguous():
        raise ValueError("mHC prenorm GEMM requires dense K/N rows and contiguous square sums")
    expected_d = (tokens, width) if num_splits is None else (splits, tokens, width)
    expected_s = (tokens,) if num_splits is None else (splits, tokens)
    _shape(d, expected_d)
    _shape(sqr_sum, expected_s)
    if tokens == 0:
        return
    num_k_blocks = k // 64
    blocks_per_split, k_remain = divmod(num_k_blocks, splits)
    for split in range(splits):
        start_block = split * blocks_per_split + min(split, k_remain)
        block_count = blocks_per_split + (split < k_remain)
        output = d if num_splits is None else d[split]
        square = sqr_sum if num_splits is None else sqr_sum[split]
        if block_count:
            start, end = start_block * 64, (start_block + block_count) * 64
            lhs = a[:, start:end].float()
            output.copy_(torch.matmul(lhs, b[:, start:end].float().T))
            square.copy_(lhs.square().sum(-1))
        else:
            output.zero_()
            square.zero_()


def _check(*typed_tensors):
    device = typed_tensors[0][0].device
    for tensor, dtype in typed_tensors:
        if tensor.device != device or device.type != "cuda":
            raise ValueError("mHC tensors must share a CUDA/HIP device")
        if tensor.dtype != dtype:
            raise TypeError("mHC tensor dtype does not match the pinned inference contract")


def _shape(tensor, expected):
    if tuple(tensor.shape) != tuple(expected):
        raise ValueError("mHC tensor shape mismatch: " + str(tuple(tensor.shape)) + " != " + str(tuple(expected)))


def _sum_splits(tensor, n_splits):
    if n_splits < 1 or tensor.shape[0] != n_splits:
        raise ValueError("mHC split count does not match its partials")
    # The CUDA epilogue accumulates split-K partials in increasing split order.
    result = torch.zeros_like(tensor[0])
    for index in range(n_splits):
        result = result + tensor[index]
    return result


def expand_to_mhc_fwd(x, out):
    _check((x, torch.bfloat16), (out, torch.bfloat16))
    if x.ndim != 2 or out.ndim != 3:
        raise ValueError("mHC expand expects [tokens, hidden] and [tokens, mult, hidden]")
    _shape(x, (out.shape[0], out.shape[2]))
    out.copy_(x[:, None, :].expand_as(out))


def mhc_pre_apply_mix_fwd(x, mix, out):
    _check((x, torch.bfloat16), (mix, torch.float32), (out, torch.bfloat16))
    if x.ndim != 3:
        raise ValueError("mHC apply expects [tokens, mult, hidden]")
    tokens, mult, hidden = x.shape
    _shape(mix, (tokens, mult))
    _shape(out, (tokens, hidden))
    result = torch.zeros((tokens, hidden), dtype=torch.float32, device=x.device)
    for index in range(mult):
        result = result + mix[:, index, None] * x[:, index, :].float()
    out.copy_(result)


def mhc_head_compute_mix_fwd(input_mix, mhc_scale, mhc_base, output_mix, mhc_pre_eps):
    _check(*((t, torch.float32) for t in (input_mix, mhc_scale, mhc_base, output_mix)))
    if input_mix.ndim != 2:
        raise ValueError("mHC head mix expects [tokens, mult]")
    _shape(mhc_scale, (1,))
    _shape(mhc_base, (input_mix.shape[1],))
    _shape(output_mix, input_mix.shape)
    output_mix.copy_(torch.sigmoid(input_mix * mhc_scale[0] + mhc_base) + mhc_pre_eps)


def mhc_pre_split_mixes_fwd(input_mixes, mhc_scale, mhc_base, pre_layer_mix,
                           post_layer_mix, comb_res_mix, mhc_post_mult_value, mhc_pre_eps):
    _check(*((t, torch.float32) for t in
             (input_mixes, mhc_scale, mhc_base, pre_layer_mix, post_layer_mix, comb_res_mix)))
    if input_mixes.ndim != 2 or pre_layer_mix.ndim != 2:
        raise ValueError("mHC split mixes expects two-dimensional tensors")
    tokens, mult = pre_layer_mix.shape
    width = mult * (mult + 2)
    _shape(input_mixes, (tokens, width))
    _shape(mhc_scale, (3,))
    _shape(mhc_base, (width,))
    _shape(post_layer_mix, (tokens, mult))
    _shape(comb_res_mix, (tokens, mult * mult))
    pre_layer_mix.copy_(torch.sigmoid(input_mixes[:, :mult] * mhc_scale[0] + mhc_base[:mult]) + mhc_pre_eps)
    post_layer_mix.copy_(torch.sigmoid(input_mixes[:, mult:2 * mult] * mhc_scale[1]
                                      + mhc_base[mult:2 * mult]) * mhc_post_mult_value)
    comb_res_mix.copy_(input_mixes[:, 2 * mult:] * mhc_scale[2] + mhc_base[2 * mult:])


def mhc_sinkhorn_fwd(comb_res_mix, comb_res_mix_out, repeat, eps):
    _check((comb_res_mix, torch.float32), (comb_res_mix_out, torch.float32))
    if comb_res_mix.ndim != 3 or comb_res_mix.shape[1] != comb_res_mix.shape[2]:
        raise ValueError("mHC Sinkhorn expects [tokens, mult, mult]")
    _shape(comb_res_mix_out, comb_res_mix.shape)
    if repeat < 1 or eps <= 0:
        raise ValueError("mHC Sinkhorn requires positive repeat and epsilon")
    result = torch.softmax(comb_res_mix, dim=-1) + eps
    result = result / (result.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        result = result / (result.sum(dim=-1, keepdim=True) + eps)
        result = result / (result.sum(dim=-2, keepdim=True) + eps)
    comb_res_mix_out.copy_(result)


def mhc_post_fwd(x, residual, post_layer_mix, comb_res_mix, out=None):
    _check((x, torch.bfloat16), (residual, torch.bfloat16),
           (post_layer_mix, torch.float32), (comb_res_mix, torch.float32))
    if residual.ndim != 4:
        raise ValueError("mHC post expects [sequences, tokens, mult, hidden]")
    seqs, tokens, mult, hidden = residual.shape
    _shape(x, (seqs, tokens, hidden))
    _shape(post_layer_mix, (seqs, tokens, mult, 1))
    _shape(comb_res_mix, (seqs, tokens, mult, mult))
    if out is None:
        out = torch.empty_like(residual)
    _check((residual, torch.bfloat16), (out, torch.bfloat16))
    _shape(out, residual.shape)
    result = post_layer_mix * x.float().unsqueeze(-2)
    # CUDA's matrix is indexed [input_head, output_head], not the reverse.
    for index in range(mult):
        result = result + comb_res_mix[..., index, :, None] * residual[..., index, None, :].float()
    out.copy_(result)
    return out


def mhc_fn_normw_merge_fwd(fn, normw, out_fn):
    _check(*((t, torch.float32) for t in (fn, normw, out_fn)))
    if fn.ndim != 2:
        raise ValueError("mHC fn must be a matrix")
    _shape(normw, (fn.shape[1],))
    _shape(out_fn, fn.shape)
    out_fn.copy_(fn * normw)


def mhc_reduce_partials_and_rmsnorm_fwd(out_mul_splitted, sqrsum_splitted,
                                      out_mul, sqrsum, out, rms_group_size, rms_eps, n_splits):
    _check(*((t, torch.float32) for t in (out_mul_splitted, sqrsum_splitted, out_mul, sqrsum, out)))
    if out_mul_splitted.ndim != 4:
        raise ValueError("mHC reduce expects [splits, tokens, groups, mixes]")
    splits, tokens, groups, width = out_mul_splitted.shape
    _shape(sqrsum_splitted, (splits, tokens, groups))
    _shape(out_mul, (tokens, groups, width))
    _shape(sqrsum, (tokens, groups))
    _shape(out, (tokens, width))
    if rms_group_size <= 0 or rms_eps <= 0:
        raise ValueError("mHC RMSNorm requires positive group width and epsilon")
    summed = _sum_splits(out_mul_splitted, n_splits)
    squares = _sum_splits(sqrsum_splitted, n_splits)
    result = torch.zeros_like(out)
    for index in range(groups):
        result = result + summed[:, index, :] * torch.rsqrt(squares[:, index, None] / rms_group_size + rms_eps)
    out_mul.copy_(summed)
    sqrsum.copy_(squares)
    out.copy_(result)


def mhc_pre_big_fuse_fwd(gemm_out_mul, gemm_out_sqrsum, mhc_scale, mhc_base,
                         residual, post_mix, comb_mix, layer_input, rms_eps,
                         mhc_pre_eps, mhc_sinkhorn_eps, mhc_post_mult_value,
                         sinkhorn_repeat, n_splits=16):
    _check((residual, torch.bfloat16), (layer_input, torch.bfloat16),
           *((t, torch.float32) for t in (gemm_out_mul, gemm_out_sqrsum, mhc_scale, mhc_base, post_mix, comb_mix)))
    if residual.ndim != 3:
        raise ValueError("mHC fused pre expects [tokens, mult, hidden]")
    tokens, mult, hidden = residual.shape
    width = mult * (mult + 2)
    _shape(gemm_out_mul, (n_splits, tokens, width))
    _shape(gemm_out_sqrsum, (n_splits, tokens))
    if rms_eps <= 0:
        raise ValueError("mHC RMSNorm epsilon must be positive")
    mixes = _sum_splits(gemm_out_mul, n_splits)
    squares = _sum_splits(gemm_out_sqrsum, n_splits)
    mixes = mixes * torch.rsqrt(squares[:, None] / (mult * hidden) + rms_eps)
    pre_mix = torch.empty((tokens, mult), dtype=torch.float32, device=residual.device)
    mhc_pre_split_mixes_fwd(mixes, mhc_scale, mhc_base, pre_mix, post_mix,
                            comb_mix, mhc_post_mult_value, mhc_pre_eps)
    mhc_sinkhorn_fwd(comb_mix.view(tokens, mult, mult), comb_mix.view(tokens, mult, mult),
                     sinkhorn_repeat, mhc_sinkhorn_eps)
    mhc_pre_apply_mix_fwd(residual, pre_mix, layer_input)


def mega_mhc(x, residual, shifted_prev_mix, post_mix, comb_res_mix, fn, mix_scales, mix_bases,
             hc_mult, hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps, num_sinkhorn_iters,
             rmsnorm_weight, rmsnorm_eps, rmsnorm_scale, new_residual, new_prev_mix, new_post_mix,
             new_comb_res_mix, y_bf16=None, y_fp8=None, y_gemm_sf=None, y_routed_sf=None,
             y_shared_sf=None, shared_sf_block_m=0, mix_backend="torch"):
    """Pinned DeepGEMM mega_mhc (post + mixes + pre + RMSNorm [+ FP8]) on HCU.

    Numerics follow vendor tests/test_mega_mhc.py::mhc_reference: FP32 post,
    FP32 (non-TF32) mix GEMM, BF16 trunk and norm input, Sinkhorn per
    mhc_sinkhorn_fwd, per-32 packed UE8M0 FP8 cast of the BF16 norm output.
    """
    from .quant_emu import per_token_cast
    if mix_backend == 'lightop':
        from .quant_lightop import per_token_cast

    if (shifted_prev_mix is None) != (new_prev_mix is None):
        raise ValueError("Shifted mHC needs both shifted_prev_mix and new_prev_mix")
    if (y_routed_sf is None) != (y_shared_sf is None) or (y_gemm_sf is not None and y_routed_sf is not None):
        raise ValueError("FP8 output needs y_gemm_sf alone, or y_routed_sf with y_shared_sf")
    if y_shared_sf is not None:
        raise NotImplementedError("HCU mega_mhc does not write the Mega MoE shared-expert SF layout")
    if y_bf16 is None and y_fp8 is None:
        raise ValueError("mega_mhc needs at least one of y_bf16 / y_fp8")
    tokens, hidden = x.shape
    trunk = x.float().unsqueeze(1) * post_mix
    trunk = trunk + torch.einsum("tij,tih->tjh", comb_res_mix, residual.float())
    new_residual.copy_(trunk)
    trunk = new_residual.float()
    norm_input = None
    if shifted_prev_mix is not None:
        norm_input = (trunk * shifted_prev_mix).sum(dim=1).to(torch.bfloat16)
    if mix_backend == "lightop":
        from .mhc_lightop import hc_projection

        mixes = hc_projection(new_residual.flatten(1), fn, hc_norm_eps)
    elif mix_backend == "torch":
        mixes = torch.matmul(trunk.flatten(1), fn.float().T)  # FP32; HIP matmul does not use TF32
        mixes = mixes * torch.rsqrt(trunk.square().sum((1, 2)) / (hc_mult * hidden) + hc_norm_eps)[:, None]
    else:
        raise ValueError("Unknown mHC mix projection backend")
    scales = torch.cat((mix_scales[0].expand(hc_mult), mix_scales[1].expand(hc_mult),
                        mix_scales[2].expand(hc_mult * hc_mult)))
    mixes = mixes * scales + mix_bases
    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(2) + hc_pre_eps
    new_post_mix.copy_((mixes[:, hc_mult:2 * hc_mult].sigmoid() * hc_post_scale).unsqueeze(2))
    comb = mixes[:, 2 * hc_mult:].reshape(tokens, hc_mult, hc_mult).contiguous()
    mhc_sinkhorn_fwd(comb, new_comb_res_mix, num_sinkhorn_iters, sinkhorn_eps)
    if new_prev_mix is not None:
        new_prev_mix.copy_(pre_mix)
    if norm_input is None:
        norm_input = (trunk * pre_mix).sum(dim=1).to(torch.bfloat16)
    value = norm_input.float()
    y = (value * torch.rsqrt(value.square().mean(1) + rmsnorm_eps).unsqueeze(1)
         * (rmsnorm_weight.float() * rmsnorm_scale)).to(torch.bfloat16)
    if y_bf16 is not None:
        y_bf16.copy_(y)
    if y_fp8 is not None:
        q, sf = per_token_cast(y, "e4m3", 32, round_sf=True, use_packed_ue8m0=True)
        y_fp8.copy_(q)
        (y_gemm_sf if y_gemm_sf is not None else y_routed_sf).copy_(sf)
