"""Native Engram inference candidates; independent of the deepinfer runtime."""

import torch


def engram_hash(ngram_token_ids, multipliers, vocab_sizes, offsets, image_token_mask=None):
    """XOR bounded nonnegative token hashes, then map into per-layer tables.

    Negative token IDs are accepted only for masked image rows. Signed-overflow
    hash inputs are not qualified; the model's multiplier builder bounds them.
    """
    if ngram_token_ids.ndim != 2 or multipliers.ndim != 2 or vocab_sizes.ndim != 3:
        raise ValueError("Engram hash expects token[ T,N ], multiplier[ L,N ], vocab[ L,N-1,E ]")
    tokens, ngram = ngram_token_ids.shape
    layers, multiplier_ngram = multipliers.shape
    if ngram < 2 or multiplier_ngram != ngram or vocab_sizes.shape[:2] != (layers, ngram - 1):
        raise ValueError("Engram hash ngram/layer dimensions do not match")
    if (ngram_token_ids.dtype != torch.int32 or multipliers.dtype != torch.int64
            or vocab_sizes.dtype != torch.int32 or offsets.dtype != torch.int32):
        raise TypeError("Engram hash requires INT32 IDs/vocab/offsets and INT64 multipliers")
    device = ngram_token_ids.device
    if device.type != "cuda" or any(t.device != device for t in (multipliers, vocab_sizes, offsets)):
        raise ValueError("Engram hash inputs must share a CUDA/HIP device")
    if tuple(offsets.shape) != (layers, (ngram - 1) * vocab_sizes.shape[-1]):
        raise ValueError("Engram hash offsets shape does not match the flattened tables")
    if image_token_mask is not None:
        if image_token_mask.shape != (tokens,) or image_token_mask.dtype != torch.bool or image_token_mask.device != device:
            raise ValueError("Engram image mask must be BOOL [tokens] on the input device")
        valid_ids = (ngram_token_ids >= 0) | image_token_mask[:, None]
    else:
        valid_ids = ngram_token_ids >= 0
    torch._assert_async(torch.all(valid_ids), "Unmasked negative Engram token IDs are not qualified")
    torch._assert_async(torch.all(vocab_sizes > 0), "Engram table vocab sizes must be positive")
    if tokens:
        largest_id = ngram_token_ids.clamp_min(0).amax(0).to(torch.int64).clamp_min(1)
        bounds = torch.div(torch.full_like(largest_id, torch.iinfo(torch.int64).max),
                           largest_id, rounding_mode="floor")
        torch._assert_async(torch.all((multipliers >= 0) & (multipliers <= bounds)),
                            "Engram hash products must fit nonnegative signed INT64")
    products = ngram_token_ids.to(torch.int64)[None, :, :] * multipliers[:, None, :]
    hashes = products[:, :, 0]
    pieces = []
    for index in range(1, ngram):
        hashes = hashes.bitwise_xor(products[:, :, index])
        pieces.append(torch.remainder(hashes[..., None], vocab_sizes[:, index - 1, None, :]).to(torch.int32))
    result = torch.cat(pieces, dim=-1) + offsets[:, None, :]
    if image_token_mask is not None:
        result = result.masked_fill(image_token_mask[None, :, None], -1)
    return result


def engram_gate_fwd(hidden_states, kv, weight_fused, eps, clamp_value,
                    save_for_backward=True, image_token_mask=None, inplace=False):
    """FP32 norm/dot/gate evaluation with CUDA's signed-square-root convention."""
    if inplace and save_for_backward:
        raise ValueError("Inplace Engram gate destroys inputs needed by backward")
    if hidden_states.ndim != 3:
        raise ValueError("Engram hidden states must be [tokens, hc_mult, hidden_size]")
    tokens, hc_mult, width = hidden_states.shape
    if tuple(kv.shape) != (tokens, hc_mult + 1, width) or tuple(weight_fused.shape) != (hc_mult, width):
        raise ValueError("Engram gate KV/weight shape mismatch")
    if hidden_states.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16 or weight_fused.dtype != torch.float32:
        raise TypeError("Engram gate requires BF16 hidden/KV and FP32 fused weights")
    device = hidden_states.device
    if device.type != "cuda" or kv.device != device or weight_fused.device != device:
        raise ValueError("Engram gate inputs must share a CUDA/HIP device")
    if width <= 0 or eps <= 0 or clamp_value < 0:
        raise ValueError("Engram gate requires positive width/epsilon and nonnegative clamp")
    if image_token_mask is not None:
        if image_token_mask.shape != (tokens,) or image_token_mask.dtype != torch.bool or image_token_mask.device != device:
            raise ValueError("Engram image mask must be BOOL [tokens] on the input device")
    if not save_for_backward:
        from .engram_gate_triton import inference_gate

        result = inference_gate(hidden_states, kv, weight_fused, eps, clamp_value,
                                image_token_mask, inplace)
        return result, None, None, None, None
    x = hidden_states.float()
    key = kv[:, :hc_mult, :].float()
    value = kv[:, hc_mult, :].float()
    rstd_x = torch.rsqrt(x.square().mean(-1) + eps)
    rstd_k = torch.rsqrt(key.square().mean(-1) + eps)
    raw_dot = (x * weight_fused * key).sum(-1)
    dot = raw_dot * rstd_x * rstd_k * (width ** -0.5)
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    result = (x + gate[..., None] * value[:, None, :]).to(hidden_states.dtype)
    if image_token_mask is not None:
        result = torch.where(image_token_mask[:, None, None], hidden_states, result)
    if inplace:
        hidden_states.copy_(result)
        result = hidden_states
    if save_for_backward:
        return result, raw_dot, gate, rstd_x, rstd_k
    return result, None, None, None, None
