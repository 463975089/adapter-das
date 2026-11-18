import torch


def sparse_decode(q, k_cache, indices, head_dim_v, softmax_scale, topk_length, kv_format):
    if q.ndim != 4 or indices.ndim != 3 or indices.shape[:2] != q.shape[:2]:
        raise ValueError("FP8 MLA requires q[B,S,H,D] and indices[B,S,K]")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("FP8 MLA queries must be BF16 or FP16")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1 or k_cache.element_size() != 1:
        raise ValueError("FP8 MLA requires a byte-sized cache[pages,page_size,1,D]")
    if any(t.device != q.device for t in (k_cache, indices)):
        raise ValueError("FP8 MLA tensors must share a device")
    if kv_format != "V32_NO_ROPE":
        raise ValueError("FP8 MLA requires the V32_NO_ROPE layout")
    nope_dim = 512
    scale_bytes = nope_dim // 128 * 4
    row_bytes = nope_dim + scale_bytes
    if k_cache.shape[-1] != row_bytes or q.shape[-1] != nope_dim:
        raise ValueError("FP8 MLA cache width does not match its declared V32 layout")
    if not 0 < head_dim_v <= nope_dim:
        raise ValueError("FP8 MLA value width must fit the NoPE channels")
    batch, seqlen, heads, _ = q.shape
    width = indices.shape[-1]
    if batch == 0 or seqlen == 0 or width == 0:
        return (
            q.new_zeros((batch, seqlen, heads, head_dim_v)),
            torch.full((batch, heads, seqlen), -torch.inf, device=q.device),
        )
    loc = indices.long()
    valid = (loc >= 0) & (loc < k_cache.shape[0] * k_cache.shape[1])
    if topk_length is not None:
        lengths = topk_length.reshape(batch, -1)
        if lengths.shape[1] not in (1, seqlen):
            raise ValueError("FP8 MLA topk_length must have B or B*S elements")
        valid = valid & (torch.arange(width, device=q.device) < lengths[..., None])
    safe = torch.where(valid, loc, 0)
    page_size = k_cache.shape[1]
    raw = k_cache.view(torch.uint8)[safe // page_size, safe % page_size, 0, :]
    values = raw[..., :nope_dim].contiguous().view(torch.float8_e4m3fn).float()
    scales = raw[..., nope_dim:nope_dim + scale_bytes].contiguous().view(torch.float32)
    keys = (values.reshape(*values.shape[:-1], 4, 128) * scales[..., None]).flatten(-2)
    keys = torch.where(valid[..., None], keys, 0.0)
    queries = q.float().reshape(batch * seqlen, heads, -1)
    keys = keys.reshape(batch * seqlen, width, -1)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    scores = torch.bmm(queries, keys.transpose(1, 2)) * scale
    scores = scores.masked_fill(~valid.reshape(batch * seqlen, 1, width), -torch.inf)
    any_valid = valid.reshape(batch * seqlen, width).any(-1)[:, None, None]
    safe_scores = torch.where(any_valid, scores, 0.0)
    probs = torch.where(any_valid, safe_scores.softmax(-1), 0.0)
    out = torch.bmm(probs, keys[..., :head_dim_v]).to(q.dtype)
    lse = scores.logsumexp(-1).reshape(batch, seqlen, heads).transpose(1, 2)
    return out.reshape(batch, seqlen, heads, head_dim_v), lse
