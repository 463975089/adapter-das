import torch


def supports_sparse(q, kv, indices, d_v, attn_sink, topk_length):
    return (
        torch.version.hip is not None
        and q.dtype == kv.dtype == torch.bfloat16
        and q.ndim == kv.ndim == indices.ndim == 3
        and q.shape[0] > 0
        and kv.shape[0] > 0
        and 0 < q.shape[1] <= 64
        and q.shape[2] == 512
        and kv.shape[1:] == (1, 512)
        and indices.shape[:2] == (q.shape[0], 1)
        and 2048 <= indices.shape[2] <= 2112
        and indices.dtype == torch.int32
        and d_v == 512
        and attn_sink is None
        and (topk_length is None or topk_length.numel() == q.shape[0])
    )


def sparse_fwd(q, kv, indices, sm_scale, d_v, topk_length=None):
    from flash_mla.flash_mla_interface import flash_mla_sparse_fwd

    valid = (indices >= 0) & (indices < kv.shape[0])
    if topk_length is not None:
        valid &= torch.arange(indices.shape[-1], device=q.device)[None, None, :] < topk_length.reshape(-1, 1, 1)
    active = valid.any(-1)
    safe = torch.where(valid, indices, -1).contiguous()
    safe[:, :, 0] = torch.where(active, safe[:, :, 0], 0)
    padding = (-safe.shape[-1]) % 64
    if padding:
        safe = torch.nn.functional.pad(safe, (0, padding), value=-1)
    heads = q.shape[1]
    if heads != 64:
        q = torch.nn.functional.pad(q, (0, 0, 0, 64 - heads))
    out, maximum, lse = flash_mla_sparse_fwd(
        q=q, kv=kv, indices=safe, sm_scale=sm_scale, d_v=d_v,
    )
    return (
        torch.where(active[:, :, None], out[:, :heads], 0),
        torch.where(active, maximum[:, :heads], -torch.inf),
        torch.where(active, lse[:, :heads], -torch.inf),
    )
