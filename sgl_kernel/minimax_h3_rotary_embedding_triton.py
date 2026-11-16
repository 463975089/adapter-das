import torch

from .compat import HAS_TRITON, check_cuda_tensor

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _h3_rotary_kernel(P, T, C, PS: tl.constexpr, TS: tl.constexpr,
                          CS: tl.constexpr, NH: tl.constexpr, HS: tl.constexpr, RD: tl.constexpr):
        token = tl.program_id(0).to(tl.int64)
        heads = tl.program_id(1) * 4 + tl.arange(0, 4)
        half: tl.constexpr = RD // 2
        lane = tl.arange(0, HS // 2)
        pos = tl.load(P + token * PS).to(tl.int64)
        c = tl.load(C + pos * CS + lane, mask=lane < half, other=0).to(tl.float32)
        s = tl.load(C + pos * CS + half + lane, mask=lane < half, other=0).to(tl.float32)
        base = T + token * TS + heads[:, None] * HS + lane[None, :]
        mask = (heads < NH)[:, None] & (lane < half)[None, :]
        a = tl.load(base, mask=mask, other=0).to(tl.float32)
        b = tl.load(base + half, mask=mask, other=0).to(tl.float32)
        dtype = T.dtype.element_ty
        bs = (b * s[None, :]).to(dtype).to(tl.float32)
        ass = (a * s[None, :]).to(dtype).to(tl.float32)
        if dtype == tl.bfloat16:
            ac = (a * c[None, :]).to(dtype).to(tl.float32)
            bc = (b * c[None, :]).to(dtype).to(tl.float32)
            y1, y2 = ac - bs, bc + ass
        else:
            y1 = tl.fma(a, c[None, :], -bs)
            y2 = tl.fma(b, c[None, :], ass)
        tl.store(base, y1, mask=mask)
        tl.store(base + half, y2, mask=mask)


def minimax_h3_rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox):
    if head_size not in (64, 128) or not is_neox:
        raise ValueError("MiniMax-H3 RoPE requires head_size=64 or 128 and NeoX layout")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != head_size * 3 // 4 or cos_sin_cache.stride(1) != 1:
        raise ValueError("MiniMax-H3 RoPE requires a [positions,head_size*3/4] cache")
    if positions.ndim != 1 or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("MiniMax-H3 positions must be a 1D integer tensor")
    tensors = [query] + ([] if key is None else [key])
    for tensor in tensors:
        if tensor.ndim != 2 or tensor.shape[0] != positions.numel() or tensor.shape[1] % head_size or tensor.stride(1) != 1:
            raise ValueError("MiniMax-H3 Q/K must have contiguous head rows")
        if tensor.device != query.device or tensor.dtype != query.dtype:
            raise ValueError("MiniMax-H3 Q/K device and dtype must match")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("MiniMax-H3 Q/K must be floating point")
    if cos_sin_cache.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("MiniMax-H3 cache must be floating point")
    if positions.device != query.device or cos_sin_cache.device != query.device:
        raise ValueError("MiniMax-H3 positions/cache must share the Q/K device")
    if cos_sin_cache.dtype != query.dtype:
        raise ValueError("MiniMax-H3 cache dtype must match Q/K")
    if positions.numel() == 0:
        return
    if key is not None and key.data_ptr() == query.data_ptr():
        if key.shape != query.shape or key.stride() != query.stride():
            raise ValueError("Aliased MiniMax-H3 Q/K must have identical views")
        key = None
    half = cos_sin_cache.shape[1] // 2
    from hcu_adapter.kill_switch import op_disabled
    if not query.is_cuda or op_disabled("minimax_h3_rotary_embedding"):
        cos = cos_sin_cache[positions, :half].unsqueeze(1)
        sin = cos_sin_cache[positions, half:].unsqueeze(1)
        for tensor in (query, key):
            if tensor is None or tensor.shape[1] == 0:
                continue
            view = tensor.view(tensor.shape[0], -1, head_size)
            a, b = view[..., :half].clone(), view[..., half:2 * half].clone()
            if tensor.dtype == torch.bfloat16:
                y1, y2 = a * cos - b * sin, b * cos + a * sin
            else:
                ad, bd, cd, sd = a.double(), b.double(), cos.double(), sin.double()
                bs = (bd * sd).to(tensor.dtype).double()
                ass = (ad * sd).to(tensor.dtype).double()
                y1 = (ad * cd - bs).float().to(tensor.dtype)
                y2 = (bd * cd + ass).float().to(tensor.dtype)
            view[..., :half].copy_(y1)
            view[..., half:2 * half].copy_(y2)
        return
    if not HAS_TRITON:
        raise RuntimeError("MiniMax-H3 RoPE requires Triton on HCU")
    check_cuda_tensor(positions, query, key, cos_sin_cache)
    for tensor in (query, key):
        if tensor is None or tensor.shape[1] == 0:
            continue
        heads = tensor.shape[1] // head_size
        _h3_rotary_kernel[(positions.numel(), triton.cdiv(heads, 4))](
            positions, tensor, cos_sin_cache, positions.stride(0), tensor.stride(0),
            cos_sin_cache.stride(0), heads, head_size, cos_sin_cache.shape[1], num_warps=4, enable_fp_fusion=False,
        )



from hcu_adapter import registry as _registry

_registry.register("minimax_h3_rotary_embedding", "H3 partial RoPE Triton/torch candidate",
                   "sgl_kernel.minimax_h3_rotary_embedding", _registry.MECH_SEAM)
