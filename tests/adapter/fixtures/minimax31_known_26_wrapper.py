import torch
def fused_minimax_m3_qknorm_rope_kv_insert(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: torch.Tensor | None = None,
    index_k_norm_weight: torch.Tensor | None = None,
    num_index_heads: int = 0,
    slot_mapping: torch.Tensor | None = None,
    index_slot_mapping: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    index_cache: torch.Tensor | None = None,
    block_size: int = 0,
    q_out: torch.Tensor | None = None,
    index_q_out: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
    skip_index_branch: bool = False,
    q_fp8_out: torch.Tensor | None = None,
    q_fp8_scale: float = 1.0,
    kv_k_scale: torch.Tensor | None = None,
    kv_v_scale: torch.Tensor | None = None,
    write_qkv: bool = True,
) -> None:
    """Fused MiniMax-M3 attention pre-processing (in-place).

    Applies Gemma RMSNorm + partial NeoX RoPE to ``qkv`` in place. ``qkv`` is a
    single fused tensor:

    - dense layer (``num_index_heads == 0``): ``[q | k | v]``;
    - sparse layer (``num_index_heads > 0``): ``[q | k | v | index_q |
      index_k]`` — the index branch is read straight out of ``qkv``.

    When ``kv_cache`` is given, also scatter-inserts the normed/roped k & v
    into the paged KV cache by ``slot_mapping`` and, on sparse layers, the
    index key into ``index_cache`` by ``index_slot_mapping``.
    ``kv_cache_dtype`` selects the cache storage/conversion path; a dense
    layer's insert stores exactly what ``reshape_and_cache_flash`` would with
    unit k/v scales. If ``index_slot_mapping`` is omitted, ``slot_mapping`` is
    used for both caches.

    If ``q_out`` / ``index_q_out`` (contiguous ``[N, nq*128]`` / ``[N,
    niq*128]``) are given, the normed/roped q / index_q are written there
    instead of in place — folding the de-interleave into this kernel's store so
    callers skip a separate ``.contiguous()`` copy before the SM100 sparse
    attention's flat TMA descriptor.

    If ``q_fp8_out`` is given, the same normalized q is also written in FP8
    E4M3 using ``q_fp8_scale`` as its dequantization scale. Without ``q_out``,
    q is then written only in FP8 and the q slice of ``qkv`` is left as is.

    ``kv_cache_dtype="nvfp4"`` quantizes k/v into the packed HND
    ``[num_blocks, 2 * num_kv_heads, block_size, 72]`` uint8 cache using the
    device dequantization scales ``kv_k_scale``/``kv_v_scale``. Slot
    ``2 * head + side`` (K = 0, V = 1) holds that head's E2M1 data followed by
    its E4M3 block scales, so every head is one contiguous run of the page.

    When ``skip_index_branch`` is true, sparse rows still keep their packed
    ``[index_q | index_k]`` tail, but the kernel only processes the main q/k/v
    branches and main KV cache. This is used by MiniMax-M3 index-topk reuse
    layers that consume top-k block ids selected by an earlier sparse layer.

    ``write_qkv=False`` needs ``q_fp8_out`` without ``q_out``, for callers that
    read only the FP8 q, ``index_q_out`` and the caches: k and index_k need not
    be rewritten in ``qkv``. The persistent kernel (SM80+, ``rotary_dim`` 64)
    skips those stores; the fallback kernel may still make them.
    """
    torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        num_heads,
        num_kv_heads,
        rotary_dim,
        eps,
        index_q_norm_weight,
        index_k_norm_weight,
        num_index_heads,
        slot_mapping,
        index_slot_mapping,
        kv_cache,
        index_cache,
        block_size,
        q_out,
        index_q_out,
        kv_cache_dtype,
        skip_index_branch,
        q_fp8_out,
        q_fp8_scale,
        kv_k_scale,
        kv_v_scale,
        write_qkv,
    )
