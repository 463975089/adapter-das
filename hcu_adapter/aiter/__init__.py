# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""`aiter` (AMD ROCm kernel library) adapter for HCU.

The DTK vendor wheel ports most triton kernels to HCU, so `_base.LibStubFinder`
prefers the real package and we only fill gaps:

- `fused_clamp_act_mul` (DeepseekV2MLP `_is_hip` fast path): absent from the
  wheel — composed from upstream HCU-safe triton kernels (silu_and_mul_clamp +
  per-token-group-128 fp8 quant);
- `aiter.silu_and_mul` / `aiter.dtypes`: fill-in bridges in case the real
  top-level module doesn't export them;
- loud lazy stubs for modules missing from the wheel entirely (custom
  allreduce, etc.), so unsupported paths fail with a clear message.

Registered via `hcu_adapter.register_all()`.
"""

import torch

from hcu_adapter._base import LibStubFinder, is_registered

_FP8_DTYPES = (
    torch.float8_e4m3fn,
    getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn),
)


def _hcu_silu_and_mul(out, input, limit=0.0):
    """aiter's ROCm silu_and_mul: (out, input, limit); limit=0 = no clamp."""
    from sgl_kernel.elementwise import silu_and_mul

    silu_and_mul(input, out)
    if limit:
        out.clamp_(max=limit)
    return out


def _hcu_fused_clamp_act_mul(
    gate_up,
    swiglu_limit=None,
    activation="silu",
    dtype_quant=None,
    transpose_scale=False,
    **kwargs,
):
    """aiter `fused_clamp_act_mul` (used by DeepseekV2MLP under _is_hip):
    silu(min(g, lim)) * clamp(u, -lim, lim), composed from two upstream
    HCU-safe triton kernels — sglang's silu_and_mul_clamp (identical clamp
    semantics) plus per-token-group-128 fp8 quant when the caller asks for
    fp8 output (`dtype_quant` set, e.g. the Glm5Next AIS MoE epilogue).

    Returns `out [M, N/2]` in bf16, or `(out_fp8, scale [M, N/128] fp32)`
    matching the plain (non-bpreshuffle) aiter layout the downstream
    block-fp8 `down_proj` tuple input expects.
    """
    from sglang.kernels.ops.activation.silu_and_mul_clamp_hip import (
        silu_and_mul_clamp_triton,
    )

    out = silu_and_mul_clamp_triton(
        gate_up, float(swiglu_limit or 0.0), fp8_grid=False, emit_fp8=False
    )
    if dtype_quant is None:
        return out
    fp8_dtype = torch.float8_e4m3fn
    # The DTK wheel's aiter.dtypes.fp8 is torch.uint8 (its kernels read/write
    # fp8 bytes through uint8 views); any uint8 request means fp8 output here.
    if isinstance(dtype_quant, torch.dtype) and dtype_quant not in (
        fp8_dtype,
        torch.uint8,
    ) and dtype_quant not in _FP8_DTYPES:
        raise NotImplementedError(
            f"hcu-adapter 'aiter': fused_clamp_act_mul dtype_quant={dtype_quant}"
            " (only e4m3 fp8 is supported on HCU)"
        )
    from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_8bit

    q, s = per_token_group_quant_8bit(out, 128, fp8_dtype)
    return q, s


class _AiterDtypes:
    """`aiter.dtypes` fill-in for top-level imports when the real package
    doesn't export it; HCU targets e4m3 (fn), not ROCm's fnuz."""

    fp8 = torch.float8_e4m3fn
    fp8_e4m3 = torch.float8_e4m3fn
    bf16 = torch.bfloat16
    fp32 = torch.float32
    int32 = torch.int32


def _hcu_cp_gather_indexer_k_quant_cache(
    kv_cache, dst_k, dst_scale, page_indices, cu_seq_lens, preshuffle=False, **kw
):
    """aiter `cp_gather_indexer_k_quant_cache` (missing from the DTK wheel):
    gather per-token fp8 index-K rows (+scales) out of the paged indexer cache
    into flat [seq_len_sum, ...] buffers. Semantics identical to sglang's own
    `GetKAndS.triton` gather (_get_k_and_s_triton) — same paged layout, same
    outputs — so we delegate to it and reshape to this function's ABI.
    kv_cache: [pages, page_size, 128B + 4B] viewed fp8; page_indices: [reqs,
    pages] (aiter's is flattened per-request; sglang passes [1, pages]);
    cu_seq_lens: [reqs+1] i32 cumulative pooled lengths.
    """
    from sglang.kernels.ops.attention.dsa.index_buf_accessor import (
        _get_k_and_s_triton,
    )

    assert dst_k.shape[0] >= int(cu_seq_lens[-1].item()) if cu_seq_lens.numel() > 1 else True
    pages = page_indices if page_indices.dim() == 2 else page_indices.view(1, -1)
    # aiter semantics: concatenate each request's pages and keep the first
    # cu_seq_lens[-1] tokens — with a single request (or identical flat
    # ordering) page order == token order, so the flat gather matches.
    seq_lens = (cu_seq_lens[1:] - cu_seq_lens[:-1]).to(torch.int64)
    seq_len_sum = int(dst_k.shape[0])
    k_out, s_out = _get_k_and_s_triton(
        buf=kv_cache.view(kv_cache.shape[0], -1).view(torch.uint8),
        page_indices=pages,
        seq_lens=seq_lens,
        seq_len_sum=seq_len_sum,
        max_seq_len=int(seq_lens.max().item()) if seq_lens.numel() else 0,
        page_size=kv_cache.shape[1],
        index_head_dim=128,
    )
    dst_k.copy_(k_out)
    dst_scale.copy_(s_out)


def _extra(fullname):
    if fullname == "aiter.ops.triton.utils.types":
        return {"e4m3_dtype": torch.float8_e4m3fn}
    if fullname == "aiter.ops.triton.utils._triton.kernel_repr":
        return {"make_kernel_repr": lambda *a, **k: (lambda *ca, **ck: "")}
    if fullname == "aiter.ops.triton.fusions.fused_clamp_act_mul":
        return {"fused_clamp_act_mul": _hcu_fused_clamp_act_mul}
    if fullname == "aiter.ops.cache":
        return {"cp_gather_indexer_k_quant_cache": _hcu_cp_gather_indexer_k_quant_cache}
    if fullname in (
        "aiter.ops.triton.gemm",
        "aiter.ops.triton.gemm.fused",
        "aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat",
    ):
        # The quark W4A4-MXFP4 scheme top-level-imports three kernels under
        # this subpackage plus fused_gemm_afp4wfp4_split_cat. The DTK wheel
        # ships the three plain ones as TOP-LEVEL modules (no gemm/ package);
        # only split_cat is missing. Re-export the real ones so the import
        # chain clears, and give split_cat the loud LibStub default (model
        # registry scans of quark-quantized models stay warning-free; actual
        # use raises loudly with the gap name).
        out = {}
        try:
            import importlib

            for sub, names in (
                ("aiter.ops.triton.gemm_afp4wfp4",
                 ("gemm_afp4wfp4",)),
                ("aiter.ops.triton.gemm_afp4wfp4_pre_quant_atomic",
                 ("gemm_afp4wfp4_pre_quant",)),
                ("aiter.ops.triton.quant",
                 ("dynamic_mxfp4_quant",)),
            ):
                try:
                    real = importlib.import_module(sub)
                    for n in names:
                        if hasattr(real, n):
                            out[n] = getattr(real, n)
                except Exception:
                    pass
        except Exception:
            pass
        return out
    if fullname == "aiter.ops.triton.pa_mqa_logits":
        # Newer aiter moved pa_mqa_logits under ops.triton.attention; sglang
        # still imports the old path. Re-export the wheel's real kernels —
        # guarded: re-entrant imports inside find/create must never raise
        # (missing on both paths just leaves the loud Missing stub).
        try:
            import importlib

            real = importlib.import_module(
                "aiter.ops.triton.attention.pa_mqa_logits")
            return {
                name: getattr(real, name)
                for name in ("deepgemm_fp8_paged_mqa_logits",)
                if hasattr(real, name)
            }
        except Exception:
            return {}
    if fullname == "aiter.ops.triton.fp8_mqa_logits":
        try:
            import importlib

            real = importlib.import_module(
                "aiter.ops.triton.attention.fp8_mqa_logits")
            return {
                name: getattr(real, name)
                for name in ("fp8_mqa_logits",)
                if hasattr(real, name)
            }
        except Exception:
            return {}
    if fullname == "aiter":
        return {"silu_and_mul": _hcu_silu_and_mul, "dtypes": _AiterDtypes}
    return {}


class _AiterFinder(LibStubFinder):
    def __init__(self):
        super().__init__("aiter")

    def _extra_attrs(self, fullname):
        return _extra(fullname)

    # kill-switch guards on REAL wheel attrs (only active while the op is in
    # HCU_DISABLE_OPS): flash attention is a historically flaky vendor surface
    # (flexible sizes, many capture scenarios); blocking its entries makes
    # sglang fail loudly instead of silently running a suspect kernel.
    def _guard_attrs(self, fullname):
        if fullname == "aiter":
            return {
                "flash_attn_varlen_func": "aiter_fa",
                "flash_attn_func": "aiter_fa",
                "flash_attn_with_kvcache": "aiter_fa",
            }
        if fullname in (
            "aiter.ops.triton.attention",
            "aiter.ops.triton.attention.attn_fwd",
            "aiter.ops.triton.pa_mqa_logits",
            "aiter.ops.triton.attention.pa_mqa_logits",
        ):
            return {
                "attn_fwd": "aiter_fa",
                "mha_fwd": "aiter_fa",
                "flash_attn_varlen_func": "aiter_fa",
            }
        return {}


def register() -> None:
    if not is_registered(_AiterFinder, "aiter"):
        import sys

        sys.meta_path.insert(0, _AiterFinder())
