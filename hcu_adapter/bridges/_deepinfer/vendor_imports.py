"""Fill missing pinned hard-import symbols; never return fake numerical data."""

import importlib
import sys
import types

from . import hcu_env, registry
from hcu_adapter.ops import selector

_DEEP_GEMM_RUNTIME = (
    ("set_num_sms", "deep_gemm_set_num_sms"),
    ("get_num_sms", "deep_gemm_get_num_sms"),
    ("set_mk_alignment_for_contiguous_layout", "set_mk_alignment_for_contiguous_layout"),
    ("get_mk_alignment_for_contiguous_layout", "get_mk_alignment_for_contiguous_layout"),
    ("get_theoretical_mk_alignment_for_contiguous_layout",
     "get_theoretical_mk_alignment_for_contiguous_layout"),
    ("get_tma_aligned_size", "get_tma_aligned_size"),
)


def _missing(symbol):
    def unavailable(*args, **kwargs):
        raise NotImplementedError("Unimplemented HCU vendor symbol: " + symbol)

    unavailable._hcu_unimplemented = True
    return unavailable


_GROUPED_GEMM = {"torch": "hcu_adapter.kernels.grouped_gemm_emu", "triton": "hcu_adapter.kernels.grouped_gemm_triton"}


def install(grouped_gemm_backend="torch"):
    """grouped_gemm_backend selects the m_grouped_fp8_fp4_gemm_nt_contiguous provider explicitly."""
    if grouped_gemm_backend not in _GROUPED_GEMM:
        raise ValueError("grouped_gemm_backend must be one of " + ", ".join(sorted(_GROUPED_GEMM)))
    import deep_gemm
    import flash_mla

    from hcu_adapter.kernels.weight_scale_emu import transform_sf_into_required_layout
    from hcu_adapter.kernels.mhc_emu import tf32_hc_prenorm_gemm
    from hcu_adapter.kernels.gemm import cublaslt_gemm_nt

    current = getattr(deep_gemm, "cublaslt_gemm_nt", None)
    if current is not None and current is not cublaslt_gemm_nt:
        raise RuntimeError("An existing dense NT GEMM provider is already installed")
    deep_gemm.cublaslt_gemm_nt = cublaslt_gemm_nt
    registry.register("deep_gemm.cublaslt_gemm_nt", "hcu_adapter.kernels.gemm.cublaslt_gemm_nt",
                      "deep_gemm.cublaslt_gemm_nt", "seam")

    existing = getattr(deep_gemm, "transform_sf_into_required_layout", None)
    if existing is not None and existing is not transform_sf_into_required_layout:
        raise RuntimeError("An existing DeepGEMM scale-layout provider is already installed")
    deep_gemm.transform_sf_into_required_layout = transform_sf_into_required_layout
    registry.register("deep_gemm.transform_sf_into_required_layout",
                      "hcu_adapter.kernels.weight_scale_emu.transform_sf_into_required_layout",
                      "deep_gemm.transform_sf_into_required_layout", "seam")
    hc_gemm = getattr(deep_gemm, "tf32_hc_prenorm_gemm", None)
    if hc_gemm is not None and hc_gemm is not tf32_hc_prenorm_gemm:
        raise RuntimeError("An existing DeepGEMM mHC prenorm provider is already installed")
    deep_gemm.tf32_hc_prenorm_gemm = tf32_hc_prenorm_gemm
    bf16_mega_gate = selector.provider("deep_gemm.bf16_mega_gate",
                                       "lightop" if hcu_env.lightop_ops_enabled() else "torch")

    current = getattr(deep_gemm, "bf16_mega_gate", None)
    if current is not None and current is not bf16_mega_gate:
        raise RuntimeError("An existing DeepGEMM bf16_mega_gate provider is already installed")
    deep_gemm.bf16_mega_gate = bf16_mega_gate
    registry.register("deep_gemm.bf16_mega_gate", bf16_mega_gate.__module__ + '.bf16_mega_gate',
                      "deep_gemm.bf16_mega_gate", "seam")
    mega_mhc = selector.provider("deep_gemm.mega_mhc",
                                 "lightop" if hcu_env.lightop_ops_enabled() else "torch")

    current = getattr(deep_gemm, "mega_mhc", None)
    if current is not None and current is not mega_mhc:
        raise RuntimeError("An existing DeepGEMM mega_mhc provider is already installed")
    deep_gemm.mega_mhc = mega_mhc
    registry.register("deep_gemm.mega_mhc", mega_mhc.__module__ + '.mega_mhc', "deep_gemm.mega_mhc", "seam")
    registry.register("deep_gemm.tf32_hc_prenorm_gemm",
                      "hcu_adapter.kernels.mhc_emu.tf32_hc_prenorm_gemm",
                      "deep_gemm.tf32_hc_prenorm_gemm", "seam")
    from hcu_adapter.kernels import runtime_config

    for name, impl in _DEEP_GEMM_RUNTIME:
        provider = getattr(runtime_config, impl)
        current = getattr(deep_gemm, name, None)
        if current is not None and current is not provider:
            raise RuntimeError("An existing DeepGEMM runtime provider is already installed: " + name)
        setattr(deep_gemm, name, provider)
        registry.register("deep_gemm." + name, "hcu_adapter.kernels.runtime_config." + impl,
                          "deep_gemm." + name, "seam")
    for name in ("fp8_fp4_gemm_nt", "fp8_einsum", "m_grouped_fp8_fp4_gemm_nt_contiguous"):
        module = _GROUPED_GEMM[grouped_gemm_backend] if name.startswith("m_grouped") else "hcu_adapter.kernels.gemm_emu"
        provider = getattr(importlib.import_module(module), name)
        current = getattr(deep_gemm, name, None)
        if current is not None and current is not provider:
            raise RuntimeError("An existing DeepGEMM provider is already installed: " + name)
        setattr(deep_gemm, name, provider)
        registry.register("deep_gemm." + name, module + "." + name, "deep_gemm." + name, "seam")
    print("HCU adapter runtime: deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous -> " + grouped_gemm_backend
          + " (candidate)", flush=True)
    if not hasattr(flash_mla, "fused_norm_rope_attn_rope_cast"):
        flash_mla.fused_norm_rope_attn_rope_cast = types.SimpleNamespace(**{
            name: _missing("flash_mla.fused_norm_rope_attn_rope_cast." + name)
            for name in ("prefill", "decode", "permute_q_b_proj", "permute_wv_proj")
        })
    from hcu_adapter.kernels.mla_layout_emu import permute_q_b_proj, permute_wv_proj

    for name, provider in (("permute_q_b_proj", permute_q_b_proj),
                           ("permute_wv_proj", permute_wv_proj)):
        op = "flash_mla.fused_norm_rope_attn_rope_cast." + name
        current = getattr(flash_mla.fused_norm_rope_attn_rope_cast, name, None)
        if current is not None and current is not provider and not getattr(current, "_hcu_unimplemented", False):
            raise RuntimeError("An existing FlashMLA projection provider is already installed: " + name)
        setattr(flash_mla.fused_norm_rope_attn_rope_cast, name, provider)
        registry.register(op, "hcu_adapter.kernels.mla_layout_emu." + name, op, "seam")
    try:
        importlib.import_module("flash_attn.cute.interface")
    except ModuleNotFoundError as error:
        if error.name not in ("flash_attn", "flash_attn.cute", "flash_attn.cute.interface"):
            raise
        for name in ("flash_attn", "flash_attn.cute", "flash_attn.cute.interface"):
            if name in sys.modules:
                continue
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
            parent, _, child = name.rpartition(".")
            if parent:
                setattr(sys.modules[parent], child, module)
        sys.modules["flash_attn.cute.interface"]._flash_attn_fwd = _missing("flash_attn.cute.interface._flash_attn_fwd")
    from . import vision_attention

    vision_attention.install()
    from hcu_adapter.kernels import mla_attention_emu as mla_attention
    from hcu_adapter.kernels.mla_attention_emu import fused_prefill

    for name in ("fp8_fp4_mqa_logits", "fp8_fp4_sparse_mqa_logits", "get_sparse_mqa_logits_metadata",
                 "fp8_fp4_paged_mqa_logits", "get_paged_mqa_logits_metadata",
                 "fp8_fp4_paged_sparse_mqa_logits", "get_paged_sparse_mqa_logits_metadata"):
        provider = getattr(mla_attention, name)
        if (hcu_env.lightop_ops_enabled() and name in (
                'fp8_fp4_mqa_logits', 'fp8_fp4_sparse_mqa_logits',
                'fp8_fp4_paged_mqa_logits', 'fp8_fp4_paged_sparse_mqa_logits')):
            from hcu_adapter.kernels import lightop_mqa

            provider = getattr(lightop_mqa, name)
        current = getattr(deep_gemm, name, None)
        if current is not None and current is not provider:
            raise RuntimeError("An existing DeepGEMM MQA logits provider is already installed: " + name)
        setattr(deep_gemm, name, provider)
        registry.register("deep_gemm." + name, provider.__module__ + '.' + name, "deep_gemm." + name, "seam")

    from hcu_adapter.kernels.mla_attention_emu import fused_decode
    if hcu_env.native_flashmla_enabled():
        from lightop.version import git_hash as lightop_commit
        from flash_mla.version import git_hash as flash_commit
        import flash_mla.cuda as native_mla

        if (lightop_commit != 'b2f8c9d' or flash_commit != '1287a01'
                or not hasattr(native_mla, 'sparse_prefill_fwd_fp32')):
            raise RuntimeError('Native FlashMLA requires the pinned LightOp and optional FP32 API')
        from hcu_adapter.kernels.native_flashmla import fused_prefill, fused_decode

    fused = flash_mla.fused_norm_rope_attn_rope_cast
    for name, provider in (("prefill", fused_prefill), ("decode", fused_decode)):
        current = getattr(fused, name, None)
        if current is not None and current is not provider and not getattr(current, "_hcu_unimplemented", False):
            raise RuntimeError("An existing FlashMLA fused provider is already installed: " + name)
        setattr(fused, name, provider)
        op = "flash_mla.fused_norm_rope_attn_rope_cast." + name
        registry.register(op, provider.__module__ + '.fused_' + name, op, "seam")
    registry.register("vendors.import_surface", "hcu_adapter.bridges._deepinfer.vendor_imports.install",
                      "flash_mla/deep_gemm/flash_attn hard imports", "seam")
