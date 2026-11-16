# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""V4.1 PCP attention classes and admission rules.

Moved here from the plugin: PCP preparation, the owned backends and the local
scoring contract are upstream-defined behaviour, so they belong to the adapter
that already owns the V4.1 HCU seams. The plugin keeps only its private model
publication and injects the FP8 reference scorer through `set_indexer_reference`.

Imports stay framework-free at module load; every vLLM symbol is resolved inside
`build_classes()` so `hcu_adapter` can be imported without vLLM present.
"""
from __future__ import annotations

import threading

from hcu_adapter import hcu_env
from hcu_adapter.kill_switch import op_disabled

# Same operator-facing knob the bridge records and health-checks. `select_attention`
# must read it too: that function decides which class actually runs, so a `stock`
# mode or a kill switch reaching only the bookkeeping would leave the documented
# rollback path inert while the owned PCP classes were still built and executed.
MODE_VARIABLE = 'HCU_VLLM_PCP_IMPL'
OP_NAME = 'vllm_pcp'

_lock = threading.RLock()
_reference = None
_engram_validator = None
_classes = None
_forward_context_resolver = None


class PcpAttentionError(RuntimeError):
    """V4.1 PCP attention is unavailable or its contracts drifted."""


def selected_mode() -> str:
    """Effective PCP mode: `native` runs the owned classes, `stock` does not.

    The kill switch wins over the mode variable, matching `bridges/vllm_pcp.py`;
    a second, divergent reading of the same knob is what the review caught.
    """
    if op_disabled(OP_NAME):
        return 'stock'
    value, _ = hcu_env.resolve(MODE_VARIABLE, host='vllm')
    if value not in ('stock', 'native'):
        raise PcpAttentionError(f'{MODE_VARIABLE} must be stock or native')
    return value


def set_indexer_reference(function) -> None:
    """Bind the plugin-owned FP8 reference scorer used by the PCP indexer.

    The direction rule is plugin -> adapter only, so the adapter cannot import
    the plugin's FP8 indexer module. The plugin binds its scorer during
    activation; scoring without a bound scorer is a refusal, never a silent
    fallback.
    """
    global _reference
    if not callable(function):
        raise PcpAttentionError("PCP indexer reference must be callable")
    with _lock:
        if _reference is not None and _reference is not function:
            raise PcpAttentionError(
                "PCP indexer reference already bound to another owner; restart process"
            )
        _reference = function


def set_engram_validator(function) -> None:
    """Bind the plugin-owned Engram config validator (adapter cannot import it)."""
    global _engram_validator
    if not callable(function):
        raise PcpAttentionError("PCP engram validator must be callable")
    with _lock:
        if _engram_validator is not None and _engram_validator is not function:
            raise PcpAttentionError(
                "PCP engram validator already bound to another owner; restart process"
            )
        _engram_validator = function


def set_forward_context_resolver(function) -> None:
    """Bind the host forward-context reader.

    The plugin re-exported `get_forward_context` and the CPU replay patches it
    there; once the attention class lives in the adapter, that patch would no
    longer reach it. The resolver keeps a single, explicit seam.
    """
    global _forward_context_resolver
    if not callable(function):
        raise PcpAttentionError("PCP forward-context resolver must be callable")
    with _lock:
        if (_forward_context_resolver is not None
                and _forward_context_resolver is not function):
            raise PcpAttentionError(
                "PCP forward-context resolver already bound; restart process"
            )
        _forward_context_resolver = function


def _forward_context():
    resolver = _forward_context_resolver
    if resolver is not None:
        return resolver()
    from vllm.forward_context import get_forward_context

    return get_forward_context()


def bound_reference():
    """Read-only: the scorer currently bound, or None. Never binds or repairs."""
    return _reference


def bound_engram_validator():
    """Read-only: the Engram validator currently bound, or None."""
    return _engram_validator


def _require_reference():
    reference = _reference
    if reference is None:
        raise PcpAttentionError(
            "PCP indexer reference is not bound; activate the HCU plugin first"
        )
    return reference


def _require_engram_validator():
    validator = _engram_validator
    if validator is None:
        raise PcpAttentionError(
            "PCP Engram validator is not bound; activate the HCU plugin first"
        )
    return validator


def _field(config, name, default=None):
    from collections.abc import Mapping

    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def is_v41(hf_config) -> bool:
    """Recognize the V4.1 family from a resolved HF config.

    Local copy of the plugin predicate: the adapter may not import the plugin,
    and this depends only on configuration fields.
    """
    for config in (hf_config, _field(hf_config, "text_config")):
        if config is None:
            continue
        if _field(config, "model_type") in ("deepseek_v41", "deepseek_v41_text"):
            return True
        if "DeepseekV41ForCausalLM" in (_field(config, "architectures") or ()):
            return True
    return False


def validate_config(cfg) -> bool:
    """Validate the native PCP structure before allocating model parameters.

    Width is unconstrained. PP/DP/DCP=1 follows the native V41 PCP contract;
    speculative decoding needs a larger CR2 ring contract than Core provides.
    Engram uses the plugin-owned private registered-host TPxPCP table adapter,
    so its validator is injected rather than imported.
    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    p = cfg.parallel_config
    if p.prefill_context_parallel_size == 1:
        return False
    errors = []
    if not is_v41(cfg.model_config.hf_config):
        errors.append("V41 model family required")
    if p.prefill_context_parallel_size < 2:
        errors.append("positive PCP width required")
    if not cfg.model_config.enforce_eager:
        errors.append("eager execution required by host-driven reference scoring")
    for name in ("pipeline_parallel_size", "data_parallel_size", "decode_context_parallel_size"):
        if getattr(p, name) != 1:
            errors.append(f"{name}=1 required by native V41 PCP metadata")
    if cfg.speculative_config is not None:
        errors.append("native PCP CR2 ring does not support speculative decoding")
    if cfg.lora_config is not None or p.enable_eplb:
        errors.append("LoRA/EPLB are not supported by this PCP model adapter")
    if p.enable_dbo:
        errors.append("DBO would change per-step metadata/collective ordering")
    if not cfg.use_v2_model_runner:
        errors.append("model runner V2 PCP metadata required")
    if cfg.kernel_config.monokernel:
        errors.append("monokernel bypasses common PCP preparation")
    if cfg.kv_transfer_config is not None:
        errors.append("KV transfer requires a separate replicated-cache contract")
    # SWA bounded replay: native decoder replay pads each rank's replay batch to
    # the global kept rows and rebuilds PCP metadata for it; this attention reads
    # only that metadata. Prefix-cache SWA replay stays off on ROCm natively.
    if getattr(cfg.model_config.hf_config, "engram_layer_ids", None):
        if not hasattr(cfg, "engram_config"):
            errors.append("HCU PCP Engram requires resolved Engram config")
        else:
            try:
                _require_engram_validator()(cfg)
            except PcpAttentionError as error:
                errors.append(str(error))
            except ValueError as error:
                errors.append(str(error))
    if (cfg.attention_config.resolve_indexer_kv_dtype("fp8") != "fp8"
            or cfg.attention_config.indexer_sparse_logits):
        errors.append("source-owned FP8 indexer with dense logits required")
    if cfg.attention_config.backend not in (
        None, AttentionBackendEnum.ROCM_FLASHMLA_SPARSE_DSV4,
        AttentionBackendEnum.ROCM_FLASHMLA_SPARSE_DSV41,
    ):
        errors.append("ROCM_FLASHMLA_SPARSE_DSV41 backend required")
    if errors:
        raise ValueError("HCU V41 PCP attention: " + "; ".join(errors))
    return True


def build_classes():
    """Resolve and cache the owned PCP classes against the imported vLLM.

    The plugin publishes these through `ModelRegistry`, so the classes must be
    created in the process that will run the model, after vLLM is importable.
    Build once; a second caller with a different vLLM module identity refuses
    instead of silently mixing two class hierarchies.
    """
    global _classes
    with _lock:
        if _classes is not None:
            return _classes

        import torch
        from vllm.config import get_current_vllm_config
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
        from vllm.models.deepseek_v41.amd.rocm import (
            DeepseekV41ROCMAiterMLAAttention,
            DeepseekV41ROCMAiterMLASparseBackend,
            DeepseekV41ROCMAiterSparseSWABackend,
        )
        from vllm.models.deepseek_v41.attention import DeepseekV4Attention, DeepseekV4Indexer

        class HcuV41PcpBackend(DeepseekV41ROCMAiterMLASparseBackend):
            @classmethod
            def supports_pcp(cls) -> bool:
                return True

        class HcuV41PcpSwaBackend(DeepseekV41ROCMAiterSparseSWABackend):
            @classmethod
            def supports_pcp(cls) -> bool:
                return True

        class HcuV41PcpSparseIndexer(SparseAttnIndexer):
            """Local scoring over the full cache replicated by common V41 sources.

            This is an owned layer selected by an owned model, independent of the
            optional PCP1 OOT scorer. Never falsify the native parallel configuration.
            """

            def __init__(self, *args, **kwargs):
                import inspect

                validate_config(get_current_vllm_config())
                bound = inspect.signature(SparseAttnIndexer.__init__).bind(self, *args, **kwargs)
                bound.apply_defaults()
                values = bound.arguments
                if (not values["skip_k_cache_insert"] or values["use_fp4_cache"]
                        or values["head_dim"] != 128 or values["compress_ratio"] not in (1, 2)):
                    raise ValueError("PCP indexer requires source-owned FP8 K, D128, CR1/2")
                super().__init__(*args, **kwargs)

            def forward_hip(self, hidden_states, q_quant, k, weights):
                if k is not None or not isinstance(q_quant, torch.Tensor):
                    raise ValueError("PCP scoring expects one FP8 Q and source-owned K")
                return _require_reference()(
                    hidden_states, self.k_cache.prefix, self.k_cache.kv_cache, q_quant,
                    weights, self.topk_tokens, self.head_dim, self.max_model_len,
                    self.max_total_seq_len, self.topk_indices_buffer, self.compress_ratio,
                    self.candidate_blocks, self.candidate_block_size, self.candidate_write,
                )

            # CustomOp disables hardware dispatch when the user excludes its op name.
            # The owned model still needs the same scorer in that configuration.
            forward_native = forward_hip

        class HcuV41PcpIndexer(DeepseekV4Indexer):
            attn_cls = HcuV41PcpSparseIndexer

        class HcuV41PcpAttention(DeepseekV41ROCMAiterMLAAttention):
            backend_cls = HcuV41PcpBackend
            swa_backend_cls = HcuV41PcpSwaBackend

            def __init__(self, *args, **kwargs):
                cfg = args[0] if args else kwargs["vllm_config"]
                if not validate_config(cfg):
                    raise ValueError("owned PCP attention requires PCP greater than one")
                super().__init__(*args, **kwargs)
                # Common preparation serializes PCP gathers. AMD's fast compressor
                # chain publishes local latent against global slots and cannot run here.
                self.aux_stream_list = None

            def _indexer_cls(self, k_cache):
                if k_cache is not None and k_cache.rocm_mxfp4:
                    raise ValueError("PCP adapter requires the FP8 SHUFFLE index K layout")
                return HcuV41PcpIndexer

            forward = DeepseekV4Attention.forward
            _alloc_attn_out = DeepseekV4Attention._alloc_attn_out

            def forward_mqa(self, q, kv, positions, output):
                if not isinstance(output, torch.Tensor):
                    raise TypeError("HCU PCP attention requires its BF16 output allocation")
                metadata = _forward_context().attn_metadata
                if metadata is None:
                    return super().forward_mqa(q, kv, positions, output)
                if not isinstance(metadata, dict) or self.swa_cache_layer.prefix not in metadata:
                    raise RuntimeError("HCU PCP attention requires native local SWA metadata")
                swa = metadata[self.swa_cache_layer.prefix]
                count = swa.num_decode_tokens + swa.num_prefill_tokens
                if count < 0 or count > q.shape[0] or q.shape[0] != output.shape[0]:
                    raise ValueError("PCP local query/output extent mismatch")
                # Collectives already ran during preparation, including on empty ranks.
                # AMD inverse RoPE sees only initialized, actual local query rows.
                output[count:].zero_()
                if count:
                    return super().forward_mqa(
                        q[:count], kv[:count], positions[:count], output[:count]
                    )

            def _forward_decode(self, q, *args, **kwargs):
                # A native PCP placeholder may retain a request with zero actual tokens.
                if q.shape[0] == 0:
                    return 0
                return super()._forward_decode(q, *args, **kwargs)

        _classes = dict(
            attention=HcuV41PcpAttention,
            indexer=HcuV41PcpIndexer,
            sparse_indexer=HcuV41PcpSparseIndexer,
            backend=HcuV41PcpBackend,
            swa_backend=HcuV41PcpSwaBackend,
        )
        return _classes


def select_attention(cfg):
    """Return the owned attention class for a PCP config, else the native one.

    Admission is two gates, not one: the deployment has to look like PCP *and* the
    operator has to have selected the owned implementation. Without the second gate
    `stock` (the default) and `HCU_DISABLE_OPS=vllm_pcp` would be inert, because
    this function -- not the bridge's bookkeeping -- is what picks the class.
    """
    from vllm.models.deepseek_v41.amd import model as native

    if selected_mode() == 'stock' or not validate_config(cfg):
        return native._select_dsv4_attn_cls(cfg)
    return build_classes()["attention"]
