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

from hcu_adapter.infra import hcu_env
from hcu_adapter.infra.kill_switch import op_disabled

# Same operator-facing knob the bridge records and health-checks. `select_attention`
# must read it too: that function decides which class actually runs, so a `stock`
# mode or a kill switch reaching only the bookkeeping would leave the documented
# rollback path inert while the owned PCP classes were still built and executed.
MODE_VARIABLE = 'HCU_VLLM_PCP_IMPL'
OP_NAME = 'vllm_pcp'
Q_ROPE_OP = 'vllm_pcp_q_rope'

_lock = threading.RLock()
_reference = None
_engram_validator = None
_classes = None
_forward_context_resolver = None
_q_rope_provider = None
_q_rope_binding = None
_q_rope_failure = None
_forward_mqa_provider = None
_kv_transfer_validator = None


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


def q_rope_selected() -> bool:
    from .pcp_rope_policy import selected_mode as rope_mode

    return rope_mode(Q_ROPE_OP) == "hcu"


def _refuse_q_rope_change():
    global _q_rope_failure
    _q_rope_failure = "PCP Q RoPE policy/provider changed after binding; restart process"
    raise PcpAttentionError(_q_rope_failure)


def require_q_rope_healthy() -> None:
    """Validate the published child policy without rebinding it."""
    with _lock:
        if _q_rope_failure is not None:
            raise PcpAttentionError(_q_rope_failure)
        if _q_rope_binding is None:
            return
        selected, provider = _q_rope_binding
        try:
            current = q_rope_selected()
        except BaseException:
            _refuse_q_rope_change()
        if current != selected or _q_rope_provider is not provider:
            _refuse_q_rope_change()


def set_q_rope_provider(provider) -> None:
    """Seal the effective policy and bind its worker startup initializer.

    Stock uses no provider. The HCU initializer receives the Core cos/sin
    table during dummy profiling and returns an input preparation callable.
    """
    global _q_rope_provider, _q_rope_binding
    if provider is not None and not callable(provider):
        raise PcpAttentionError("PCP Q RoPE provider must be callable")
    with _lock:
        require_q_rope_healthy()
        if _q_rope_binding is not None:
            if _q_rope_provider is not provider:
                _refuse_q_rope_change()
            return
        selected = q_rope_selected()
        if selected != (provider is not None):
            raise PcpAttentionError("PCP Q RoPE provider does not match the selected policy")
        _q_rope_provider = provider
        _q_rope_binding = (selected, provider)


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


def set_kv_transfer_validator(function) -> None:
    """Bind an optional extension's transfer contract; default is refusal."""
    global _kv_transfer_validator
    if not callable(function):
        raise PcpAttentionError("PCP transfer validator must be callable")
    with _lock:
        if _kv_transfer_validator is not None and _kv_transfer_validator is not function:
            raise PcpAttentionError("PCP transfer validator already bound; restart process")
        _kv_transfer_validator = function


def bound_kv_transfer_validator():
    return _kv_transfer_validator


def validate_kv_transfer_config(cfg):
    """Common PCP transfer gate, also used by specialized cache-format owners."""
    if cfg.kv_transfer_config is None:
        return
    if _kv_transfer_validator is None:
        raise PcpAttentionError('KV transfer requires a separate replicated-cache contract')
    if _kv_transfer_validator(cfg) is not None:
        raise PcpAttentionError('PCP transfer validator must return None')


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


def set_forward_mqa_provider(provider):
    """Bind an optional forward_mqa provider before the owned classes are built."""
    global _forward_mqa_provider
    if not callable(provider):
        raise PcpAttentionError("PCP forward_mqa provider must be callable")
    with _lock:
        if _forward_mqa_provider is provider:
            return
        if _classes is not None or _forward_mqa_provider is not None:
            raise PcpAttentionError("PCP forward_mqa provider already bound; restart process")
        _forward_mqa_provider = provider


def bound_forward_mqa_provider():
    """Return the optional forward_mqa provider without constructing or changing classes."""
    return _forward_mqa_provider


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
    DSpark uses the owned CR2 history metadata for verification rollback.
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
        from .pcp_dspark import validate_dspark_config
        validate_dspark_config(cfg)
    if cfg.lora_config is not None or p.enable_eplb:
        errors.append("LoRA/EPLB are not supported by this PCP model adapter")
    if p.enable_dbo:
        errors.append("DBO would change per-step metadata/collective ordering")
    if not cfg.use_v2_model_runner:
        errors.append("model runner V2 PCP metadata required")
    if cfg.kernel_config.monokernel:
        errors.append("monokernel bypasses common PCP preparation")
    if cfg.kv_transfer_config is not None:
        try:
            validate_kv_transfer_config(cfg)
        except (ValueError, PcpAttentionError) as error:
            errors.append(str(error))
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
    from hcu_adapter.features.dsv41.fp4_indexer import applicable as fp4_applicable, validate_config as validate_fp4
    if fp4_applicable(cfg):
        # The FP4 owner supplies its producer/cache/reader while PCP preserves
        # replicated K preparation and rank-local query metadata.
        validate_fp4(cfg)
    elif (cfg.attention_config.resolve_indexer_kv_dtype("fp8") != "fp8"
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

        def forward_mqa_impl(self, q, kv, positions, output):
            return DeepseekV41ROCMAiterMLAAttention.forward_mqa(
                self, q, kv, positions, output)

        if _forward_mqa_provider is not None:
            forward_mqa_impl = _forward_mqa_provider(DeepseekV41ROCMAiterMLAAttention)
            if not callable(forward_mqa_impl):
                raise PcpAttentionError("PCP provider must supply a callable forward_mqa")

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
                self._initialize_q_rope = None
                self._prepare_q_rope = None
                require_q_rope_healthy()
                if q_rope_selected() and cfg.model_config.enforce_eager:
                    if _q_rope_provider is None:
                        raise PcpAttentionError("PCP Q RoPE provider is not bound")
                    self._initialize_q_rope = _q_rope_provider

            def _indexer_cls(self, k_cache):
                if k_cache is not None and k_cache.rocm_mxfp4:
                    raise ValueError("PCP adapter requires the FP8 SHUFFLE index K layout")
                return HcuV41PcpIndexer

            forward = DeepseekV4Attention.forward
            _alloc_attn_out = DeepseekV4Attention._alloc_attn_out

            def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
                if _q_rope_failure is not None:
                    raise PcpAttentionError(_q_rope_failure)
                native = super()._fused_qnorm_rope_kv_insert
                if ((self._initialize_q_rope is None and self._prepare_q_rope is None)
                        or not self.use_pcp or self.kv_mxfp8
                        or self.kv_cache_torch_dtype != torch.uint8
                        or self.accepts_unnormed_unroped_query
                        or self.n_local_heads != self.padded_heads
                        or self.rope_head_dim != 64 or self.rotary_emb.is_neox_style):
                    return native(q, kv, positions, attn_metadata)

                if attn_metadata is None and self._initialize_q_rope is not None:
                    # Core sizes the automatic KV cache after this dummy pass.
                    self._prepare_q_rope = self._initialize_q_rope(
                        self.rotary_emb.cos_sin_cache
                    )
                    self._initialize_q_rope = None
                if self._prepare_q_rope is None or not isinstance(attn_metadata, dict):
                    return native(q, kv, positions, attn_metadata)

                swa = attn_metadata.get(self.swa_cache_layer.prefix)
                cache = self.swa_cache_layer.kv_cache
                if (swa is None or swa.num_prefill_tokens <= 0 or swa.num_decode_tokens
                        or swa.cache_slot_mapping is None or swa.cache_positions is None
                        or cache.dtype != torch.uint8 or cache.ndim != 3
                        or cache.shape[-1] != 584):
                    return native(q, kv, positions, attn_metadata)

                cos_sin = self.rotary_emb.cos_sin_cache
                rotate = self._prepare_q_rope(q, positions, cos_sin)
                if rotate is None:
                    return native(q, kv, positions, attn_metadata)

                from vllm.distributed import get_pcp_group

                # Preserve the Core PCP KV-collective order and original RoPE table.
                gathered_kv = get_pcp_group().all_gather(kv.contiguous(), dim=0)
                torch.ops._C.fused_deepseek_v4_kv_rope_insert(
                    gathered_kv, cache, swa.cache_slot_mapping, swa.cache_positions,
                    cos_sin, swa.block_size, None, self.kv_mxfp8,
                )
                rotate()
                return q

            def forward_mqa(self, q, kv, positions, output):
                if not isinstance(output, torch.Tensor):
                    raise TypeError("HCU PCP attention requires its BF16 output allocation")
                metadata = _forward_context().attn_metadata
                if metadata is None:
                    return forward_mqa_impl(self, q, kv, positions, output)
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
                    return forward_mqa_impl(
                        self, q[:count], kv[:count], positions[:count], output[:count]
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
    if cfg.speculative_config is not None:
        from .pcp_dspark import HcuV41PcpDSparkAttention
        return HcuV41PcpDSparkAttention
    return build_classes()["attention"]
