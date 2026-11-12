# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""V4.1 FP8 indexer reference scoring, cache decoding and graph contract.

Moved here from the plugin. The 16x16 SHUFFLE page layout, the FP32 scale tail,
the packed-cache gather, the torch/triton scoring and the candidate-block
selection are upstream vLLM definitions (``rocm_aiter_mla_sparse`` and
``kernels/attention/dsa/candidate_blocks``). The HCU gap is only that the native
FP8 indexer refuses an AITER-off HCU before it executes, so this is upstream
behaviour the device cannot take on its own: adapter work, not plugin mechanism.

The plugin keeps the two things that are genuinely private to it: the FP4 cache
branch (LightOp) and the OOT publication of the resulting class.

Module load stays framework-free. Every vLLM/torch symbol is resolved inside
`build()` so `hcu_adapter` remains importable without vLLM installed. Host
collaborators whose historical patch points live in the plugin module are read
through `bind_runtime_resolvers`, so the adapter never imports the plugin.
"""
from __future__ import annotations

import threading

OP_NAME = "vllm::hcu_v41_fp8_indexer_reference"

_lock = threading.RLock()
_built = None
_resolvers: dict[str, object] = {}
_owner = None


class V41IndexerError(RuntimeError):
    """The V4.1 FP8 indexer reference is unavailable or its contract drifted."""


def bind_runtime_resolvers(owner: str, **resolvers) -> None:
    """Bind the host collaborators the scoring path reads.

    Each entry is a zero-argument callable evaluated at call time, so a patch on
    the plugin module attribute (the historical seam) still reaches the
    implementation. Rebinding the same owner is idempotent; a different owner is
    refused rather than silently changing what the indexer reads.
    """
    global _owner
    if not owner:
        raise V41IndexerError("V41 indexer resolver owner must be named")
    for name, resolver in resolvers.items():
        if not callable(resolver):
            raise V41IndexerError(f"V41 indexer collaborator {name!r} must be callable")
    with _lock:
        if _owner is not None and _owner != owner:
            raise V41IndexerError(
                f"V41 indexer collaborators already bound to {_owner!r}; restart process"
            )
        _owner = owner
        _resolvers.update(resolvers)


def bound_resolvers() -> tuple[str, ...]:
    """Read-only: names of the collaborators currently bound. Never binds."""
    return tuple(sorted(_resolvers))


def resolver_owner() -> str | None:
    """Read-only: the module that bound the collaborators, if any.

    Exposed so the owner can detect a hot-reload that replaced the binding
    module while the resolver table still points at the previous one.
    """
    return _owner


def _resolve(name):
    resolver = _resolvers.get(name)
    if resolver is None:
        raise V41IndexerError(
            f"V41 indexer collaborator {name!r} is not bound; activate the HCU plugin first"
        )
    return resolver()


def register_splitting_op(config) -> bool:
    """Append the reference op to a finalized Inductor piecewise split list.

    Must be called with the resolved ``VllmConfig`` after
    ``CompilationConfig.set_splitting_ops_for_v1`` has finalized the list. Do
    not use the early activation path: ``get_cached_compilation_config`` is not
    valid there on the v0.33 worker startup path.
    """
    compilation = getattr(config, "compilation_config", None)
    if compilation is None:
        return False
    from vllm.config import CompilationMode

    if getattr(compilation, "mode", None) != CompilationMode.VLLM_COMPILE:
        return False
    if getattr(compilation, "use_inductor_graph_partition", False):
        # The cudagraph_unsafe op tag is authoritative in this mode.
        return False
    cudagraph_mode = getattr(compilation, "cudagraph_mode", None)
    has_piecewise = getattr(cudagraph_mode, "has_piecewise_cudagraphs", None)
    if not callable(has_piecewise) or not has_piecewise():
        return False
    splitting_ops = getattr(compilation, "splitting_ops", None)
    if not isinstance(splitting_ops, list):
        raise RuntimeError(
            "CompilationConfig.splitting_ops must be finalized as a list "
            "before registering the V41 reference indexer"
        )
    if OP_NAME not in splitting_ops:
        splitting_ops.append(OP_NAME)
    return True


def graph_capture_allowed(config) -> bool:
    """Return whether a non-eager reference indexer has a capture break.

    Full cudagraphs are intentionally rejected: they are monolithic capture
    contexts, so a host-metadata custom op cannot be broken out of the replay.
    Use PIECEWISE (or the breakable piecewise wrapper) for this reference.
    """
    from vllm.compilation.breakable_cudagraph import (
        is_breakable_cudagraph_enabled,
    )

    compilation = getattr(config, "compilation_config", None)
    cudagraph_mode = getattr(compilation, "cudagraph_mode", None)
    if (
        cudagraph_mode is not None
        and cudagraph_mode.has_full_cudagraphs()
    ):
        return False
    if is_breakable_cudagraph_enabled():
        return True
    if compilation is None:
        return False
    from vllm.config import CompilationMode

    return bool(
        compilation.mode == CompilationMode.VLLM_COMPILE
        and (
            getattr(compilation, "use_inductor_graph_partition", False)
            or OP_NAME in (getattr(compilation, "splitting_ops", None) or [])
        )
    )


def resolve_graph_capture_allowed():
    """Return the graph-capture predicate this process admits by.

    The rule itself is adapter-owned, so a process that never activated the
    plugin still admits by ``graph_capture_allowed`` above. When the plugin
    activates it binds a zero-argument ``graph_allowed`` resolver that reads its
    historical module object; honouring that resolver is what keeps a patch on
    the plugin's patch point reaching every caller -- including
    ``hcu_adapter.fp4_indexer``, which must not import the plugin.
    """
    resolver = _resolvers.get("graph_allowed")
    return graph_capture_allowed if resolver is None else resolver()


def build():
    """Resolve the reference implementation against the imported vLLM.

    Built once per process. The result is a plain mapping so the plugin can
    re-publish the pieces it needs without the adapter naming the plugin.
    """
    global _built
    with _lock:
        if _built is not None:
            return _built

        import torch
        from vllm.compilation.breakable_cudagraph import eager_break_during_capture
        from vllm.logger import init_logger
        from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
            apply_candidate_mask,
            select_candidate_blocks,
        )
        from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
        from vllm.utils.torch_utils import _resolve_layer_name, direct_register_custom_op
        from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
            FP8_DTYPE,
            cp_gather_indexer_k_quant_cache_triton,
            fp8_mqa_logits_torch,
            rocm_aiter_sparse_attn_indexer as native_indexer,
        )

        from hcu_adapter.dsv41_channel_fp8 import _is_v41
        from hcu_adapter.fp4_indexer import applicable as fp4_applicable

        logger = init_logger(__name__)
        _registered = False

        def _collaborator(name):
            # No silent fallback: an unbound collaborator is a refusal, because
            # the plugin module is the single place these are patched.
            return _resolve(name)

        # Names that used to be closure-fixed at build() time and that a
        # historical `patch.object(v41_indexer, ...)` is documented to reach.
        # They are read through resolvers so the patch still changes what
        # executes. Types and dtypes stay closure-fixed on purpose: patching a
        # metadata class or a dtype is not a supported seam, and turning them
        # into resolvers would add hot-path indirection for no behaviour gain.
        def _native_indexer():
            resolver = _resolvers.get("native_indexer")
            return native_indexer if resolver is None else resolver()

        def _mqa_logits():
            resolver = _resolvers.get("mqa_logits")
            return fp8_mqa_logits_torch if resolver is None else resolver()

        def _layer_name():
            resolver = _resolvers.get("layer_name")
            return _resolve_layer_name if resolver is None else resolver()

        def _cache_flat(cache, head_dim):
            if cache.dtype != torch.uint8 or cache.ndim not in (3, 4):
                raise ValueError("FP8 indexer cache must be uint8 [pages,block,128+4]")
            if cache.shape[-1] != head_dim + 4 or (
                    cache.ndim == 4 and cache.shape[2] != 1):
                raise ValueError("FP8 indexer cache byte width/head count mismatch")
            pages, block = cache.shape[:2]
            if head_dim != 128 or block <= 0 or (block > 1 and block % 16):
                raise ValueError(
                    "V41 FP8 indexer requires D128 and block1 or16-aligned SHUFFLE")
            try:
                return cache.view(pages, block * (head_dim + 4))
            except RuntimeError as error:
                raise ValueError(
                    "FP8 indexer cache has incompatible packed inner strides") from error

        def _cache_pages(cache, head_dim, page_ids):
            block = cache.shape[1]
            flat = _cache_flat(cache, head_dim)[page_ids]
            pages = flat.shape[0]
            values = flat[:, :block * head_dim].contiguous().view(FP8_DTYPE)
            if block == 1:
                values = values.reshape(pages, block, head_dim)
            else:
                values = values.reshape(pages, block // 16, head_dim // 16, 16, 16)
                values = values.permute(0, 1, 3, 2, 4).reshape(pages, block, head_dim)
            scales = flat[:, block * head_dim:].contiguous().view(torch.float32).reshape(
                pages, block)
            return values, scales

        def decode_logits(q, cache, weights, context_lens, block_tables, max_model_len):
            if q.ndim != 4 or q.dtype != FP8_DTYPE:
                raise ValueError(
                    "V41 reference Q must be FP8 [requests,next_n,heads,128]")
            batch, next_n, heads, dim = q.shape
            flat = _cache_flat(cache, dim)
            if (weights.dtype != torch.float32
                    or weights.shape != (batch * next_n, heads)
                    or weights.device != q.device or cache.device != q.device
                    or block_tables.ndim != 2 or block_tables.shape[0] != batch):
                raise ValueError("FP8 indexer weights/table/device contract mismatch")
            if context_lens.ndim == 1 and context_lens.shape == (batch,):
                limits = (context_lens[:, None] - next_n +
                          torch.arange(next_n, device=context_lens.device)[None, :] + 1
                          ).clamp_min(0)
            elif context_lens.shape == (batch, next_n):
                limits = context_lens
            else:
                raise ValueError(
                    "FP8 indexer requires final1D or effective2D context lengths")
            if q.is_cuda:
                from hcu_adapter.v41_fp8_indexer_triton import rectangular_logits
                return rectangular_logits(q, cache, weights, context_lens,
                                          block_tables, max_model_len)
            logits = torch.full((batch * next_n, max_model_len), -float("inf"),
                                dtype=torch.float32, device=q.device)
            block = cache.shape[1]
            for req in range(batch):
                length = int(limits[req].max().item())
                if length < 0 or length > max_model_len:
                    raise ValueError("FP8 indexer context exceeds checked model length")
                if length == 0:
                    continue
                count = (length + block - 1) // block
                if count > block_tables.shape[1]:
                    raise ValueError("FP8 indexer block table is too short")
                pages = block_tables[req, :count].long()
                if bool(((pages < 0) | (pages >= flat.shape[0])).any().item()):
                    raise ValueError("FP8 indexer block ID is outside cache allocation")
                values, scales = _cache_pages(cache, dim, pages)
                key = values.reshape(-1, dim)[:length].float()
                scale = scales.reshape(-1)[:length]
                scores = torch.einsum("qhd,nd->hqn", q[req].float(), key).relu()
                scores = (scores * weights[req * next_n:(req + 1) * next_n].t()[:, :, None]
                          ).sum(0)
                scores = scores * scale[None, :]
                visible = (torch.arange(length, device=q.device)[None, :]
                           < limits[req, :, None])
                logits[req * next_n:(req + 1) * next_n, :length] = scores.masked_fill(
                    ~visible, -float("inf"))
            return logits

        def local_topk(logits, starts, ends, k):
            rows, width = logits.shape
            output = torch.full((rows, k), -1, dtype=torch.int32, device=logits.device)
            if not rows or not width:
                return output
            top = logits.topk(min(k, width), dim=-1)
            selected = (top.indices - starts[:, None]).to(torch.int32)
            selected.masked_fill_(top.values == -float("inf"), -1)
            ordered = top.indices.masked_fill(top.values == -float("inf"), width).sort(
                -1).values
            ordered = (ordered - starts[:, None]).to(torch.int32).masked_fill(
                ordered == width, -1)
            output[:, :top.indices.shape[1]] = torch.where(
                (ends - starts <= k)[:, None], ordered, selected)
            return output

        def _select(logits, starts, ends, out, candidates, block_size, write):
            if candidates is not None:
                helper = _collaborator(
                    "candidate_select" if write else "candidate_mask")
                if not logits.is_cuda and helper in (select_candidate_blocks,
                                                     apply_candidate_mask):
                    # Both upstream candidate helpers open with
                    # `assert logits.is_cuda`. A CPU replay that substitutes
                    # them never exercised this path, so a green replay is not
                    # evidence it works. Refuse explicitly instead of letting a
                    # bare AssertionError surface from inside the kernel.
                    raise V41IndexerError(
                        "V41 candidate-block selection needs CUDA; the reference "
                        "indexer's CPU candidate path is not implemented")
                if write:
                    helper(logits, starts, ends, candidates.shape[1], block_size,
                           candidates)
                else:
                    helper(logits, starts, ends, candidates, block_size)
            out.copy_(local_topk(logits, starts, ends, out.shape[1]))

        @eager_break_during_capture
        def indexer(hidden_states: torch.Tensor, prefix: str, cache: torch.Tensor,
                    q: torch.Tensor, weights: torch.Tensor, topk_tokens: int,
                    head_dim: int, max_model_len: int, total_seq_lens: int,
                    out: torch.Tensor, compress_ratio: int,
                    candidates: torch.Tensor | None = None,
                    candidate_block_size: int = 0,
                    candidate_write: bool = False) -> torch.Tensor:
            prefix = _layer_name()(prefix)
            context = _resolve("forward_context")
            if not isinstance(context.attn_metadata, dict):
                # Reuse the actual native profiling reservations/fake ABI; this
                # branch never reads cache bytes or invokes the faulty paged
                # decode fallback.
                return _native_indexer()(
                    hidden_states, prefix, cache, q, None, weights,
                    128, "ue8m0", topk_tokens, head_dim,
                    max_model_len, total_seq_lens, out, True,
                    compress_ratio, candidates,
                    candidate_block_size, candidate_write)
            meta = context.attn_metadata[prefix]
            if not isinstance(meta, DeepseekV32IndexerMetadata):
                raise TypeError("V41 reference indexer requires native indexer metadata")
            actual = meta.num_decode_tokens + meta.num_prefill_tokens
            if (q.ndim != 3 or q.dtype != FP8_DTYPE or q.shape[0] < actual
                    or q.shape[2] != head_dim):
                raise ValueError("V41 reference query layout/token count mismatch")
            if (weights.dtype != torch.float32 or weights.shape != q.shape[:2]
                    or weights.device != q.device):
                raise ValueError(
                    "V41 indexer requires same-device FP32 per-head weights")
            if (out.dtype != torch.int32 or out.ndim != 2 or out.shape[0] < actual
                    or out.shape[1] < topk_tokens or out.device != q.device):
                raise ValueError(
                    "V41 indexer output buffer shape/dtype/device mismatch")
            selected_out = out[:, :topk_tokens]
            if cache.device != q.device:
                raise ValueError("V41 indexer cache must be on the query device")
            _cache_flat(cache, head_dim)
            if candidates is not None:
                if (candidates.dtype != torch.int32 or candidates.ndim != 2
                        or candidates.shape[0] < actual or candidates.shape[1] <= 0
                        or candidates.device != q.device
                        or type(candidate_block_size) is not int
                        or candidate_block_size <= 0):
                    raise ValueError(
                        "V41 candidate schema requires int32 rows and positive block size")
                if meta.decode is not None and meta.decode.requires_padding:
                    if not q.is_cuda:
                        counts = meta.decode.decode_lens.tolist()
                        padded_rows = len(counts) * max(counts, default=0)
                        if candidates.shape[0] < padded_rows:
                            raise ValueError(
                                "ragged candidate bank lacks padded request-major rows")
                    if meta.num_prefill_tokens:
                        raise ValueError(
                            "padded decode candidates overlap prefill rows; "
                            "use separate batches")
            elif candidate_write:
                raise ValueError("candidate_write requires a candidate buffer")
            out[:hidden_states.shape[0]].fill_(-1)
            if meta.num_prefill_tokens:
                if meta.prefill is None:
                    raise ValueError("missing V41 prefill chunk metadata")
                for chunk in meta.prefill.chunks:
                    start, end = chunk.token_start, chunk.token_end
                    keys = torch.empty((chunk.total_seq_lens, head_dim),
                                       dtype=FP8_DTYPE, device=q.device)
                    scales = torch.empty((chunk.total_seq_lens, 4),
                                         dtype=torch.uint8, device=q.device)
                    _collaborator("gather_kv")(
                        cache, keys, scales, chunk.block_table, chunk.cu_seq_lens,
                        token_to_seq=chunk.token_to_seq)
                    logits = _mqa_logits()(
                        q[start:end], (keys, scales.view(torch.float32)),
                        weights[start:end], chunk.cu_seqlen_ks, chunk.cu_seqlen_ke)
                    _select(logits, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke,
                            selected_out[start:end],
                            None if candidates is None else candidates[start:end],
                            candidate_block_size, candidate_write)
            if q.is_cuda:
                if meta.num_decode_tokens:
                    if meta.decode is None:
                        raise ValueError("missing V41 decode metadata")
                    from hcu_adapter.v41_fp8_indexer_triton import decode_indexer
                    decode_indexer(q, cache, weights, meta.decode,
                                   meta.num_decode_tokens, selected_out, compress_ratio,
                                   candidates, candidate_block_size, candidate_write,
                                   max_model_len, _select)
                return out
            if meta.num_decode_tokens:
                if meta.decode is None:
                    raise ValueError("missing V41 decode metadata")
                decode = meta.decode
                lengths = decode.decode_lens.tolist()
                if sum(lengths) != meta.num_decode_tokens or any(n < 0 for n in lengths):
                    raise ValueError("V41 decode_lens do not cover actual query rows")
                offset = 0
                candidate_stride = max(lengths, default=0) if decode.requires_padding else None
                if candidates is not None and candidate_write and candidate_stride is not None:
                    candidates[:len(lengths) * candidate_stride].fill_(-1)
                # Q and outputs stay compact. Candidate rows retain Core's padded
                # request-major ABI without allocating padded queries or weights.
                for req, count in enumerate(lengths):
                    if not count:
                        continue
                    if decode.seq_lens.ndim == 2:
                        limits = decode.seq_lens[req:req + 1, :count]
                        if limits.shape[1] != count:
                            raise ValueError("V41 effective context rows are incomplete")
                    else:
                        if count > 1 and compress_ratio != 1:
                            raise ValueError(
                                "ratio2 multi-query decode requires effective "
                                "per-query context lengths")
                        limits = (decode.seq_lens[req:req + 1, None] - count +
                                  torch.arange(count, device=q.device)[None, :] + 1
                                  ).clamp_min(0)
                    end = offset + count
                    logits = decode_logits(q[offset:end].unsqueeze(0), cache,
                                           weights[offset:end], limits,
                                           decode.block_table[req:req + 1], max_model_len)
                    starts = torch.zeros(count, dtype=torch.int64, device=q.device)
                    bank_start = req * candidate_stride if candidate_stride is not None else offset
                    bank_rows = None if candidates is None else candidates[
                        bank_start:bank_start + count]
                    _select(logits, starts, limits.reshape(-1), selected_out[offset:end],
                            bank_rows, candidate_block_size, candidate_write)
                    offset = end
            return out

        def _fake(hidden_states, prefix, cache, q, weights, topk_tokens, head_dim,
                  max_model_len, total_seq_lens, out, compress_ratio, candidates=None,
                  candidate_block_size=0, candidate_write=False):
            return out

        def register_op(op_func=None):
            nonlocal _registered
            if _registered:
                return
            direct_register_custom_op(
                op_name="hcu_v41_fp8_indexer_reference",
                op_func=indexer if op_func is None else op_func,
                mutates_args=["out", "candidates"], fake_impl=_fake,
                dispatch_key="CompositeExplicitAutograd",
                tags=(torch._C.Tag.cudagraph_unsafe,))
            _registered = True

        class HcuV41IndexerBase(SparseAttnIndexer):
            """Reference indexer base. The plugin publishes the OOT subclass.

            Only the FP4 branch is plugin-private and is overridden by the owner
            class; everything else here is upstream V4.1 FP8 behaviour.
            """

            def __init__(self, *args, **kwargs):
                cfg = _resolve("current_config")
                is_v41 = _is_v41(cfg.model_config.hf_config)
                if is_v41:
                    _resolve("health")
                    if _resolve("enabled"):
                        _resolve("register_split")(cfg)
                self._hcu_owned = _resolve("enabled") and is_v41
                self._hcu_fp4 = is_v41 and fp4_applicable(cfg)
                if self._hcu_fp4:
                    self._fp4_preflight(cfg)
                    import inspect

                    values = inspect.signature(SparseAttnIndexer.__init__).bind(
                        self, *args, **kwargs)
                    values.apply_defaults()
                    v = values.arguments
                    if (not v["use_fp4_cache"] or not v["skip_k_cache_insert"]
                            or v["compress_ratio"] not in (1, 2) or v["head_dim"] != 128):
                        raise ValueError(
                            "FP4 owner requires source-owned FP4 K, ratio1/2, D128")
                    super().__init__(*args, **kwargs)
                    return
                if self._hcu_owned:
                    import inspect

                    bound = inspect.signature(SparseAttnIndexer.__init__).bind(
                        self, *args, **kwargs)
                    bound.apply_defaults()
                    p = cfg.parallel_config
                    if cfg.kernel_config.monokernel:
                        raise ValueError(
                            "V41 reference indexer cannot mix candidate buffers with "
                            "monokernel producers")
                    if not cfg.model_config.enforce_eager and not _resolve(
                            "graph_allowed")(cfg):
                        raise ValueError(
                            "V41 reference indexer requires eager execution or a "
                            "PIECEWISE breakable/inductor-split graph capture; "
                            "full cudagraph cannot break out the host-metadata op")
                    if (p.prefill_context_parallel_size != 1
                            or p.decode_context_parallel_size != 1):
                        raise ValueError(
                            "V41 reference indexer requires PCP1/DCP1; "
                            "PCP8 is a separate adapter")
                    values = bound.arguments
                    if (values["use_fp4_cache"] or not values["skip_k_cache_insert"]
                            or values["compress_ratio"] not in (1, 2)
                            or values["head_dim"] != 128):
                        raise ValueError(
                            "V41 reference indexer requires source-owned FP8 K, "
                            "ratio1/2, D128")
                super().__init__(*args, **kwargs)
                if self._hcu_owned:
                    logger.info(
                        "HCU_V41_FP8_INDEXER_REFERENCE: graph/eager PCP1/DCP1 "
                        "ratio=%s cache=%s", self.compress_ratio, self.k_cache.prefix)

            def _fp4_preflight(self, cfg):
                raise V41IndexerError(
                    "V41 FP4 indexer preflight is not bound; the plugin owns it")

            def _fp4_execute(self, hidden_states, q_quant, weights):
                raise V41IndexerError(
                    "V41 FP4 indexer execution is not bound; the plugin owns it")

            def forward_hip(self, hidden_states, q_quant, k, weights):
                if getattr(self, "_hcu_fp4", False):
                    _resolve("health")
                    if k is not None:
                        raise ValueError("FP4 reader must not insert a second K")
                    return self._fp4_execute(hidden_states, q_quant, weights)
                if not self._hcu_owned:
                    return super().forward_hip(hidden_states, q_quant, k, weights)
                if k is not None or not isinstance(q_quant, torch.Tensor):
                    raise ValueError(
                        "V41 reference indexer consumes source-owned K and one FP8 Q "
                        "tensor")
                return torch.ops.vllm.hcu_v41_fp8_indexer_reference(
                    hidden_states, _layer_name()(self.k_cache.prefix),
                    self.k_cache.kv_cache, q_quant, weights, self.topk_tokens,
                    self.head_dim, self.max_model_len, self.max_total_seq_len,
                    self.topk_indices_buffer, self.compress_ratio,
                    self.candidate_blocks, self.candidate_block_size,
                    self.candidate_write)

        _built = dict(
            base=HcuV41IndexerBase,
            indexer=indexer,
            local_topk=local_topk,
            decode_logits=decode_logits,
            select=_select,
            register_op=register_op,
            cache_flat=_cache_flat,
            cache_pages=_cache_pages,
            fake=_fake,
            fp8_dtype=FP8_DTYPE,
            native_indexer=native_indexer,
        )
        return _built


def fetch(name):
    """Resolve one built piece by name (used by the plugin's binding layer)."""
    return build()[name]
