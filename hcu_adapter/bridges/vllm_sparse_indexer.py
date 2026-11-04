# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""C4 indexer ABI glue over public vLLM metadata and official numerical helpers.

Opt-in compatibility for real gfx936/v0.33. Imports stay framework-free until
runtime activation; no framework file, architecture flag or helper is rebound.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from importlib import import_module
import inspect
import json
import os
from pathlib import Path

from packaging.version import Version

from hcu_adapter import hcu_env, registry
from hcu_adapter.kill_switch import op_disabled
from hcu_adapter.runtime_lifecycle import ProcessLocalLifecycle

MODE_VARIABLE = 'HCU_VLLM_SPARSE_INDEXER_IMPL'
COMPILE_FACTOR = 'VLLM_HCU_ADAPTER_SPARSE_INDEXER_IMPLEMENTATION'
OP_NAME = 'hcu_adapter_sparse_indexer_c4'


class SparseIndexerCompatibilityError(RuntimeError):
    """Requested C4 compatibility is unavailable or its publication drifted."""


_lifecycle = ProcessLocalLifecycle(runtime_action='vllm-sparse-indexer',
                                 error_type=SparseIndexerCompatibilityError)
_publication = None
_failure = None


@dataclass(frozen=True)
class _Publication:
    pid: int
    original: object
    replacement: object
    envs: object
    factor_getter: object


def selected_mode() -> str:
    if op_disabled('vllm_sparse_indexer'):
        return 'stock'
    value, _ = hcu_env.resolve(MODE_VARIABLE)
    if value not in ('stock', 'native'):
        raise ValueError(f'{MODE_VARIABLE} must be stock or native')
    return value


def _refuse(reason, error=None):
    global _failure
    # Publish refusal before user-defined exception formatting/reentry/fork.
    if _failure is None:
        _failure = reason
        if error is not None:
            try:
                _failure = f'{reason}: {type(error).__name__}: {error}'
            except BaseException:
                pass
    raise SparseIndexerCompatibilityError(_failure) from error


def _check_publication(p):
    if selected_mode() != 'native':
        _refuse('sparse indexer policy changed after registration; start a fresh process')
    if type(p.original.__new__(p.original)) is not p.replacement:
        _refuse('official sparse indexer constructor selection changed')
    if p.envs.environment_variables.get(COMPILE_FACTOR) is not p.factor_getter:
        _refuse('sparse indexer compilation factor ownership changed')


def require_healthy() -> None:
    """Read only: never activate, restore a registry, or clear failed state."""
    _lifecycle.require_runtime_healthy()
    if _failure is not None:
        raise SparseIndexerCompatibilityError(_failure)
    p = _publication
    if p is None:
        if selected_mode() == 'native':
            raise SparseIndexerCompatibilityError('sparse indexer requested before runtime activation')
        return
    if p.pid != os.getpid():
        raise SparseIndexerCompatibilityError('sparse indexer inherited from another PID; activate in this child first')
    try:
        _check_publication(p)
    except BaseException as error:
        _refuse('sparse indexer health check failed', error)


def _require_parameters(fn, names, defaults=None):
    parameters = inspect.signature(fn).parameters
    if tuple(parameters) != names or any(
            parameters[k].default != v for k, v in (defaults or {}).items()):
        raise SparseIndexerCompatibilityError(f'official sparse indexer ABI changed: {fn.__name__}')


def _declare(original, official, common, workspace, platform, context):
    # Official public kernel launchers accept host-known bounds, unlike the
    # convenience pack/unpack helpers that synchronize GPU lengths with .item().
    global torch
    torch = import_module('torch')
    pack = common.PackSeqTritonKernel()
    unpack = common.UnpackSeqTritonKernel()

    def indexer(hidden: torch.Tensor, prefix: str, cache: torch.Tensor,
                query: torch.Tensor, weights: torch.Tensor, output: torch.Tensor,
                max_len: int, max_total: int) -> None:
        require_healthy()
        metadata_map = context.get_forward_context().attn_metadata
        if not isinstance(metadata_map, dict):
            # Preserve the upstream profiling memory reservation contract.
            official.rocm_aiter_sparse_attn_indexer(hidden, prefix, cache,
                query, None, weights, 128, 'ue8m0', 512, 128, max_len, max_total,
                output, skip_k_cache_insert=True, compress_ratio=4)
            return None
        metadata = metadata_map[prefix]
        if (cache.ndim != 3 or cache.shape[1:] != (64, 132)
                or cache.dtype != torch.uint8 or not cache.is_contiguous()
                or cache.device != query.device):
            raise SparseIndexerCompatibilityError('unsupported C4 physical cache ABI')
        output[:hidden.shape[0]].fill_(-1)
        if metadata.prefill is not None:
            keys, scale_bytes = workspace.current_workspace_manager().get_simultaneous(
                ((max_total, 128), platform.current_platform.fp8_dtype()),
                ((max_total, 4), torch.uint8))
            for chunk in metadata.prefill.chunks:
                live_keys = keys[:chunk.total_seq_lens]
                live_scales = scale_bytes[:chunk.total_seq_lens]
                official.cp_gather_indexer_k_quant_cache_triton(cache, live_keys,
                    live_scales, chunk.block_table, chunk.cu_seq_lens,
                    chunk.token_to_seq, cache_layout='NORMAL')
                rows = slice(chunk.token_start, chunk.token_end)
                scores = official.rocm_fp8_mqa_logits(query[rows],
                    (live_keys, live_scales.view(torch.float32)), weights[rows],
                    chunk.cu_seqlen_ks, chunk.cu_seqlen_ke)
                torch.ops._C.top_k_per_row_prefill(scores, chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke, output[rows], scores.shape[0],
                    scores.stride(0), scores.stride(1), 512)
        if metadata.decode is not None and metadata.num_decode_tokens:
            decode = metadata.decode
            count = metadata.num_decode_tokens
            batch = decode.decode_lens.shape[0]
            if decode.requires_padding:
                width = decode.write_max_decode_len
                query_bytes = torch.empty((batch, width, 64*128),
                    dtype=torch.uint8, device=query.device)
                packed_weights = torch.empty((batch, width, 64),
                    dtype=weights.dtype, device=weights.device)
                packed_ids = torch.full((batch, width, 512), -1,
                    dtype=output.dtype, device=output.device)
                pack(query[:count].contiguous().view(torch.uint8).reshape(count, -1),
                    query_bytes, decode.decode_lens, N=count, D=64*128,
                    Lmax=width, pad_value=0, pad_is_uint8=True, block_t=64, block_d=64)
                pack(weights[:count].contiguous(), packed_weights, decode.decode_lens,
                    N=count, D=64, Lmax=width, pad_value=0, pad_is_uint8=False,
                    block_t=64, block_d=64)
                queries = query_bytes.view(query.dtype).view(batch, width, 64, 128)
                score_weights = packed_weights.reshape(batch*width, 64)
                destination = packed_ids.view(batch*width, 512)
            else:
                queries = query[:count].reshape(batch, -1, 64, 128)
                score_weights = weights[:count]
                destination = output[:count]
            scores = official.rocm_fp8_paged_mqa_logits_triton(queries,
                cache.unsqueeze(-2), score_weights, decode.seq_lens,
                decode.block_table, max_len)
            torch.ops._C.top_k_per_row_decode(scores, queries.shape[1],
                decode.seq_lens, destination, scores.shape[0],
                scores.stride(0), scores.stride(1), 512)
            if decode.requires_padding:
                unpack(packed_ids, output[:count], decode.decode_lens,
                    B=batch, Lmax=width, D=512, block_t=64, block_d=64)
        return None

    def fake(hidden, prefix, cache, query, weights, output, max_len, max_total):
        return None

    class HcuSparseAttnIndexer(original):
        def forward_hip(self, hidden_states, q_quant, k, weights):
            if self.compress_ratio != 4 or self.use_fp4_cache:
                return super().forward_hip(hidden_states, q_quant, k, weights)
            # Reject unsupported variants before any output/cache write. Cache
            # insertion/compression remains the original model's responsibility.
            if (not self.skip_k_cache_insert or k is not None
                    or self.candidate_blocks is not None or self.candidate_block_size
                    or self.candidate_write or self.dcp_world_size > 1 or self.use_pcp):
                raise SparseIndexerCompatibilityError(
                    'C4 candidate requires original compressor, no candidates and PCP/DCP=1')
            if (not isinstance(q_quant, torch.Tensor) or q_quant.ndim != 3
                    or q_quant.shape[1:] != (64, 128)
                    or q_quant.dtype != platform.current_platform.fp8_dtype()
                    or weights.shape != q_quant.shape[:2] or weights.dtype != torch.float32
                    or self.quant_block_size != 128 or self.scale_fmt != 'ue8m0'
                    or self.topk_tokens != 512 or self.head_dim != 128):
                raise SparseIndexerCompatibilityError('unsupported C4 indexer query/scale/top-k ABI')
            cache = self.k_cache.kv_cache
            output = self.topk_indices_buffer
            if (output.ndim != 2 or output.shape[1] != 512
                    or output.shape[0] < q_quant.shape[0] or output.dtype != torch.int32
                    or not output.is_contiguous() or not q_quant.is_contiguous()
                    or not weights.is_contiguous()
                    or hidden_states.shape[0] != q_quant.shape[0]
                    or q_quant.device.type != 'cuda' or hidden_states.device != q_quant.device
                    or weights.device != q_quant.device or output.device != q_quant.device):
                raise SparseIndexerCompatibilityError('unsupported C4 cache/output buffer ABI')
            # Cache binding/health are checked inside the opaque operator.
            # Profiling may still hold an unbound placeholder; Dynamo must not
            # trace Python locks or a step's mutable attention metadata here.
            torch.ops.vllm.hcu_adapter_sparse_indexer_c4(hidden_states,
                self.k_cache.prefix, cache, q_quant, weights, output,
                self.max_model_len, self.max_total_seq_len)
            return output

    return indexer, fake, HcuSparseAttnIndexer


def install() -> bool:
    """Activate only after canonical host/DTK admission, once per process."""
    global _publication
    if _failure is not None:
        raise SparseIndexerCompatibilityError(_failure)
    try:
        mode = selected_mode()
    except BaseException as error:
        if _publication is not None:
            _refuse('sparse indexer policy invalid after registration; start a fresh process', error)
        raise
    if mode == 'stock' and _publication is None:
        return False

    def initialize():
        global _publication
        from hcu_adapter.bridges.vllm import detect_hcu, _installed_vllm_version
        detected = detect_hcu()
        if not detected.detected or detected.architecture != 'gfx936':
            raise SparseIndexerCompatibilityError('C4 indexer is only tested on real gfx936')
        version = Version(_installed_vllm_version())
        if version.epoch or version.release[:2] != (0, 33):
            raise SparseIndexerCompatibilityError('C4 indexer requires the vLLM 0.33 release line')
        if _publication is not None:
            _check_publication(_publication)
            _publication = replace(_publication, pid=os.getpid())
            return
        layer = import_module('vllm.model_executor.layers.sparse_attn_indexer')
        official = import_module('vllm.v1.attention.ops.rocm_aiter_mla_sparse')
        common = import_module('vllm.v1.attention.ops.common')
        workspace = import_module('vllm.v1.worker.workspace')
        platform = import_module('vllm.platforms')
        context = import_module('vllm.forward_context')
        metadata = import_module('vllm.v1.attention.backends.mla.indexer')
        envs = import_module('vllm.envs')
        register = import_module('vllm.utils.torch_utils').direct_register_custom_op
        original = layer.SparseAttnIndexer
        _require_parameters(original.forward_hip, ('self', 'hidden_states', 'q_quant', 'k', 'weights'))
        _require_parameters(original.__init__, ('self', 'k_cache', 'quant_block_size', 'scale_fmt',
            'topk_tokens', 'head_dim', 'max_model_len', 'max_total_seq_len', 'topk_indices_buffer',
            'skip_k_cache_insert', 'use_fp4_cache', 'compress_ratio', 'candidate_blocks',
            'candidate_block_size', 'candidate_write'), dict(skip_k_cache_insert=False,
            use_fp4_cache=False, compress_ratio=1, candidate_blocks=None,
            candidate_block_size=0, candidate_write=False))
        _require_parameters(official.rocm_fp8_paged_mqa_logits_triton,
            ('q_fp8', 'kv_cache_fp8', 'weights', 'context_lens', 'block_tables', 'max_model_len'))
        _require_parameters(official.rocm_fp8_mqa_logits,
            ('q', 'kv', 'weights', 'cu_seqlen_ks', 'cu_seqlen_ke'))
        _require_parameters(official.cp_gather_indexer_k_quant_cache_triton,
            ('k_cache', 'k_fp8', 'k_fp8_scale', 'block_table', 'cu_seqlen', 'token_to_seq',
             'block_tile_size', 'head_tile_size', 'cache_layout'),
            dict(block_tile_size=16, head_tile_size=16, cache_layout=None))
        _require_parameters(official.rocm_aiter_sparse_attn_indexer,
            ('hidden_states', 'k_cache_prefix', 'kv_cache', 'q_fp8', 'k', 'weights',
             'quant_block_size', 'scale_fmt', 'topk_tokens', 'head_dim', 'max_model_len',
             'total_seq_lens', 'topk_indices_buffer', 'skip_k_cache_insert', 'compress_ratio',
             'candidate_blocks', 'candidate_block_size', 'candidate_write'),
            dict(skip_k_cache_insert=False, compress_ratio=1, candidate_blocks=None,
                 candidate_block_size=0, candidate_write=False))
        _require_parameters(common.PackSeqTritonKernel.__call__, ('self', 'x_reshaped', 'out',
            'lengths', 'N', 'D', 'Lmax', 'pad_value', 'pad_is_uint8', 'block_t', 'block_d'))
        _require_parameters(common.UnpackSeqTritonKernel.__call__, ('self', 'packed_reshaped',
            'out', 'lengths', 'B', 'Lmax', 'D', 'block_t', 'block_d'))
        if type(original.__new__(original)) is not original:
            raise SparseIndexerCompatibilityError('another owner already selects a sparse indexer OOT')
        if COMPILE_FACTOR in envs.environment_variables:
            raise SparseIndexerCompatibilityError('sparse indexer compilation factor has another owner')
        implementation, fake, replacement = _declare(original, official, common, workspace, platform, context)
        if hasattr(torch.ops.vllm, OP_NAME):
            raise SparseIndexerCompatibilityError('sparse indexer Torch operator already has another owner')
        files = (Path(__file__), *(Path(inspect.getfile(x)) for x in
                    (layer, official, common, workspace, metadata)))
        content = dict(sources=[hashlib.sha256(p.read_bytes()).hexdigest() for p in files],
                       torch=torch.__version__, hip=torch.version.hip, vllm=str(version))
        factor = 'official-c4:' + hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        def factor_getter():
            return factor
        try:
            register(OP_NAME, implementation, mutates_args=['output'], fake_impl=fake,
                     dispatch_key=platform.current_platform.dispatch_key)
            original.register_oot(replacement)
            envs.environment_variables[COMPILE_FACTOR] = factor_getter
            _publication = _Publication(os.getpid(), original, replacement, envs, factor_getter)
            _check_publication(_publication)
            registry.register(op='vllm_sparse_indexer', impl='native',
                target='vllm.model_executor.layers.sparse_attn_indexer.SparseAttnIndexer',
                mechanism=registry.MECH_SEAM, source=__name__,
                detail='candidate gfx936/v0.33 C4 public metadata + official logits/top-k; original compressor')
        except BaseException as error:
            _refuse('sparse indexer publication failed; start a fresh process', error)

    def guarded_initialize():
        try:
            initialize()
        except BaseException as error:
            _refuse('sparse indexer activation failed; start a fresh process', error)

    _lifecycle.run_once('vllm-sparse-indexer', guarded_initialize)
    require_healthy()
    return True
