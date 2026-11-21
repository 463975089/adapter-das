# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Native Core PCP indexer with HCU FP8 scoring ABIs; no copied scheduler loop."""
from __future__ import annotations

from importlib import import_module
from functools import wraps
from contextvars import ContextVar
import inspect
import os
import threading
from types import SimpleNamespace as NS

from hcu_adapter.infra import registry
from hcu_adapter.infra.kill_switch import op_disabled
from hcu_adapter.infra.native_namespace import bind_native

OP = 'vllm_hy4_sparse_indexer'
OP_NAME = 'hcu_hy4_sparse_attn_indexer'
_lock = threading.RLock()
_state = None
_failure = None


def _provider():
    for name in ('aiter.ops.triton.attention.fp8_mqa_logits',
                 'aiter.ops.triton.fp8_mqa_logits'):
        try:
            module = import_module(name)
        except ModuleNotFoundError as error:
            if error.name and (name == error.name or name.startswith(error.name + '.')):
                continue
            raise
        function = module.fp8_mqa_logits
        params = list(inspect.signature(function).parameters.values())
        if ([p.name for p in params[:6]] !=
                ['Q', 'KV', 'kv_scales', 'weights', 'cu_starts', 'cu_ends']
                or any(p.default is inspect.Parameter.empty for p in params[6:])):
            raise RuntimeError('HY4 FP8 MQA vendor ABI changed')
        return module, function
    raise RuntimeError('HY4 FP8 MQA vendor provider is unavailable')


def make_prefill(provider, compact, temporary_bytes):
    """Bound the full vendor result while retaining Core's compressed budget."""
    def prefill(q, kv, weights, starts, ends, max_seqlen_k):
        import torch
        query, q_scale = q
        keys, scales = kv
        if (q_scale is not None or query.dtype != torch.float8_e4m3fn
                or keys.dtype != query.dtype or scales.dtype != torch.float32):
            raise ValueError('HY4 indexer requires FP8 Q/K, folded Q scale and FP32 K scale')
        rows, columns = query.shape[0], keys.shape[0]
        if rows and max_seqlen_k and 4 * columns > temporary_bytes:
            raise ValueError('HY4 vendor full-logit row exceeds the temporary memory budget')
        output = torch.empty((rows, max_seqlen_k), dtype=torch.float32, device=query.device)
        if not rows or not max_seqlen_k:
            return output
        if not columns:
            return output.fill_(-float('inf'))
        step = max(1, temporary_bytes // (4 * columns))
        for begin in range(0, rows, step):
            stop = min(rows, begin + step)
            scores = provider(query[begin:stop], keys, scales, weights[begin:stop],
                              starts[begin:stop], ends[begin:stop])
            if (scores.shape != (stop - begin, columns) or scores.dtype != torch.float32
                    or scores.stride(1) != 1 or scores.device != query.device):
                raise RuntimeError('HY4 vendor MQA result ABI changed')
            compact(scores, output, starts[begin:stop], ends[begin:stop], begin)
            del scores
        return output
    return prefill


class DecodeLayout:
    """Use Core's public launchers with host metadata, including weight padding."""
    def __init__(self, pack, unpack):
        self.current = ContextVar('hy4_indexer_decode', default=None)
        self.pack_kernel = pack
        self.unpack_kernel = unpack

    def pack(self, x, lengths, pad_value=0, block_t=64, block_d=64):
        import torch
        decode, count = self.current.get()
        batch, width = lengths.shape[0], decode.write_max_decode_len
        if count != x.shape[0] or width < 1 or count > batch * width:
            raise ValueError('HY4 decode padding metadata does not cover its input')
        shape = x.shape[1:]
        # Copy FP8 bytes without a floating point cast or a GPU length query.
        byte_copy = x.dtype == torch.float8_e4m3fn
        source = x.contiguous().view(torch.uint8) if byte_copy else x.contiguous()
        source = source.reshape(count, -1)
        output = torch.empty((batch, width, source.shape[1]),
                             dtype=source.dtype, device=x.device)
        self.pack_kernel(source, output, lengths, N=count, D=source.shape[1],
            Lmax=width, pad_value=0, pad_is_uint8=byte_copy,
            block_t=block_t, block_d=block_d)
        return output.view(x.dtype).reshape(batch, width, *shape)

    def unpack(self, x, lengths, block_t=64, block_d=64):
        import torch
        _, count = self.current.get()
        batch, width = x.shape[:2]
        flat = x.contiguous().reshape(batch, width, -1)
        output = torch.empty((count, flat.shape[-1]), dtype=x.dtype, device=x.device)
        self.unpack_kernel(flat, output, lengths, B=batch, Lmax=width,
            D=flat.shape[-1], block_t=block_t, block_d=block_d)
        return output.reshape(count, *x.shape[2:])

    def weights(self, weights):
        state = self.current.get()
        if state is None:
            return weights
        decode, count = state
        if count and (decode.requires_padding or count % decode.decode_lens.shape[0]):
            return self.pack(weights[:count], decode.decode_lens).flatten(0, 1)
        return weights

    def scope(self, body, resolve_name):
        @wraps(body)
        def scoped(*args, **kwargs):
            from vllm.forward_context import get_forward_context
            metadata = get_forward_context().attn_metadata
            state = None
            if isinstance(metadata, dict):
                layer = metadata[resolve_name(args[1])]
                if layer.num_decodes:
                    state = (layer.decode, layer.num_decode_tokens)
            token = self.current.set(state)
            try:
                return body(*args, **kwargs)
            finally:
                self.current.reset(token)
        return scoped


def make_paged(provider, layout=None):
    def paged(q, kv_cache, weights, context_lens, block_tables,
              schedule_metadata, max_model_len, indices=None):
        if q[1] is not None or indices is not None:
            raise ValueError('HY4 HCU paged indexer does not support MX or varlen indices')
        if layout is not None:
            weights = layout.weights(weights)
        if weights.shape[0] != q[0].shape[0] * q[0].shape[1]:
            raise ValueError('HY4 paged indexer Q and weights have different padded row counts')
        return provider(q[0], kv_cache, weights, context_lens, block_tables,
                        max_model_len, cache_layout='NORMAL')
    return paged


def reserve_profile_temporary(body, temporary_bytes):
    @wraps(body)
    def profiled(*args, **kwargs):
        from vllm.forward_context import get_forward_context
        if not isinstance(get_forward_context().attn_metadata, dict):
            import torch
            # Keep vendor staging alive during Core's original profile so it
            # contributes to the measured peak before the KV budget is set.
            staging = torch.empty(temporary_bytes, dtype=torch.uint8, device=args[0].device)
            result = body(*args, **kwargs)
            del staging
            return result
        return body(*args, **kwargs)
    return profiled


def _check():
    global _failure
    if op_disabled(OP):
        if _state is not None:
            _failure = 'HY4 sparse indexer was disabled after publication; restart the process'
        raise RuntimeError('HY4 sparse indexer was disabled')
    if _failure is not None:
        raise RuntimeError(_failure)
    if _state is not None:
        import torch
        pid, _, operator, module, provider = _state
        if (pid != os.getpid() or module.fp8_mqa_logits is not provider
                or getattr(torch.ops.vllm, OP_NAME) is not operator):
            _failure = 'HY4 sparse indexer publication changed owner; restart the process'
            raise RuntimeError(_failure)


def _build():
    import torch
    import vllm.envs as envs
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture
    from vllm.model_executor.layers import sparse_attn_indexer as native
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import direct_register_custom_op
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_fp8_paged_mqa_logits_triton
    from vllm.v1.attention.ops.common import PackSeqTritonKernel, UnpackSeqTritonKernel
    from hcu_adapter.kernels.compact_mqa_logits_triton import compact

    module, provider = _provider()
    if hasattr(torch.ops.vllm, OP_NAME):
        raise RuntimeError('HY4 sparse indexer operator already has an owner')
    budget = min(128, envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB) * 1024 * 1024
    layout = DecodeLayout(PackSeqTritonKernel(), UnpackSeqTritonKernel())
    body = bind_native(inspect.unwrap(native.sparse_attn_indexer), {
        'fp8_fp4_mqa_logits': make_prefill(provider, compact, budget),
        'fp8_fp4_paged_mqa_logits': make_paged(rocm_fp8_paged_mqa_logits_triton, layout),
        'pack_seq_triton': layout.pack, 'unpack_seq_triton': layout.unpack})
    body = layout.scope(body, native._resolve_layer_name)
    direct_register_custom_op(op_name=OP_NAME,
        op_func=eager_break_during_capture(_guard_execution(reserve_profile_temporary(body, budget))),
        mutates_args=['topk_indices_buffer', 'candidate_blocks'],
        fake_impl=native.sparse_attn_indexer_fake,
        dispatch_key=current_platform.dispatch_key,
        tags=(torch._C.Tag.cudagraph_unsafe,))
    operator = getattr(torch.ops.vllm, OP_NAME)
    # The native wrapper uses torch only to dispatch this custom op. Its body,
    # kwargs/defaults and every other Core import remain unchanged.
    facade = NS(ops=NS(vllm=NS(sparse_attn_indexer=operator)))
    route = bind_native(native.SparseAttnIndexer.forward_cuda, {'torch': facade})
    registry.register(op=OP, impl=__name__ + '.forward',
        target='torch.ops.vllm.' + OP_NAME, mechanism=registry.MECH_SEAM)
    return os.getpid(), route, operator, module, provider


def _guard_execution(body):
    @wraps(body)
    def guarded(*args, **kwargs):
        # Runs behind the opaque operator boundary, not in Dynamo's graph.
        _check()
        return body(*args, **kwargs)
    return guarded


def prepare(config=None):
    """Publish before model compilation; retain eager callers and owner checks."""
    global _state, _failure
    _check()
    if config is not None:
        from hcu_adapter.features.dsv41.v41_fp8_indexer import register_splitting_op
        # Reuse the resolved-config helper with this opaque operator's name.
        # A renamed op must retain Core's original piecewise eager boundary.
        bind_native(register_splitting_op, {'OP_NAME': 'vllm::' + OP_NAME})(config)
    if _state is None:
        with _lock:
            _check()
            if _state is None:
                try:
                    _state = _build()
                except BaseException as error:
                    _failure = 'HY4 sparse indexer activation failed; restart the process'
                    raise RuntimeError(_failure) from error
    return _state[1]


def forward(layer, hidden_states, q_quant, k, weights):
    import torch
    if torch.compiler.is_compiling():
        route = _state[1]  # The layer constructor publishes before tracing.
    else:
        route = prepare()
    if (layer.use_fp4_cache or layer.compress_ratio != 1 or layer.dcp_world_size != 1
            or layer.candidate_blocks is not None):
        raise ValueError('HY4 HCU indexer requires FP8 ratio1/DCP1 without candidate blocks')
    return route(layer, hidden_states, q_quant, k, weights)
