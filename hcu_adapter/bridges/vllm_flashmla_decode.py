# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Startup rebind of V4.1's imported decode global, retaining its method ABI."""
from __future__ import annotations

from dataclasses import dataclass, replace
import functools
from importlib import import_module
import inspect
import os

from packaging.version import Version

from hcu_adapter import hcu_env, registry
from hcu_adapter.kill_switch import op_disabled
from hcu_adapter.runtime_lifecycle import ProcessLocalLifecycle

MODE_VARIABLE = 'HCU_VLLM_FLASHMLA_DECODE'
SUPPORTED_ARCHES = ('gfx936', 'gfx938')
OP = 'vllm_flashmla_decode'
TARGET_MODULE = 'vllm.models.deepseek_v41.amd.rocm'
TARGET = TARGET_MODULE + '.rocm_sparse_attn_decode'
PARAMETERS = ('q', 'kv_cache', 'swa_k_cache', 'swa_only', 'topk_indices',
    'topk_lens', 'swa_indices', 'swa_lens', 'swa_ragged_indices',
    'swa_ragged_indptr', 'topk_ragged_indices', 'topk_ragged_indptr',
    'attn_sink', 'scale', 'head_dim', 'nope_head_dim', 'rope_head_dim', 'output',
    'extra_cache_nan_free', 'adaptive_splits', 'inv_rope_positions',
    'inv_rope_cos_sin_cache', 'output_mxfp8')
DEFAULTS = dict(extra_cache_nan_free=False, adaptive_splits=False,
    inv_rope_positions=None, inv_rope_cos_sin_cache=None, output_mxfp8=None)
METHOD_PARAMETERS = ('self', 'q', 'positions', 'kv_cache', 'swa_metadata',
                     'attn_metadata', 'swa_only', 'output', 'output_mxfp8')


class FlashMLADecodeCompatibilityError(RuntimeError):
    """Native decode cannot safely activate or its startup publication drifted."""


@dataclass(frozen=True)
class _Publication:
    pid: int
    module: object
    original: object
    replacement: object
    cls: object
    method: object
    epilogue: object


_publication = None
_failure = None
_lifecycle = ProcessLocalLifecycle(runtime_action='vllm-flashmla-decode',
                                 error_type=FlashMLADecodeCompatibilityError)


def selected_mode():
    if op_disabled(OP):
        return 'stock'
    mode, source = hcu_env.resolve(MODE_VARIABLE, host='vllm')
    if mode not in ('stock', 'native'):
        raise FlashMLADecodeCompatibilityError(MODE_VARIABLE + ' must be stock or native')
    if source == 'default' and _publication is None:
        # An implicit default must stay inert on hosts outside this candidate's
        # scope. Explicit environment/config selections retain strict admission.
        from hcu_adapter.bridges.vllm import detect_hcu, _installed_vllm_version
        detected = detect_hcu()
        if not detected.detected or detected.architecture not in SUPPORTED_ARCHES:
            return 'stock'
        version = Version(_installed_vllm_version() or '0')
        if version.epoch or version.release[:2] != (0, 33):
            return 'stock'
    return mode


def _refuse(reason, error=None):
    global _failure
    if _failure is None:
        _failure = reason
    raise FlashMLADecodeCompatibilityError(_failure) from error


def _require_signature(fn, names, defaults=None, integer_return=False):
    signature = inspect.signature(fn)
    params = tuple(signature.parameters.values())
    if (tuple(p.name for p in params) != names
            or any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
                or p.default != (defaults or {}).get(p.name, inspect.Parameter.empty) for p in params)
            or (integer_return and signature.return_annotation not in (int, 'int'))):
        raise FlashMLADecodeCompatibilityError('FlashMLA decode ABI changed: ' + fn.__name__)


def _check(p):
    if selected_mode() != 'native':
        _refuse('FlashMLA decode policy changed; start a fresh process')
    if (p.module.rocm_sparse_attn_decode is not p.replacement
            or p.module.DeepseekV41ROCMAiterMLAAttention is not p.cls
            or p.cls._forward_decode is not p.method
            or p.module.rocm_inverse_rope_mxfp8_rows is not p.epilogue):
        _refuse('FlashMLA decode binding/method/epilogue drift; start a fresh process')


def require_healthy():
    _lifecycle.require_runtime_healthy()
    if _failure is not None:
        raise FlashMLADecodeCompatibilityError(_failure)
    if _publication is not None:
        if _publication.pid != os.getpid():
            raise FlashMLADecodeCompatibilityError('FlashMLA decode requires child runtime activation')
        try:
            _check(_publication)
        except BaseException as error:
            _refuse('FlashMLA decode publication/policy drift; start a fresh process', error)


def install():
    global _publication
    if _failure is not None:
        raise FlashMLADecodeCompatibilityError(_failure)
    try:
        mode = selected_mode()
    except BaseException as error:
        if _publication is not None:
            _refuse('FlashMLA decode policy invalid after installation', error)
        raise
    if mode == 'stock' and _publication is None:
        return False

    def initialize():
        global _publication
        from hcu_adapter.bridges.vllm import detect_hcu, _installed_vllm_version
        detected = detect_hcu()
        if not detected.detected or detected.architecture not in SUPPORTED_ARCHES:
            raise FlashMLADecodeCompatibilityError(
                f'packed FlashMLA decode requires real gfx936/gfx938; got {detected.architecture!r}')
        version = Version(_installed_vllm_version())
        if version.epoch or version.release[:2] != (0, 33):
            raise FlashMLADecodeCompatibilityError('packed FlashMLA decode requires vLLM 0.33')
        if _publication is not None:
            _check(_publication)
            _publication = replace(_publication, pid=os.getpid())
            return
        module = import_module(TARGET_MODULE)
        original = module.rocm_sparse_attn_decode
        # anchor-baseline(③): v0.33 _forward_decode has 8 non-self parameters,
        # output_mxfp8=None and -> int. It delegates through this 23-parameter
        # module global; retain the method, metadata construction and AITER branch.
        _require_signature(original, PARAMETERS, DEFAULTS, integer_return=True)
        cls = module.DeepseekV41ROCMAiterMLAAttention
        method = cls._forward_decode
        _require_signature(method, METHOD_PARAMETERS, {'output_mxfp8': None}, integer_return=True)
        if (method.__globals__ is not vars(module)
                or 'rocm_sparse_attn_decode' not in method.__code__.co_names
                or hasattr(original, '__wrapped__')):
            raise FlashMLADecodeCompatibilityError('V4.1 decode global call site or ownership changed')
        epilogue = module.rocm_inverse_rope_mxfp8_rows
        _require_signature(epilogue, ('o', 'positions', 'cos_sin_cache', 'rope_head_dim', 'out_data', 'out_scale'))
        if module.current_platform.is_fp8_fnuz():
            raise FlashMLADecodeCompatibilityError('packed FlashMLA expects OCP FP8 cache bytes, not FNUZ')
        native = import_module('flash_mla.flash_mla_interface')
        provider, factory = native.flash_mla_with_kvcache, native.get_mla_metadata
        _require_signature(provider, ('q', 'k_cache', 'block_table', 'cache_seqlens',
            'head_dim_v', 'tile_scheduler_metadata', 'num_splits', 'softmax_scale',
            'causal', 'is_fp8_kvcache', 'indices', 'attn_sink', 'extra_k_cache',
            'extra_indices_in_kvcache', 'topk_length', 'extra_topk_length'),
            dict(num_splits=None, softmax_scale=None, causal=False, is_fp8_kvcache=False,
                 indices=None, attn_sink=None, extra_k_cache=None,
                 extra_indices_in_kvcache=None, topk_length=None, extra_topk_length=None))
        kinds = tuple(p.kind for p in inspect.signature(factory).parameters.values())
        if kinds != (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise FlashMLADecodeCompatibilityError('native get_mla_metadata ABI changed')
        from hcu_adapter.kernels.mla_sparse_decode_dtk import try_decode

        @functools.wraps(original)
        def routed(q, kv_cache, swa_k_cache, swa_only, topk_indices, topk_lens,
                   swa_indices, swa_lens, swa_ragged_indices, swa_ragged_indptr,
                   topk_ragged_indices, topk_ragged_indptr, attn_sink, scale,
                   head_dim, nope_head_dim, rope_head_dim, output,
                   extra_cache_nan_free=False, adaptive_splits=False,
                   inv_rope_positions=None, inv_rope_cos_sin_cache=None,
                   output_mxfp8=None) -> int:
            require_healthy()
            arguments = dict(q=q, kv_cache=kv_cache, swa_k_cache=swa_k_cache,
                swa_only=swa_only, topk_indices=topk_indices, topk_lens=topk_lens,
                swa_indices=swa_indices, swa_lens=swa_lens,
                swa_ragged_indices=swa_ragged_indices, swa_ragged_indptr=swa_ragged_indptr,
                topk_ragged_indices=topk_ragged_indices, topk_ragged_indptr=topk_ragged_indptr,
                attn_sink=attn_sink, scale=scale, head_dim=head_dim,
                nope_head_dim=nope_head_dim, rope_head_dim=rope_head_dim, output=output,
                extra_cache_nan_free=extra_cache_nan_free, adaptive_splits=adaptive_splits,
                inv_rope_positions=inv_rope_positions, inv_rope_cos_sin_cache=inv_rope_cos_sin_cache,
                output_mxfp8=output_mxfp8)
            rotated = try_decode(**arguments, provider=provider,
                                 metadata_factory=factory, epilogue=epilogue)
            if rotated is not None:  # Zero is a successful BF16 result, not fallback.
                return rotated
            return original(**arguments)

        registry.register(op=OP, impl='native', target=TARGET,
            mechanism=registry.MECH_REBIND, source=__name__,
            detail='candidate gfx936/gfx938 D512 packed FP8; no BF16 KV gather; fresh plan per call; GPU/Graph/perf pending')
        module.rocm_sparse_attn_decode = routed
        _publication = _Publication(os.getpid(), module, original, routed, cls, method, epilogue)

    def guarded_initialize():
        try:
            initialize()
        except BaseException as error:
            _refuse('FlashMLA decode activation failed; start a fresh process', error)

    _lifecycle.run_once('vllm-flashmla-decode', guarded_initialize)
    require_healthy()
    return True
