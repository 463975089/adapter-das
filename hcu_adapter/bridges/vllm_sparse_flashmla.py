# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Public sparse FlashMLA backend registration for actual HCU vendor kernels."""
from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module
import inspect
import logging
import os

from packaging.version import Version

from hcu_adapter.infra import registry
from hcu_adapter.infra.kill_switch import op_disabled
from hcu_adapter.infra.runtime_lifecycle import ProcessLocalLifecycle

OP = 'vllm_sparse_flashmla'
BACKEND = 'hcu_adapter.kernels.vllm_sparse_flashmla_dtk.HcuSparseFlashMLABackend'
TARGET = 'vllm.v1.attention.backends.registry:FLASHMLA_SPARSE'
_LOG = logging.getLogger('hcu.adapter')
_lifecycle = ProcessLocalLifecycle(runtime_action=OP)
_publication = None
_failure = None


@dataclass(frozen=True)
class _Publication:
    pid: int
    owner: object
    enum: object
    register: object
    backend: object
    vendor: object
    functions: tuple


def _native_provider():
    vendor = import_module('flash_mla.flash_mla_interface')
    parameters = inspect.signature(vendor.flash_mla_sparse_fwd).parameters
    names = ('q', 'kv', 'indices', 'sm_scale', 'd_v', 'attn_sink', 'topk_length')
    if (tuple(parameters) != names or parameters['d_v'].default != 512
            or parameters['attn_sink'].default is not None
            or parameters['topk_length'].default is not None):
        raise ValueError('DTK sparse FlashMLA prefill ABI is unsupported')
    decode = inspect.signature(vendor.flash_mla_with_kvcache).parameters
    if not {'q', 'k_cache', 'block_table', 'cache_seqlens', 'head_dim_v',
            'tile_scheduler_metadata', 'is_fp8_kvcache', 'indices',
            'softmax_scale', 'attn_sink'}.issubset(decode):
        raise ValueError('DTK sparse FlashMLA decode ABI is unsupported')
    if not callable(vendor.get_mla_metadata):
        raise ValueError('DTK sparse FlashMLA metadata provider is unavailable')
    return vendor


def _functions(vendor):
    return (vendor.flash_mla_sparse_fwd, vendor.flash_mla_with_kvcache,
            vendor.get_mla_metadata)


def _backend_registry():
    return import_module('vllm.v1.attention.backends.registry')


def require_healthy():
    _lifecycle.require_runtime_healthy()
    if _failure is not None:
        raise RuntimeError(_failure)
    p = _publication
    if p is None:
        return
    if p.pid != os.getpid():
        raise RuntimeError('sparse FlashMLA requires child runtime activation')

    def verify():
        record = registry.get(OP)
        if (op_disabled(OP) or p.enum.get_path() != BACKEND
                or p.enum.get_class() is not p.backend
                or p.owner.register_backend is not p.register
                or _functions(p.vendor) != p.functions
                or record is None or record.get('target') != TARGET
                or record.get('source') != __name__
                or record.get('mechanism') != registry.MECH_SEAM):
            raise RuntimeError('sparse FlashMLA publication changed owner or policy')
    try:
        verify()
    except BaseException as error:
        _refuse('sparse FlashMLA publication changed owner or policy', error)


def _refuse(reason, error=None):
    global _failure
    if _failure is None:
        _failure = reason
    raise RuntimeError(_failure) from error


def require_available():
    require_healthy()
    if _publication is None:
        raise RuntimeError('HCU sparse FlashMLA is unavailable; install a compatible '
                           'DTK flash-mla provider and enable ' + OP)


def install():
    global _publication
    _lifecycle.require_runtime_healthy()
    if _publication is None:
        if op_disabled(OP):
            return False
        from hcu_adapter.bridges.vllm import detect_hcu, _installed_vllm_version
        detected = detect_hcu()
        version = Version(_installed_vllm_version() or '0')
        if (not detected.detected or detected.architecture not in ('gfx936', 'gfx938')
                or version.epoch or version.release[:2] != (0, 33)):
            return False
        try:
            vendor = _native_provider()
        except (ImportError, AttributeError, OSError, ValueError, TypeError):
            _LOG.info('HCU sparse FlashMLA: optional vendor unavailable or incompatible')
            return False
    else:
        vendor = _publication.vendor

    def publish():
        global _publication
        if _publication is not None:
            _publication = replace(_publication, pid=os.getpid())
            require_healthy()
            return
        owner = _backend_registry()
        enum = owner.AttentionBackendEnum.FLASHMLA_SPARSE
        if enum.is_overridden():
            raise RuntimeError('sparse FlashMLA backend has a foreign owner')
        from hcu_adapter.kernels.vllm_sparse_flashmla_dtk import build_backend
        backend = build_backend(vendor)
        functions = _functions(vendor)
        if enum.is_overridden():
            raise RuntimeError('sparse FlashMLA backend acquired a foreign owner')
        registry.register(op=OP, impl='native', target=TARGET,
            mechanism=registry.MECH_SEAM, source=__name__,
            detail='Core D576 sparse MLA with sink, DTK ABI; DCP excluded; model/perf pending')
        owner.register_backend(enum, BACKEND)
        _publication = _Publication(os.getpid(), owner, enum,
                                   owner.register_backend, backend, vendor, functions)
    _lifecycle.run_once(OP, publish)
    require_healthy()
    return True
