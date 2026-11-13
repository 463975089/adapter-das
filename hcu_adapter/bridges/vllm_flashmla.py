# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Startup-only V4.1 prefill rebind; official MLA backend selection is preserved."""
from __future__ import annotations

from dataclasses import dataclass, replace
import functools
from importlib import import_module
import inspect
import logging
import os

from packaging.version import Version

from hcu_adapter import hcu_env, registry
from hcu_adapter.kill_switch import op_disabled
from hcu_adapter.runtime_lifecycle import ProcessLocalLifecycle

MODE_VARIABLE = 'HCU_VLLM_FLASHMLA_PREFILL'
OP = 'vllm_flashmla_prefill'
TARGET_MODULE = 'vllm.models.deepseek_v41.amd.rocm'
TARGET = TARGET_MODULE + '.rocm_sparse_attn_prefill'
_PARAMETERS = ('q', 'kv', 'indices', 'topk_length', 'scale', 'head_dim',
               'nope_head_dim', 'rope_head_dim', 'attn_sink', 'output',
               'ragged_indices', 'ragged_indptr')
_LOG = logging.getLogger('hcu.adapter')


class FlashMLACompatibilityError(RuntimeError):
    """Requested FlashMLA compatibility is unavailable or has drifted."""


@dataclass(frozen=True)
class _Publication:
    pid: int
    module: object
    original: object
    replacement: object


_publication = None
_failure = None
_lifecycle = ProcessLocalLifecycle(runtime_action='vllm-flashmla-prefill',
                                 error_type=FlashMLACompatibilityError)


SUPPORTED_ARCHES = ('gfx936', 'gfx938')


def selected_mode():
    if op_disabled(OP):
        return 'stock'
    mode, source = hcu_env.resolve(MODE_VARIABLE, host='vllm')
    if mode not in ('stock', 'native'):
        raise FlashMLACompatibilityError(MODE_VARIABLE + ' must be stock or native')
    if source == 'default' and _publication is None:
        # An implicit default must stay inert on hosts outside this candidate's
        # scope. Explicit environment/config selections retain strict admission.
        from hcu_adapter.bridges.vllm import detect_hcu, _installed_vllm_version
        detection = detect_hcu()
        if not detection.detected or detection.architecture not in SUPPORTED_ARCHES:
            return 'stock'
        version = Version(_installed_vllm_version() or '0')
        if version.epoch or version.release[:2] != (0, 33):
            return 'stock'
    return mode


def _refuse(reason, error=None):
    global _failure
    # Latch before formatting a foreign exception (which may itself raise).
    if _failure is None:
        _failure = reason
    raise FlashMLACompatibilityError(_failure) from error


def _require_signature(function, names, defaults):
    try:
        parameters = tuple(inspect.signature(function).parameters.values())
    except (TypeError, ValueError) as error:
        raise FlashMLACompatibilityError('cannot inspect FlashMLA prefill ABI') from error
    if (tuple(p.name for p in parameters) != names
            or any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
                   or p.default != defaults.get(p.name, inspect.Parameter.empty)
                   for p in parameters)):
        raise FlashMLACompatibilityError('FlashMLA prefill ABI changed: ' + function.__name__)


def _check_publication(p):
    if selected_mode() != 'native':
        _refuse('FlashMLA prefill policy changed; start a fresh process for stock/kill')
    if p.module.rocm_sparse_attn_prefill is not p.replacement:
        _refuse('FlashMLA prefill binding changed; start a fresh process')


def require_healthy():
    """Read-only health: never rebind or undo a captured/compiled call path."""
    _lifecycle.require_runtime_healthy()
    if _failure is not None:
        raise FlashMLACompatibilityError(_failure)
    if _publication is not None:
        if _publication.pid != os.getpid():
            raise FlashMLACompatibilityError('FlashMLA prefill requires child runtime activation')
        try:
            _check_publication(_publication)
        except BaseException as error:
            _refuse('FlashMLA prefill publication/policy drift; start a fresh process', error)


def install():
    """Install after host ABI admission, before extensions/model compilation."""
    global _publication
    if _failure is not None:
        raise FlashMLACompatibilityError(_failure)
    try:
        mode = selected_mode()
    except BaseException as error:
        if _publication is not None:
            _refuse('FlashMLA prefill policy invalid after installation', error)
        raise
    if mode == 'stock' and _publication is None:
        return False

    def initialize():
        global _publication
        from hcu_adapter.bridges.vllm import detect_hcu, _installed_vllm_version
        detection = detect_hcu()
        if not detection.detected or detection.architecture not in SUPPORTED_ARCHES:
            raise FlashMLACompatibilityError('FlashMLA prefill requires real gfx936/gfx938')
        version = Version(_installed_vllm_version())
        if version.epoch or version.release[:2] != (0, 33):
            raise FlashMLACompatibilityError('FlashMLA prefill requires vLLM 0.33')
        if _publication is not None:
            _check_publication(_publication)
            _publication = replace(_publication, pid=os.getpid())
            return
        # anchor-baseline(③): v0.33 rocm_sparse_attn_prefill retains 12 parameters,
        # ragged_indices/indptr=None. Bind ONLY V4.1's imported module global;
        # the shared ROCm op, V4.0 and already cached external aliases stay original.
        module = import_module(TARGET_MODULE)
        original = module.rocm_sparse_attn_prefill
        _require_signature(original, _PARAMETERS,
                           {'ragged_indices': None, 'ragged_indptr': None})
        if hasattr(original, '__wrapped__'):
            raise FlashMLACompatibilityError('FlashMLA prefill target already wrapped')
        try:
            provider = import_module('flash_mla.flash_mla_interface').flash_mla_sparse_fwd
        except (ImportError, AttributeError) as error:
            raise FlashMLACompatibilityError('native FlashMLA prefill explicitly requested but unavailable') from error
        # The DTK 35220f1 Python entry accepts seven positional arguments;
        # this path does not use DeepInfer's optional FP32 API or its pins.
        _require_signature(provider, ('q', 'kv', 'indices', 'sm_scale', 'd_v',
                                      'attn_sink', 'topk_length'),
                           {'d_v': 512, 'attn_sink': None, 'topk_length': None})
        from hcu_adapter.kernels.mla_sparse_prefill_dtk import try_prefill

        @functools.wraps(original)
        def routed(q, kv, indices, topk_length, scale, head_dim, nope_head_dim,
                   rope_head_dim, attn_sink, output, ragged_indices=None,
                   ragged_indptr=None):
            require_healthy()
            arguments = dict(q=q, kv=kv, indices=indices, topk_length=topk_length,
                scale=scale, head_dim=head_dim, nope_head_dim=nope_head_dim,
                rope_head_dim=rope_head_dim, attn_sink=attn_sink, output=output,
                ragged_indices=ragged_indices, ragged_indptr=ragged_indptr)
            if try_prefill(**arguments, provider=provider):
                return None
            return original(**arguments)

        # Register before publishing a function that may be cached by consumers.
        registry.register(op=OP, impl='native', target=TARGET,
                          mechanism=registry.MECH_REBIND, source=__name__,
                          detail='candidate V4.1 D512 BF16 prefill; dense indices only; device/model/perf pending')
        module.rocm_sparse_attn_prefill = routed
        _publication = _Publication(os.getpid(), module, original, routed)
        _LOG.info('HCU V4.1 prefill: native FlashMLA enabled; unsupported metadata uses stock')

    def guarded_initialize():
        try:
            initialize()
        except BaseException as error:
            _refuse('FlashMLA prefill activation failed; start a fresh process', error)

    _lifecycle.run_once('vllm-flashmla-prefill', guarded_initialize)
    require_healthy()
    return True
