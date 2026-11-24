# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Explicit target-only HiSparse provider seam; import performs no native work.

The companion plugin calls ensure_hisparse_ops only for an explicit HiSparse
request. No JIT, backend fallback, device probe, or vLLM import occurs here at
module import. Stable valid-index packing is implemented; DFly/owner/device
qualification is deliberately NOT implied.
"""

import importlib
import importlib.machinery
import os
from pathlib import Path
import sys
import threading

from ._hisparse_abi import SCHEMAS

_NAMES = tuple(schema.split('(', 1)[0] for schema in SCHEMAS)
_LOCK = threading.Lock()
_PID = os.getpid()
_LOADED = False
_FAILURE = None
_LIBRARY = None


def _packaged_library():
    # PathFinder avoids executing sgl_kernel.__init__ and its vendor bootstrap.
    package = importlib.machinery.PathFinder.find_spec('sgl_kernel', sys.path)
    if package is None or not package.submodule_search_locations:
        raise RuntimeError('HiSparse native wheel is absent; build setup_hip.py first')
    spec = importlib.machinery.PathFinder.find_spec(
        'sgl_kernel.hisparse_ops', package.submodule_search_locations)
    if spec is None or spec.origin is None:
        raise RuntimeError('HiSparse native library is absent; build setup_hip.py first')
    return str(Path(spec.origin).resolve())


def _states(torch):
    states = []
    for name, expected in zip(_NAMES, SCHEMAS):
        qualified = '_C_cache_ops::' + name
        try:
            schema = torch._C._dispatch_find_schema_or_throw(qualified, '').schema()
        except RuntimeError:
            states.append(False)
            continue
        canonical = str(torch._C.parse_schema(qualified + expected[len(name):]))
        if str(schema) != canonical:
            raise RuntimeError('HiSparse Core033 schema drift: ' + name)
        if not torch._C._dispatch_has_kernel_for_dispatch_key(qualified, 'CUDA'):
            raise RuntimeError('HiSparse schema without HIP/CUDA implementation: ' + name)
        states.append(True)
    return states


def ensure_hisparse_ops(library_path=None):
    """Load the four missing Core033 native ops, or validate an existing set.

    library_path is an explicit staged artifact for isolated build qualification;
    the no-argument plugin API discovers the paired native wheel. Loading is
    process-local and irreversible. Failure latches; use a fresh process after
    correcting configuration/binaries. Must be called before capture, not from
    inside an active HIP Graph. Successful load is NOT GPU/model qualification.
    """
    global _LOADED, _FAILURE, _LIBRARY
    # Never acquire a lock inherited from another thread across fork.
    if os.getpid() != _PID:
        raise RuntimeError('HiSparse provider inherited across fork; use spawn/fresh process')
    from hcu_adapter.infra.kill_switch import op_disabled
    if op_disabled('vllm_hisparse'):
        raise RuntimeError('HiSparse explicitly disabled by HCU_DISABLE_OPS=vllm_hisparse')
    with _LOCK:
        if _FAILURE is not None:
            raise RuntimeError('HiSparse native initialization previously failed') from _FAILURE
        if _LOADED:
            if library_path is not None and str(Path(library_path).resolve()) != _LIBRARY:
                raise RuntimeError('HiSparse provider artifact cannot change after publication')
            return
        try:
            import torch
            if not torch.version.hip:
                raise RuntimeError('HiSparse adapter native provider requires HIP Torch')
            # Load Core first so a partial/incompatible native host fails before
            # a fragment can duplicate schemas. Does not select a GPU device.
            importlib.import_module('vllm._C')
            states = _states(torch)
            if any(states) and not all(states):
                raise RuntimeError('Partial HiSparse Core registration; refusing mixed providers')
            if not all(states):
                path = _packaged_library() if library_path is None else str(Path(library_path).resolve())
                if not Path(path).is_file():
                    raise FileNotFoundError('HiSparse native artifact is absent: ' + path)
                torch.ops.load_library(path)
                if not all(_states(torch)):
                    raise RuntimeError('HiSparse native artifact did not register all four ops')
                _LIBRARY = path
                from hcu_adapter.infra import registry
                registry.register('vllm_hisparse',
                                  'hcu_adapter.kernels.hisparse_cpp',
                                  'torch.ops._C_cache_ops.hisparse_*', 'seam')
            _LOADED = True
        except Exception as exc:
            _FAILURE = exc
            raise
