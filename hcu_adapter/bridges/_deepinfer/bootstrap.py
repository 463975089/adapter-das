"""One explicit HCU installer and lazy propagation to spawned CP workers.

The import hook itself imports only stdlib/glue. Heavy providers are installed
when a spawned worker imports the engine, after its inherited device mask is set.
"""

import importlib.abc
import importlib.machinery
import json
from pathlib import Path
import sys

from . import hcu_env, registry


def _install(artifacts=None, physical_device_count=None, tilelang_backend=None,
            grouped_gemm_backend=None, allow_cuda_graph=None):
    from . import configuration
    from hcu_adapter.ops import selector

    configured, config_metadata = configuration.load()
    arguments = dict(artifacts=artifacts, physical_device_count=physical_device_count,
                     tilelang_backend=tilelang_backend, grouped_gemm_backend=grouped_gemm_backend,
                     allow_cuda_graph=allow_cuda_graph)
    defaults = dict(artifacts=None, physical_device_count=None, tilelang_backend='triton',
                    grouped_gemm_backend='triton', allow_cuda_graph=True)
    chosen, sources = {}, {}
    for key, value in arguments.items():
        chosen[key] = value if value is not None else configured.get(key, defaults[key])
        sources[key] = 'argument' if value is not None else ('config' if key in configured else 'default')
    artifacts, physical_device_count = chosen['artifacts'], chosen['physical_device_count']
    tilelang_backend, grouped_gemm_backend = chosen['tilelang_backend'], chosen['grouped_gemm_backend']
    allow_cuda_graph = chosen['allow_cuda_graph']
    if tilelang_backend not in ('triton', 'torch') or grouped_gemm_backend not in ('triton', 'torch'):
        raise ValueError('HCU installer backends must be triton or torch')
    if type(allow_cuda_graph) is not bool:
        raise ValueError('allow_cuda_graph must be a boolean')
    if physical_device_count is not None and (type(physical_device_count) is not int or physical_device_count <= 0):
        raise ValueError('physical_device_count must be a positive integer')
    if hcu_env.resolve('HCU_PLATFORM_SEMANTICS') != 'gpu':
        raise ValueError('Only GPU platform semantics are supported by the HCU installer')
    upstream_defaults = hcu_env.apply_upstream_defaults(allow_cuda_graph)
    from . import (deep_ep, deep_select, lightop_gemm, native_extension, platform_guard,
                   tile_kernels, tilelang_ops, vendor_imports, worker_runtime)

    if artifacts is None:
        candidates = [Path(__file__).resolve().parents[3] / 'artifacts', Path('/opt/deepinfer-hcu/artifacts')]
        artifacts = next((str(p) for p in candidates if (p / 'hcu-build.json').is_file()), None)
        if artifacts is None:
            raise FileNotFoundError("HCU native artifacts not found; pass artifacts= to bootstrap.install")
        sources['artifacts'] = 'auto-detect'
    if physical_device_count is None:
        physical_device_count = len(list(Path('/sys/class/drm').glob('card[0-9]*/device/mem_info_vram_total')))
        if not physical_device_count:
            raise RuntimeError("Cannot infer the physical GPU inventory; pass physical_device_count=")
        sources['physical_device_count'] = 'sysfs'
    device_visibility = hcu_env.align_device_visibility(physical_device_count)
    from hcu_adapter.bridges import common

    common.install()
    platform_guard.install_inventory(physical_device_count)
    native_extension.install(artifacts)
    lightop_gemm.install()
    tilelang_ops.install(tilelang_backend)
    tile_kernels.install()
    deep_ep.install(allow_cuda_graph=allow_cuda_graph)
    deep_select.install()
    vendor_imports.install(grouped_gemm_backend=grouped_gemm_backend)
    worker_runtime.install(physical_device_count, allow_cuda_graph=allow_cuda_graph)
    platform_guard.install()
    from . import runtime_dump

    runtime_dump.write(dict(artifacts=str(artifacts), physical_device_count=physical_device_count,
                            tilelang_backend=tilelang_backend, grouped_gemm_backend=grouped_gemm_backend,
                            allow_cuda_graph=allow_cuda_graph, selection_sources=sources,
                            configuration=config_metadata, upstream_defaults=upstream_defaults,
                            device_visibility=device_visibility))


def publish_worker_context(physical_device_count, allow_cuda_graph):
    """Freeze the installed provider choices, without importing any GPU modules."""
    bindings = registry.snapshot()['bindings']
    required = ('deepinfer._C', 'tile_kernels.import_surface', 'deep_ep.EPBuffer',
                'vendors.import_surface', 'tilelang_ops.norm_tl')
    if not all(name in bindings for name in required):
        return  # A standalone worker-policy installation does not imply a full runtime.
    package = sys.modules.get('deepinfer')
    paths = getattr(package, '__path__', ())
    artifacts = next((str(Path(p).parent) for p in paths if (Path(p).parent / 'hcu-build.json').is_file()), None)
    if artifacts is None:
        raise RuntimeError("Installed HIP artifacts cannot be located for spawned workers")
    from . import tilelang_ops

    implementation = bindings['deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous']['implementation']
    grouped = 'triton' if implementation.startswith('hcu_adapter.kernels.grouped_gemm_triton.') else 'torch'
    context = dict(artifacts=artifacts, physical_device_count=physical_device_count,
                   tilelang_backend=tilelang_ops._backend, grouped_gemm_backend=grouped,
                   allow_cuda_graph=allow_cuda_graph)
    hcu_env.set_worker_context(json.dumps(context, sort_keys=True))
    registry.register('worker.bootstrap.context', 'hcu_adapter.bridges._deepinfer.bootstrap.publish_worker_context',
                      'HCU_WORKER_CONTEXT', 'seam')


class _WorkerLoader(importlib.abc.Loader):
    def __init__(self, loader, context, finder):
        self.loader, self.context, self.finder = loader, context, finder

    def create_module(self, spec):
        return self.loader.create_module(spec) if hasattr(self.loader, 'create_module') else None

    def exec_module(self, module):
        # Remove before installing providers: installer imports must never recurse.
        if self.finder in sys.meta_path:
            sys.meta_path.remove(self.finder)
        if _INSTALL_STATE is None or _INSTALL_ERROR is not None:
            install(**self.context)
        self.loader.exec_module(module)


class _WorkerFinder(importlib.abc.MetaPathFinder):
    def __init__(self, context):
        self.context = context

    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'deepinfer.engine.llm_engine':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _WorkerLoader(spec.loader, self.context, self)
        return spec


def register_worker_import_hook():
    value = hcu_env.resolve('HCU_WORKER_CONTEXT')
    if not value:
        context, _ = configuration_for_hook()
    else:
        context = json.loads(value)
    expected = {'artifacts', 'physical_device_count', 'tilelang_backend', 'grouped_gemm_backend', 'allow_cuda_graph'}
    if set(context) - expected or ('physical_device_count' in context and type(context['physical_device_count']) is not int):
        raise ValueError("Invalid inherited HCU worker context")
    from . import platform_guard

    # DP example children query device count before importing the engine. This
    # inventory bridge is CPU-only, so their later device masks remain effective.
    count = context.get('physical_device_count')
    if count is None:
        count = len(list(Path('/sys/class/drm').glob('card[0-9]*/device/mem_info_vram_total')))
    if count:
        platform_guard.install_inventory(count)
    if 'deepinfer.engine.llm_engine' in sys.modules:
        raise RuntimeError("HCU worker bootstrap must run before importing the engine")
    if not any(isinstance(finder, _WorkerFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _WorkerFinder(context))
    registry.register('worker.bootstrap.import_hook', 'hcu_adapter.bridges._deepinfer.bootstrap.register_worker_import_hook',
                      'sys.meta_path:deepinfer.engine.llm_engine', 'seam')


def configuration_for_hook():
    from . import configuration
    return configuration.load()


_INSTALL_STATE = None
_INSTALL_ERROR = None


def install(**options):
    global _INSTALL_STATE, _INSTALL_ERROR
    if _INSTALL_ERROR is not None:
        raise RuntimeError("DeepInfer installation failed; use a fresh process") from _INSTALL_ERROR
    if _INSTALL_STATE is not None:
        if options != _INSTALL_STATE:
            raise RuntimeError("DeepInfer is already installed with different options")
        return
    try:
        _install(**options)
    except Exception as error:
        _INSTALL_ERROR = error
        raise
    _INSTALL_STATE = dict(options)
