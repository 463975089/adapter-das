"""Record installed providers without probing or initializing an accelerator."""

import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

from . import hcu_env, registry
from .hcu_config import BACKEND_CATALOG


def snapshot(install_options=None):
    root = Path(__file__).resolve().parents[3]
    active = registry.snapshot()
    for op, entry in active['bindings'].items():
        entry['catalog_matches'] = {
            backend: dict(candidate)
            for backend, candidate in BACKEND_CATALOG.get(op, {}).items()
            if candidate['implementation'] == entry['implementation']
        }
    environment = {
        name: dict(value=hcu_env.resolve(name), source=hcu_env.environment.resolve(name)[1])
        for name in hcu_env.REGISTRY
    }
    versions = {}
    for name in ('torch', 'triton', 'deep-ep', 'deepgemm', 'lightop', 'transformers', 'tokenizers', 'xgrammar'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    modules = {}
    for name in ('torch', 'deep_gemm', 'deepgemm', 'deep_ep', 'flash_mla', 'flash_attn.cute.interface', 'lightop', 'tile_kernels'):
        module = sys.modules.get(name)
        if module is not None:
            modules[name] = dict(file=getattr(module, '__file__', None), package_paths=list(getattr(module, '__path__', ())))
    return dict(schema_version=1, pid=os.getpid(), adapter_root=str(root),
                install_options=install_options, environment=environment,
                versions=versions, modules=modules, **active)


def write(install_options=None):
    """Emit one atomic report per PID; shared CP paths cannot overwrite peers."""
    registry.register('runtime.dump', 'hcu_adapter.bridges._deepinfer.runtime_dump.write', 'HCU_ADAPTER_DUMP', 'seam')
    requested = hcu_env.resolve('HCU_ADAPTER_DUMP')
    if not requested:
        return None
    report = snapshot(install_options)
    data = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + '\n'
    if requested == '1':
        print('HCU_ADAPTER_DUMP ' + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        return None
    if '{pid}' in requested:
        destination = Path(requested.replace('{pid}', str(os.getpid()))).expanduser()
    else:
        path = Path(requested).expanduser()
        destination = path.with_name(path.stem + f'.{os.getpid()}' + (path.suffix or '.json'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=destination.parent,
                                         prefix='.' + destination.name + '.', delete=False) as handle:
            temporary = handle.name
            handle.write(data)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()
    print(f'HCU adapter runtime: provider dump -> {destination}', flush=True)
    return str(destination)
