# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU build-entry regression; compiler/build APIs are doubles, not a HIP build."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]

CHILD = r'''
import json, os, runpy, sys, types
from pathlib import Path
root, legacy, mode, scenario = map(str, sys.argv[1:])
root, legacy = Path(root), Path(legacy)
for name in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
             'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
    assert os.environ[name] == '-1'
assert sys.flags.no_site and sys.flags.isolated
sys.path.insert(0, str(root))
if scenario != 'clean':
    # Execute a startup hook that genuinely imports an older package from disk.
    sys.path.insert(0, str(legacy))
    runpy.run_path(str(legacy / 'sitecustomize.py'))
    sys.path.remove(str(legacy))
    assert Path(sys.modules['hcu_adapter'].__file__).parent == legacy / 'hcu_adapter'
    old_package = sys.modules['hcu_adapter']
    if scenario not in ('clean', 'shadowed_package'):
        import hcu_adapter.kernels.hisparse_cpp
        old_helper = sys.modules['hcu_adapter.kernels.hisparse_cpp']

# A same-named loader module in sys.modules is not authoritative either.
if scenario == 'cached_loader_name':
    loader_cache = types.ModuleType('_das_hisparse_build')
    loader_cache.extension = old_helper.extension
    sys.modules['_das_hisparse_build'] = loader_cache

# Record real file-spec creation without replacing its loader or execution.
import importlib.util
real_spec_from_file_location = importlib.util.spec_from_file_location
origins = {}
def record_spec(name, location, *args, **kwargs):
    if name in ('_das_hisparse_build', '_das_build_version'):
        origins[name] = str(Path(location).resolve())
    return real_spec_from_file_location(name, location, *args, **kwargs)
importlib.util.spec_from_file_location = record_spec

calls = []

def forbidden_device_probe(*args, **kwargs):
    raise AssertionError('build metadata must use explicit AMDGPU_TARGETS')

def extension(**kwargs):
    calls.append(kwargs)
    return kwargs

class BuildExtension:
    @staticmethod
    def with_options(**kwargs):
        assert kwargs == {'use_ninja': True}
        return 'recorded-build-extension'

torch = types.ModuleType('torch')
torch.__version__ = '2.13.0'
torch.version = types.SimpleNamespace(hip='cpu-test-double')
torch.cuda = types.SimpleNamespace(is_available=forbidden_device_probe,
                                  get_device_properties=forbidden_device_probe)
utils = types.ModuleType('torch.utils')
cpp = types.ModuleType('torch.utils.cpp_extension')
cpp.BuildExtension, cpp.CUDAExtension = BuildExtension, extension
torch.utils, utils.cpp_extension = utils, cpp
sys.modules.update({'torch': torch, 'torch.utils': utils,
                    'torch.utils.cpp_extension': cpp})
setuptools = types.ModuleType('setuptools')
setup_calls = []
setuptools.setup = lambda **kwargs: setup_calls.append(kwargs)
def cpu_extension(name, **kwargs):
    return extension(name=name, **kwargs)
setuptools.Extension = cpu_extension
sys.modules['setuptools'] = setuptools
sys.argv = [str(root / 'setup_hip.py')]
if mode == 'isolated':
    sys.argv.append('--hisparse-only')
try:
    runpy.run_path(str(root / 'setup_hip.py'), run_name='__main__')
except Exception as exc:
    print(json.dumps({'ok': False, 'error_type': type(exc).__name__,
                      'error': str(exc), 'setup_calls': len(setup_calls)}))
else:
    assert len(setup_calls) == 1
    setup = setup_calls[0]
    expected = (['sgl_kernel.hisparse_ops'] if mode == 'isolated' else
                ['sgl_kernel.cpu_metadata', 'sgl_kernel.native_ops', 'sgl_kernel.hisparse_ops'])
    assert [v['name'] for v in setup['ext_modules']] == expected
    assert [v['name'] for v in calls] == expected
    hisparse = setup['ext_modules'][-1]
    assert hisparse['sources'] == [str(root / 'csrc/hisparse/hisparse_cpp.cu')]
    assert hisparse['libraries'] == ['hiprtc', 'amdhip64', 'c10', 'torch', 'torch_python']
    assert hisparse['extra_link_args'] == ['-Wl,-rpath,$ORIGIN/../../torch/lib',
                                         '-L/usr/lib/' + __import__('platform').machine().lower() + '-linux-gnu']
    assert hisparse['extra_compile_args'] == {
        'cxx': ['-O2', '-std=c++17'],
        'nvcc': ['-O2', '-std=c++17', '-DUSE_ROCM=1']}
    assert hisparse['py_limited_api'] is False
    assert setup['name'] == 'sgl-kernel'
    assert setup['version'] == '0.0.0+dtk2610.torch2130'
    assert setup['cmdclass'] == {'build_ext': 'recorded-build-extension'}
    assert '--hisparse-only' not in sys.argv
    assert os.environ['PYTORCH_ROCM_ARCH'] == 'gfx938'
    if scenario != 'clean':
        assert sys.modules['hcu_adapter'] is old_package
    if scenario not in ('clean', 'shadowed_package'):
        assert sys.modules['hcu_adapter.kernels.hisparse_cpp'] is old_helper
    assert origins['_das_hisparse_build'] == str(
        (root / 'hcu_adapter/kernels/hisparse_cpp.py').resolve())
    assert origins['_das_build_version'] == str(
        (root / 'packaging/hcu-adapter/_pin.py').resolve())
    if scenario == 'cached_loader_name':
        assert sys.modules['_das_hisparse_build'] is loader_cache
    print(json.dumps({'ok': True, 'extensions': expected, 'origins': origins,
                      'cached_package_preserved': scenario != 'clean',
                      'real_torch_imported': False, 'actual_build': False}))
'''


def _run(tmp_path, mode, scenario):
    build = tmp_path / 'checkout with spaces'
    for name in ('setup_hip.py', 'hcu_adapter/kernels/hisparse_cpp.py',
                 'packaging/hcu-adapter/_pin.py', 'sgl_kernel/version.py',
                 'csrc/hisparse/hisparse_cpp.cu'):
        dest = build / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    helper = build / 'hcu_adapter/kernels/hisparse_cpp.py'
    if scenario == 'missing_checkout_helper':
        helper.unlink()
    elif scenario == 'syntax_error':
        helper.write_text('def extension(:\n', encoding='utf-8')
    elif scenario == 'all_nul':
        helper.write_bytes(b'\x00' * helper.stat().st_size)
    elif scenario == 'missing_extension':
        helper.write_text('# no extension factory\n', encoding='utf-8')
    elif scenario == 'helper_raises':
        helper.write_text('raise RuntimeError("broken checkout helper")\n',
                          encoding='utf-8')
    legacy = tmp_path / 'old-install'
    (legacy / 'hcu_adapter/kernels').mkdir(parents=True)
    (legacy / 'hcu_adapter/__init__.py').write_text('', encoding='utf-8')
    (legacy / 'hcu_adapter/kernels/__init__.py').write_text('', encoding='utf-8')
    (legacy / 'sitecustomize.py').write_text('import hcu_adapter\n', encoding='utf-8')
    if scenario not in ('clean', 'shadowed_package'):
        (legacy / 'hcu_adapter/kernels/hisparse_cpp.py').write_text(
            'def extension(*args, **kwargs):\n'
            '    raise RuntimeError("stale installed helper must not be called")\n',
            encoding='utf-8')
    env = dict(os.environ, HCU_ADAPTER_HOST='none', AMDGPU_TARGETS='gfx938',
               DAS_PLATFORM_TAG='dtk2610.torch2130', DAS_BUILD_STAMP='',
               PYTHONDONTWRITEBYTECODE='1')
    for name in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
                 'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
        env[name] = '-1'
    # An unrelated cwd is intentional: __file__, not cwd/sys.path, is authority.
    p = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', CHILD,
                        str(build), str(legacy), mode, scenario],
                       cwd=tmp_path, env=env, capture_output=True, text=True,
                       timeout=30)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads(p.stdout)


@pytest.mark.parametrize('mode', ['allops', 'isolated'])
@pytest.mark.parametrize('scenario', ['clean', 'shadowed_package', 'cached_helper', 'cached_loader_name'])
def test_setup_uses_checkout_helper_without_rebinding_cached_package(tmp_path, mode, scenario):
    result = _run(tmp_path, mode, scenario)
    assert result['ok'], result


@pytest.mark.parametrize('mode', ['allops', 'isolated'])
def test_missing_checkout_helper_refuses_installed_fallback(tmp_path, mode):
    result = _run(tmp_path, mode, 'missing_checkout_helper')
    assert not result['ok']
    assert result['error_type'] == 'FileNotFoundError', result
    assert 'hcu_adapter/kernels/hisparse_cpp.py' in result['error'].replace('\\', '/')
    assert result['setup_calls'] == 0


@pytest.mark.parametrize('mode', ['allops', 'isolated'])
@pytest.mark.parametrize(('scenario', 'error_type'), [
    ('syntax_error', 'SyntaxError'),
    ('all_nul', 'SyntaxError'),
    ('missing_extension', 'AttributeError'),
    ('helper_raises', 'RuntimeError'),
])
def test_broken_checkout_helper_refuses_installed_fallback(tmp_path, mode,
                                                         scenario, error_type):
    result = _run(tmp_path, mode, scenario)
    assert not result['ok']
    assert result['error_type'] == error_type, result
    assert result['setup_calls'] == 0


def test_loader_rebinds_to_each_checkout_without_reusing_named_cache(tmp_path,
                                                                   monkeypatch):
    # Execute the production function's AST; helper import itself needs no Torch.
    import ast
    import types

    script = ROOT / 'setup_hip.py'
    tree = ast.parse(script.read_text(encoding='utf-8'))
    node, = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == '_hisparse_build_module']
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(script), 'exec'),
         namespace)
    stale = types.ModuleType('_das_hisparse_build')
    monkeypatch.setitem(sys.modules, '_das_hisparse_build', stale)
    loaded = []
    for name in ('first checkout', 'second checkout'):
        checkout = tmp_path / name
        helper = checkout / 'hcu_adapter/kernels/hisparse_cpp.py'
        helper.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / 'hcu_adapter/kernels/hisparse_cpp.py', helper)
        namespace['root'] = checkout
        module = namespace['_hisparse_build_module']()
        assert Path(module.__file__).resolve() == helper.resolve()
        assert callable(module.extension)
        assert sys.modules['_das_hisparse_build'] is stale
        loaded.append(module)
    assert loaded[0] is not loaded[1]
