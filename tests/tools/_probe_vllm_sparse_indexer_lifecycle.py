# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Installed C4 lifecycle/cache contract with a CPU hardware-admission double.

Run outside the checkout, using the installed candidate Python:
HCU_ADAPTER_HOST=none VLLM_PLUGINS=hcu_adapter HIP_VISIBLE_DEVICES=''
CUDA_VISIBLE_DEVICES='' PYTHONPATH='' PYTHONNOUSERSITE=1 python THIS_FILE
--output DIR. This proves CPU registration/spawn/fork/hash contracts only;
real platform/native guards, device math and model execution are separate.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import traceback


def _child(queue):
    try:
        queue.put(_exercise())
    except BaseException as error:
        queue.put(dict(error=type(error).__name__+': '+str(error), traceback=traceback.format_exc()))


def _exercise():
    from hcu_adapter.bridges import vllm as host
    from hcu_adapter.bridges import vllm_sparse_indexer as m
    from vllm import envs
    from vllm.config.utils import hash_factors
    from vllm.compilation.caching import aot_compile_hash_factors
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
    envs.disable_envs_cache()
    host.detect_hcu = lambda: host.HcuDetection(True, 'gfx936', 'CPU lifecycle probe double')
    os.environ['HCU_VLLM_SPARSE_INDEXER_IMPL'] = 'stock'
    assert not m.install()
    assert type(SparseAttnIndexer.__new__(SparseAttnIndexer)) is SparseAttnIndexer
    config = type('Config', (), {'compute_hash': lambda self: 'fixed-config-hash'})()
    off = envs.compile_factors()
    off_aot = aot_compile_hash_factors(config)[0]
    assert m.COMPILE_FACTOR not in off and off_aot == hash_factors(off)
    os.environ['HCU_DISABLE_OPS'] = 'vllm_sparse_indexer'
    os.environ['HCU_VLLM_SPARSE_INDEXER_IMPL'] = 'native'
    assert not m.install() and envs.compile_factors() == off
    os.environ.pop('HCU_DISABLE_OPS')
    assert m.install()
    m.require_healthy()
    assert type(SparseAttnIndexer.__new__(SparseAttnIndexer)).__name__ == 'HcuSparseAttnIndexer'
    on = envs.compile_factors()
    on_aot = aot_compile_hash_factors(config)[0]
    assert on[m.COMPILE_FACTOR].startswith('official-c4:')
    assert hash_factors(off) != hash_factors(on) and on_aot != off_aot
    assert on_aot == hash_factors(on)
    envs.enable_envs_cache()
    assert envs.compile_factors() == on and aot_compile_hash_factors(config)[0] == on_aot
    assert envs.__getattr__(m.COMPILE_FACTOR) == on[m.COMPILE_FACTOR]
    envs.disable_envs_cache()
    fork_result = None
    if hasattr(os, 'fork'):
        readfd, writefd = os.pipe()
        pid = os.fork()
        if pid == 0:
            signal.alarm(10)
            os.close(readfd)
            try:
                try:
                    m.require_healthy()
                except m.SparseIndexerCompatibilityError:
                    pass
                else:
                    raise AssertionError('fork trusted parent publication')
                assert m.install()
                m.require_healthy()
                payload = dict(adopted=True, pid=os.getpid())
            except BaseException as error:
                payload = dict(error=type(error).__name__+': '+str(error))
            os.write(writefd, json.dumps(payload).encode())
            os.close(writefd)
            os._exit(0)
        os.close(writefd)
        raw = os.read(readfd, 65536)
        os.close(readfd)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        fork_result = json.loads(raw)
        assert fork_result.get('adopted'), fork_result
    # Exercise the actual public constructor selector, without touching its
    # private OOT registry. Restoring a binding cannot undo a latched failure.
    original_new = SparseAttnIndexer.__new__
    SparseAttnIndexer.__new__ = staticmethod(lambda cls, *args, **kwargs: object.__new__(cls))
    try:
        m.require_healthy()
    except m.SparseIndexerCompatibilityError:
        pass
    else:
        raise AssertionError('foreign selector admitted')
    SparseAttnIndexer.__new__ = original_new
    for action in (m.require_healthy, m.install):
        try:
            action()
        except m.SparseIndexerCompatibilityError:
            pass
        else:
            raise AssertionError('late drift recovered after restoring constructor')
    path = Path(m.__file__)
    return dict(status='CPU_INSTALLED_LIFECYCLE_PASS', pid=os.getpid(), module=str(path),
        module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), hardware_identity_double=True,
        stock_and_disable_preserve_original=True, ordinary_hash_differs=True,
        aot_hash_differs=True, cached_env_getter_preserved=True,
        factor=on[m.COMPILE_FACTOR], fork=fork_result, constructor_drift_latched=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('HIP_VISIBLE_DEVICES') == os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not os.environ.get('PYTHONPATH')
    args.output.mkdir(exist_ok=False)
    (args.output/'driver.py').write_bytes(Path(__file__).read_bytes())
    record = dict(status='RUNNING', gpu_or_model_acceptance=False)
    process = None
    try:
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        process = ctx.Process(target=_child, args=(queue,))
        process.start()
        child = queue.get(timeout=60)
        process.join(timeout=30)
        assert process.exitcode == 0 and child.get('status') == 'CPU_INSTALLED_LIFECYCLE_PASS', child
        parent = _exercise()
        assert parent['factor'] == child['factor'] and parent['module'] == child['module']
        record.update(status='CPU_INSTALLED_SPAWN_FORK_CACHE_DRIFT_PASS', parent=parent, spawn=child)
    except BaseException as error:
        record.update(status='CPU_INSTALLED_LIFECYCLE_HOLD', error=type(error).__name__+': '+str(error),
                      traceback=traceback.format_exc())
    finally:
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=10)
        (args.output/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    print(record['status'])
    raise SystemExit(0 if record['status'] == 'CPU_INSTALLED_SPAWN_FORK_CACHE_DRIFT_PASS' else 1)


if __name__ == '__main__':
    main()
