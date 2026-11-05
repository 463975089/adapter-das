# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Extra failure, discovery and fork invariants with CPU doubles only."""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
from threading import Event, Thread

import pytest

# Reuse explicit CPU fixture definitions; never import production from tests.
from test_vllm_allreduce import isolated, provider, enable, compat, adapter_registry, envs, ops_module, bridge, AiterCustomAllreduce


def test_framework_free_discovery_and_shared_policy():
    import hcu_adapter
    import packaging
    root = str(Path(hcu_adapter.__file__).resolve().parent.parent)
    dependencies = str(Path(packaging.__file__).resolve().parent.parent)
    code = f"import sys; sys.path[:0] = {[root, dependencies]!r}\n" + '''
class Block:
 def find_spec(self, name, path=None, target=None):
  if name.split('.')[0] in {'torch','vllm','sglang','aiter','lightop'}:
   raise AssertionError('discovery imported '+name)
sys.meta_path.insert(0,Block())
from hcu_adapter import allreduce, allreduce_aiter
from hcu_adapter.bridges import vllm_allreduce
assert not vllm_allreduce.state().installed
assert not allreduce.requested_policy().selected
'''
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


def test_unavailable_provider_remains_clean_even_required_and_killed(monkeypatch, provider):
    _, _, vendor = provider
    monkeypatch.setenv('HCU_DISABLE_OPS', 'aiter_ar')
    monkeypatch.setenv('HCU_AITER_AR','1')
    monkeypatch.setattr(vendor, 'custom_ar', False)
    with pytest.raises(compat.AllReduceCompatibilityError, match='unavailable'):
        compat.install(require_available=True)
    assert compat._installed is None and compat._commit_failure is None
    assert adapter_registry.records() == {} and compat.FACTOR not in envs.environment_variables


def test_runtime_dump_records_both_installed_bindings(monkeypatch, provider):
    enable(monkeypatch)
    compat.install()
    records=adapter_registry.records()
    assert records['vllm_allreduce']['mechanism']=='rebind'
    assert records['vllm_allreduce']['target'].endswith('AiterCustomAllreduce.__init__')
    assert records['vllm_allreduce_admission']['target'].endswith('is_custom_all_reduce_enabled')
    assert all(r['source']=='hcu_adapter.bridges.vllm_allreduce' for r in records.values())


def test_actual_fork_during_install_refuses_without_deadlock(monkeypatch, provider):
    enable(monkeypatch)
    entered, release=Event(), Event()
    errors=[]
    original=ops_module.rocm_aiter_ops.register_custom_allreduce_ops
    def blocked():
        entered.set()
        assert release.wait(10)
        return original()
    monkeypatch.setattr(ops_module.rocm_aiter_ops, 'register_custom_allreduce_ops', staticmethod(blocked))
    def worker():
        try: compat.install()
        except BaseException as error: errors.append(error)
    thread=Thread(target=worker)
    thread.start()
    assert entered.wait(5)
    reader,writer=os.pipe()
    pid=os.fork()
    if pid==0:
        os.close(reader)
        try:
            compat.state()
            message=b'WRONG-allowed'
        except compat.AllReduceCompatibilityError:
            message=b'refused'
        os.write(writer,message)
        os._exit(0)
    os.close(writer)
    try:
        ready,_,_=select.select([reader],[],[],5)
        if not ready:
            os.kill(pid,signal.SIGKILL)
            pytest.fail('inherited install lock deadlocked')
        assert os.read(reader,100)==b'refused'
    finally:
        release.set()
        thread.join(5)
        os.close(reader)
        os.waitpid(pid,0)
    assert not thread.is_alive() and errors==[]
    assert compat.state().selected


def test_readonly_query_does_not_activate_and_concurrent_prepare_is_idempotent(monkeypatch, provider):
    from concurrent.futures import ThreadPoolExecutor
    assert not compat.state().installed
    enable(monkeypatch)
    assert not compat.state().installed
    with ThreadPoolExecutor(max_workers=2) as workers:
        states=list(workers.map(lambda _:compat.install(), range(2)))
    assert states[0]==states[1] and states[0].selected


def test_late_availability_requirement_does_not_upgrade_off_install(monkeypatch):
    compat.install()
    status=compat.state()
    before=compat.effective_compile_factor()
    with pytest.raises(compat.AllReduceCompatibilityError, match='before first installation'):
        compat.install(require_available=True)
    assert compat.state()==status and compat.effective_compile_factor()==before


def test_explicit_fabric_capability_is_preserved(monkeypatch,provider):
    _,state,_=provider
    enable(monkeypatch)
    state['fabric']=True
    monkeypatch.setenv('AITER_AR_TRANSPORT','fabric')
    compat.install()
    assert json.loads(compat.effective_compile_factor())['policy']['transport']=='fabric'


@pytest.mark.parametrize('reader',['state','factor','gate','construct','install','runtime-prepare'])
def test_actual_host_and_ar_locks_do_not_invert(monkeypatch,provider,reader):
    """Real host action/health and AR locks; profile observes but replaces no API."""
    enable(monkeypatch)
    compat.install()
    receive,send=os.pipe()
    pid=os.fork()  # Native allocation was NOT attempted in the parent.
    if pid==0:
        os.close(receive)
        bridge._RUNTIME_ACTIONS=bridge.ProcessLocalActions()
        entered,health_entered=Event(),Event()
        errors=[]
        real_health_code=bridge.ProcessLocalActions.require_runtime_healthy.__code__
        def profile(frame,event,arg):
            if event=='call' and frame.f_code is real_health_code:
                health_entered.set()
        def callback():
            entered.set()
            assert health_entered.wait(3)
            compat.state()
        def boot():
            try: bridge._RUNTIME_ACTIONS.run_once('vllm-runtime',callback)
            except BaseException as error: errors.append(repr(error))
        def query():
            try:
                assert entered.wait(3)
                sys.setprofile(profile)
                calls={'state':compat.state,'factor':compat.effective_compile_factor,
                       'gate':ops_module.rocm_aiter_ops.is_custom_all_reduce_enabled,
                       'construct':lambda:AiterCustomAllreduce(None,'cuda:0'),
                       'install':compat.install,'runtime-prepare':compat.prepare_runtime}
                calls[reader]()
            except BaseException as error: errors.append(repr(error))
            finally: sys.setprofile(None)
        owner=Thread(target=boot,daemon=True)
        reading=Thread(target=query,daemon=True)
        owner.start(); reading.start()
        owner.join(1); reading.join(1)
        row={'owner_alive':owner.is_alive(),'reader_alive':reading.is_alive(),
             'reader_reached_real_health':health_entered.is_set(),'errors':errors}
        os.write(send,json.dumps(row).encode())
        os._exit(0)
    os.close(send)
    try:
        ready,_,_=select.select([receive],[],[],8)
        if not ready:
            os.kill(pid,signal.SIGKILL)
            pytest.fail('bounded concurrency child timed out')
        row=json.loads(os.read(receive,8192))
        assert row['reader_reached_real_health'] and not row['owner_alive'] and not row['reader_alive'] and not row['errors'],row
    finally:
        os.close(receive)
        os.waitpid(pid,0)
