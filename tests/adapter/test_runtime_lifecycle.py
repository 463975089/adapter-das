# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Shared CPU lifecycle contracts for independently named host consumers."""
from dataclasses import replace
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace as NS

import pytest

from hcu_adapter import extensions as ext
from hcu_adapter.runtime_lifecycle import ProcessLocalLifecycle
import hcu_adapter.runtime_lifecycle as lifecycle


CAP='shared-test-capability'


class ConsumerError(RuntimeError):
    pass


def activate(actions, checker):
    point=NS(name='owner',load=lambda: lambda: ext.ExtensionSpec(
        'owner',1,0,{CAP},lambda _: None,checker))
    def boot():
        reports=ext.activate_extensions(ext.ExtensionContext('cpu',None),[point],
            health_check_sink=actions.record_extension_health)
        actions.record_extension_reports(reports)
    actions.run_once(actions._runtime_action,boot)
    return actions.extension_reports()[0]


def fork_refusal(actions):
    reader,writer=os.pipe()
    pid=os.fork()
    if pid==0:
        os.close(reader)
        signal.alarm(4)
        try:
            assert not actions.extension_reports()
            actions.require_runtime_healthy()
        except ConsumerError:
            os.write(writer,b'REFUSED')
            os._exit(0)
        except BaseException:
            os._exit(2)
        os._exit(3)
    os.close(writer)
    try:
        ready,_,_=select.select([reader],[],[],6)
        if not ready:os.kill(pid,signal.SIGKILL)
        value=os.read(reader,32) if ready else b'TIMEOUT'
        _,status=os.waitpid(pid,0)
        assert value==b'REFUSED' and os.waitstatus_to_exitcode(status)==0
    finally:
        os.close(reader)


@pytest.fixture(params=['sglang-runtime','vllm-runtime'])
def actions(request):
    return ProcessLocalLifecycle(runtime_action=request.param,error_type=ConsumerError)


def test_exact_report_duplicate_and_sticky_failure_use_same_contract_for_both_consumers(actions):
    calls=[]
    report=activate(actions,lambda: calls.append('checked'))
    actions.require_extension_healthy(CAP)
    assert calls==['checked']
    with pytest.raises(ext.ExtensionCommitError):
        actions.record_extension_health(report,lambda: calls.append('BAD'))
    actions.record_extension_reports((replace(report),))
    with pytest.raises(ConsumerError):actions.require_extension_healthy(CAP)
    assert calls==['checked']
    actions.record_extension_reports((report,))
    with pytest.raises(ConsumerError):actions.require_runtime_healthy()
    with pytest.raises(ConsumerError):actions.run_once(actions._runtime_action,lambda: calls.append('RETRY'))
    assert calls==['checked']


@pytest.mark.skipif(not hasattr(os,'fork'),reason='actual bounded CPU fork required')
def test_failure_rendering_forks_and_raises_without_losing_host_latch(actions):
    observations=[]
    class BadString(RuntimeError):
        def __str__(self):
            fork_refusal(actions)
            observations.append('child refused')
            raise KeyboardInterrupt('diagnostic failed')
    def checker():raise BadString()
    activate(actions,checker)
    with pytest.raises(ConsumerError) as caught:actions.require_extension_healthy(CAP)
    assert isinstance(caught.value.__cause__,BadString)
    assert observations==['child refused'] and actions._health_inflight==0
    with pytest.raises(ConsumerError):actions.require_runtime_healthy()


@pytest.mark.skipif(not hasattr(os,'fork'),reason='actual bounded CPU fork required')
def test_inflight_callback_fork_refuses_without_inherited_lock_deadlock(actions):
    entered,release=threading.Event(),threading.Event()
    failures=[]
    def checker():
        entered.set()
        assert release.wait(8)
    activate(actions,checker)
    def check():
        try:actions.require_extension_healthy(CAP)
        except BaseException as error:failures.append(error)
    thread=threading.Thread(target=check,daemon=True)
    thread.start()
    try:
        assert entered.wait(3)
        fork_refusal(actions)
    finally:
        release.set()
        thread.join(4)
    assert not thread.is_alive() and not failures
    actions.require_runtime_healthy()


def test_caught_checker_reentry_cannot_heal_either_consumer(actions):
    calls=[]
    def checker():
        try:actions.run_once('other',lambda: calls.append('BAD'))
        except ext.ExtensionCommitError:pass
    activate(actions,checker)
    with pytest.raises(ConsumerError):actions.require_extension_healthy(CAP)
    assert not calls
    with pytest.raises(ConsumerError):actions.require_runtime_healthy()


def test_consumer_instances_do_not_share_reports_or_failure_state():
    a=ProcessLocalLifecycle(runtime_action='vllm-runtime',error_type=ConsumerError)
    b=ProcessLocalLifecycle(runtime_action='sglang-runtime',error_type=ConsumerError)
    activate(a,lambda: (_ for _ in ()).throw(ValueError('owner drift')))
    activate(b,lambda: None)
    with pytest.raises(ConsumerError):a.require_extension_healthy(CAP)
    b.require_extension_healthy(CAP)
    b.require_runtime_healthy()
    assert a.extension_reports()[0] is not b.extension_reports()[0]


@pytest.mark.parametrize('host',['sglang','vllm'])
def test_actual_shared_and_bridge_imports_for_both_hosts_deny_frameworks_and_torch(host):
    # -S isolates the import contract from system .pth/sitecustomize hooks.
    # Explicit paths include the actual prebound source/installed package and
    # packaging dependency, while a finder refuses every framework/kernel load.
    root=str(Path(lifecycle.__file__).resolve().parent.parent)
    paths=[root]+[p for p in sys.path if p and p!=root]
    code='''
import importlib.abc, importlib, json, sys
sys.path[:0]=PATHS
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','vllm','sglang','sgl_kernel','hcu_vllm_das'}:
            raise AssertionError('forbidden framework import: '+fullname)
sys.meta_path.insert(0,Deny())
from hcu_adapter import runtime_lifecycle as shared
sg=importlib.import_module('hcu_adapter.bridges.sglang')
vl=importlib.import_module('hcu_adapter.bridges.vllm')
assert issubclass(vl.ProcessLocalActions,shared.ProcessLocalLifecycle)
assert callable(sg.install)
from pathlib import Path
for module in (shared,sg,vl):assert Path(module.__file__).is_relative_to(ROOT)
assert not {'torch','sglang','vllm','sgl_kernel','hcu_vllm_das'}.intersection(sys.modules)
print(json.dumps({'host':HOST,'shared':shared.__file__,'sglang_bridge':sg.__file__,'vllm_bridge':vl.__file__}))
'''.replace('PATHS',repr(paths)).replace('ROOT',repr(root)).replace('HOST',repr(host))
    env=dict(os.environ,HCU_ADAPTER_HOST=host,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,'-S','-B','-c',code],env=env,text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)['host']==host
