# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Installed public mHC lifecycle/cache probe with CPU hardware-admission double.

Run from outside the checkout using the candidate installed Python:
HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' PYTHONPATH='' PYTHONNOUSERSITE=1
HCU_ADAPTER_HOST=none VLLM_PLUGINS=hcu_adapter python THIS_FILE --output DIR.
The CPU identity double proves publication/child/cache contracts only; real
native guards and tensor math require the separate gfx936 installed probe.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path


def _child(queue, policy_only=False):
    try:
        queue.put(_exercise_invalid_policy() if policy_only else _exercise())
    except BaseException as error:
        queue.put(dict(error=type(error).__name__+': '+str(error)))


def _exercise_invalid_policy():
    from hcu_adapter.bridges import vllm as host, vllm_mhc as m
    host.detect_hcu=lambda:host.HcuDetection(True,'gfx936','CPU lifecycle probe double')
    os.environ['HCU_VLLM_MHC_IMPL']='torch'
    assert m.install()
    os.environ['HCU_VLLM_MHC_IMPL']='invalid'
    try:m.install()
    except m.MhcCompatibilityError:pass
    else:raise AssertionError('invalid policy did not latch a publication failure')
    os.environ['HCU_VLLM_MHC_IMPL']='torch'
    for action in (m.require_healthy,m.install):
        try:action()
        except m.MhcCompatibilityError:pass
        else:raise AssertionError('restored policy recovered after a publication failure')
    return dict(status='CPU_INSTALLED_INVALID_POLICY_LATCH_PASS',pid=os.getpid(),
        module_sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
        invalid_direct_retry_latched=True,restored_policy_refused=True)


def _exercise():
    from hcu_adapter.bridges import vllm as host
    from hcu_adapter.bridges import vllm_mhc as m
    from vllm.config import VllmConfig,set_current_vllm_config
    from vllm import envs
    from vllm.config.utils import hash_factors
    from vllm.compilation.caching import aot_compile_hash_factors
    from vllm.model_executor.layers.mhc import MHCPreOp,MHCPostOp,MHCFusedPostPreOp,HCHeadOp
    envs.disable_envs_cache()
    # Only this CPU probe bypasses hardware identity. Production plugin still
    # checks real HCU and the canonical native/version guards before install.
    host.detect_hcu=lambda:host.HcuDetection(True,'gfx936','CPU lifecycle probe double')
    os.environ['HCU_VLLM_MHC_IMPL']='stock'
    assert not m.install()
    with set_current_vllm_config(VllmConfig()):
        assert type(MHCPreOp()) is MHCPreOp and type(MHCFusedPostPreOp()) is MHCFusedPostPreOp
    config=type('Config',(),{'compute_hash':lambda self:'fixed-config-hash'})()
    off=envs.compile_factors();off_aot=aot_compile_hash_factors(config)[0]
    assert m.COMPILE_FACTOR not in off and off_aot==hash_factors(off)
    os.environ['HCU_DISABLE_OPS']='vllm_mhc';os.environ['HCU_VLLM_MHC_IMPL']='torch'
    assert not m.install() and envs.compile_factors()==off
    os.environ.pop('HCU_DISABLE_OPS')
    assert m.install();m.require_healthy()
    with set_current_vllm_config(VllmConfig()):
        assert type(MHCPreOp()).__name__=='HcuMHCPreOp'
        assert type(MHCFusedPostPreOp()).__name__=='HcuMHCFusedPostPreOp'
        assert type(MHCPostOp()) is MHCPostOp and type(HCHeadOp()) is HCHeadOp
    on=envs.compile_factors();on_aot=aot_compile_hash_factors(config)[0]
    assert on[m.COMPILE_FACTOR].startswith('official-torch:')
    assert hash_factors(off)!=hash_factors(on) and on_aot!=off_aot and on_aot==hash_factors(on)
    envs.enable_envs_cache()
    assert envs.compile_factors()==on and aot_compile_hash_factors(config)[0]==on_aot
    assert envs.__getattr__(m.COMPILE_FACTOR)==on[m.COMPILE_FACTOR]
    envs.disable_envs_cache()
    fork_result=None
    if hasattr(os,'fork'):
        readfd,writefd=os.pipe();pid=os.fork()
        if pid==0:
            os.close(readfd)
            try:
                try:m.require_healthy()
                except m.MhcCompatibilityError:pass
                else:raise AssertionError('fork trusted parent publication')
                assert m.install();m.require_healthy();payload=dict(adopted=True,pid=os.getpid())
            except BaseException as error:payload=dict(error=type(error).__name__+': '+str(error))
            os.write(writefd,json.dumps(payload).encode());os.close(writefd);os._exit(0)
        os.close(writefd);raw=os.read(readfd,65536);os.close(readfd)
        _,status=os.waitpid(pid,0);assert os.waitstatus_to_exitcode(status)==0
        fork_result=json.loads(raw);assert fork_result.get('adopted')
    # Simulate a foreign constructor through its public allocation method;
    # production health neither reads nor repairs the private OOT registry.
    original_new=MHCPreOp.__new__
    MHCPreOp.__new__=staticmethod(lambda cls,*args,**kwargs:object.__new__(cls))
    refused=False
    try:m.require_healthy()
    except m.MhcCompatibilityError:refused=True
    assert refused
    MHCPreOp.__new__=original_new
    for action in (m.require_healthy,m.install):
        try:action()
        except m.MhcCompatibilityError:pass
        else:raise AssertionError('late drift recovered after restoring constructor')
    path=Path(m.__file__)
    return dict(status='CPU_INSTALLED_LIFECYCLE_PASS',pid=os.getpid(),module=str(path),
        module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),hardware_identity_double=True,
        stock_and_disable_preserve_originals=True,post_and_head_preserved=True,
        ordinary_hash_differs=True,aot_hash_differs=True,cached_env_getter_preserved=True,
        factor=on[m.COMPILE_FACTOR],fork=fork_result,constructor_drift_latched=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('HIP_VISIBLE_DEVICES')==os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not os.environ.get('PYTHONPATH')
    a.output.mkdir(exist_ok=False)
    ctx=multiprocessing.get_context('spawn');queue=ctx.Queue();process=ctx.Process(target=_child,args=(queue,))
    process.start();child=queue.get(timeout=60);process.join(timeout=60)
    assert process.exitcode==0 and child.get('status')=='CPU_INSTALLED_LIFECYCLE_PASS',child
    policy_queue=ctx.Queue();policy_process=ctx.Process(target=_child,args=(policy_queue,True))
    policy_process.start();policy=policy_queue.get(timeout=60);policy_process.join(timeout=60)
    assert policy_process.exitcode==0 and policy.get('status')=='CPU_INSTALLED_INVALID_POLICY_LATCH_PASS',policy
    parent=_exercise();assert parent['factor']==child['factor'] and parent['module']==child['module']
    assert policy['module_sha256']==parent['module_sha256']
    record=dict(status='CPU_INSTALLED_SPAWN_FORK_CACHE_DRIFT_PASS',parent=parent,spawn=child,invalid_policy=policy,
        production_changed=False,gpu_or_model_acceptance=False)
    (a.output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    (a.output/'driver.py').write_bytes(Path(__file__).read_bytes())
    print(record['status'])


if __name__=='__main__':
    main()
