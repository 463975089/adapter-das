# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Offline mHC publication/failure contracts; real installed/GPU probes separate.

Run: HIP_VISIBLE_DEVICES='' HCU_ADAPTER_HOST=none python -m pytest
tests/adapter/test_vllm_mhc_bridge.py. No framework/provider math is executed.
"""
import importlib
import os
from types import SimpleNamespace

import pytest

from hcu_adapter.bridges import vllm as host
from hcu_adapter.bridges import vllm_mhc as module
from hcu_adapter import registry


@pytest.fixture
def seams(monkeypatch):
    m = importlib.reload(module)
    monkeypatch.setenv('HCU_VLLM_MHC_IMPL','torch')
    monkeypatch.delenv('HCU_DISABLE_OPS',raising=False)
    monkeypatch.setattr(host,'detect_hcu',lambda:host.HcuDetection(True,'gfx936','test'))
    monkeypatch.setattr(host,'_installed_vllm_version',lambda:'0.33.0+compatible')
    selection={}; calls=[]; records=[]
    class Base:
        def __new__(cls):return object.__new__(selection.get(cls,cls))
        @classmethod
        def register_oot(cls,replacement):selection[cls]=replacement;return replacement
    class Pre(Base):
        def forward_hip(self,residual,fn,hc_scale,hc_base,rms_eps,hc_pre_eps,hc_sinkhorn_eps,
                        hc_post_mult_value,sinkhorn_repeat,n_splits=1,norm_weight=None,norm_eps=0.0):pass
    class Fused(Base):
        def forward_hip(self,x,residual,post_layer_mix,comb_res_mix,fn,hc_scale,hc_base,
                        rms_eps,hc_pre_eps,hc_sinkhorn_eps,hc_post_mult_value,sinkhorn_repeat,
                        n_splits=1,tile_n=1,norm_weight=None,norm_eps=0.0):pass
    class Delayed(Base):
        def _maybe_post(self,residual,sublayer_out,post_layer_mix,comb_res_mix):return residual
        def forward_hip(self,residual,fn,hc_scale,hc_base,rms_eps,hc_pre_eps,hc_sinkhorn_eps,
                        hc_post_mult_value,sinkhorn_repeat,pre_mix=None,x=None,norm_weight=None,
                        norm_eps=1e-6,sublayer_out=None,post_layer_mix=None,comb_res_mix=None):pass
    class Post(Base):
        def forward_hip(self,x,residual,post_layer_mix,comb_res_mix):pass
    def pre(residual,fn,hc_scale,hc_base,rms_eps,hc_pre_eps,hc_sinkhorn_eps,hc_post_mult_value,
            sinkhorn_repeat,n_splits=1):raise AssertionError('offline math forbidden')
    def post(x,residual,post_layer_mix,comb_res_mix):raise AssertionError('offline math forbidden')
    def register(name,function,*,mutates_args,fake_impl):calls.append(name)
    class ReplacementPre(Pre):pass
    class ReplacementFused(Fused):pass
    class ReplacementDelayed(Delayed):pass
    class ReplacementPost(Post):pass
    mhc=SimpleNamespace(MHCPreOp=Pre,MHCFusedPostPreOp=Fused,
                        MHCPreDelayedOp=Delayed,MHCPostOp=Post)
    envs=SimpleNamespace(environment_variables={})
    kernels=SimpleNamespace(mhc_pre_torch=pre,mhc_post_torch=post)
    modules={'vllm.model_executor.layers.mhc':mhc,
        'vllm.model_executor.kernels.mhc.torch':kernels,'vllm.ir':object(),'vllm.envs':envs,
        'vllm.ir.ops.layernorm':object(),
        'vllm.utils.torch_utils':SimpleNamespace(direct_register_custom_op=register)}
    monkeypatch.setattr(m,'import_module',modules.__getitem__)
    monkeypatch.setattr(m.inspect,'getfile',lambda obj:__file__)
    monkeypatch.setattr(m,'_declare',lambda *args:((pre,object(),ReplacementPre),
        (post,object(),ReplacementFused),(None,None,ReplacementPost),(None,None,ReplacementDelayed)))
    monkeypatch.setattr(m,'torch',SimpleNamespace(__version__='test',version=SimpleNamespace(hip='test')),raising=False)
    monkeypatch.setattr(registry,'register',lambda **kw:records.append(kw))
    yield SimpleNamespace(m=m,selection=selection,calls=calls,records=records,mhc=mhc,envs=envs,modules=modules)
    # Do not let an offline publication affect unrelated host tests.
    importlib.reload(module)


@pytest.mark.parametrize('arch',['gfx936','gfx938'])
def test_stock_and_kill_switch_import_no_framework(seams,monkeypatch,arch):
    s=seams
    monkeypatch.setattr(host,'detect_hcu',lambda:host.HcuDetection(True,arch,'test'))
    monkeypatch.setenv('HCU_VLLM_MHC_IMPL','stock')
    monkeypatch.setattr(s.m,'import_module',lambda name:pytest.fail('stock imported '+name))
    assert not s.m.install()
    s.m.require_healthy()
    monkeypatch.setenv('HCU_VLLM_MHC_IMPL','torch');monkeypatch.setenv('HCU_DISABLE_OPS','vllm_mhc')
    assert not s.m.install() and not s.calls


def test_kill_switch_precedes_invalid_mode(seams,monkeypatch):
    monkeypatch.setenv('HCU_VLLM_MHC_IMPL','invalid')
    monkeypatch.setenv('HCU_DISABLE_OPS','vllm_mhc')
    assert not seams.m.install()
    seams.m.require_healthy()
    assert not seams.calls


def test_partial_framework_import_baseexception_cannot_be_retried(seams,monkeypatch):
    def broken_import(name):
        raise SystemExit('framework import interrupted')
    monkeypatch.setattr(seams.m,'import_module',broken_import)
    with pytest.raises(seams.m.MhcCompatibilityError,match='activation failed'):
        seams.m.install()
    for action in (seams.m.require_healthy,seams.m.install):
        with pytest.raises(seams.m.MhcCompatibilityError):action()
    assert not seams.calls


def test_publication_once_and_stable_cache_factor(seams):
    s=seams
    assert s.m.install() and s.m.install()
    assert s.calls==['hcu_adapter_mhc_pre','hcu_adapter_mhc_fused']
    assert s.records[0]['mechanism']==registry.MECH_SEAM
    assert s.records[0]['op']=='vllm_mhc'
    assert s.envs.environment_variables[s.m.COMPILE_FACTOR]().startswith('official-torch:')


@pytest.mark.parametrize('arch',['gfx936','gfx938'])
def test_admitted_arch_publishes_official_torch_candidate(seams,monkeypatch,arch):
    s=seams
    monkeypatch.setattr(host,'detect_hcu',lambda:host.HcuDetection(True,arch,'test'))
    assert s.m.install()
    s.m.require_healthy()
    assert s.calls==['hcu_adapter_mhc_pre','hcu_adapter_mhc_fused']
    assert len(s.records)==1 and s.records[0]['impl']=='torch'
    assert type(s.mhc.MHCPreOp()) is s.selection[s.mhc.MHCPreOp]
    assert type(s.mhc.MHCFusedPostPreOp()) is s.selection[s.mhc.MHCFusedPostPreOp]
    assert s.envs.environment_variables[s.m.COMPILE_FACTOR]().startswith('official-torch:')


@pytest.mark.parametrize('arch',['gfx928','gfx92a','gfx942',None])
def test_unvalidated_arch_rejected_before_import(seams,monkeypatch,arch):
    monkeypatch.setattr(host,'detect_hcu',lambda:host.HcuDetection(True,arch,'test'))
    monkeypatch.setattr(seams.m,'import_module',lambda name:pytest.fail('unvalidated arch imported '+name))
    with pytest.raises(seams.m.MhcCompatibilityError) as failure:seams.m.install()
    assert f'got detected=True, arch={arch!r}' in str(failure.value)
    assert not seams.calls and not seams.records


@pytest.mark.parametrize('arch',['gfx936','gfx938'])
def test_arch_name_without_detected_hcu_rejected_before_import(seams,monkeypatch,arch):
    monkeypatch.setattr(host,'detect_hcu',lambda:host.HcuDetection(False,arch,'test'))
    monkeypatch.setattr(seams.m,'import_module',lambda name:pytest.fail('non-HCU imported '+name))
    with pytest.raises(seams.m.MhcCompatibilityError) as failure:seams.m.install()
    assert f'got detected=False, arch={arch!r}' in str(failure.value)
    assert not seams.calls and not seams.records


@pytest.mark.parametrize('version',['0.32.1','1!0.33.0','0.34.0'])
def test_feature_release_line_does_not_widen_host_admission(seams,monkeypatch,version):
    monkeypatch.setattr(host,'_installed_vllm_version',lambda:version)
    with pytest.raises(seams.m.MhcCompatibilityError,match='0.33'):seams.m.install()
    assert not seams.calls


def test_abi_change_and_preexisting_owner_fail_before_publication(seams,monkeypatch):
    s=seams;monkeypatch.setattr(s.mhc.MHCPreOp,'forward_hip',lambda self,unknown:None)
    with pytest.raises(s.m.MhcCompatibilityError,match='ABI'):s.m.install()
    assert not s.calls and not s.records


def test_preexisting_owner_is_not_replaced(seams):
    s=seams
    class Foreign(s.mhc.MHCPreOp):pass
    s.selection[s.mhc.MHCPreOp]=Foreign
    with pytest.raises(s.m.MhcCompatibilityError,match='another owner'):s.m.install()
    assert s.selection[s.mhc.MHCPreOp] is Foreign and not s.calls


@pytest.mark.parametrize('mutation',['clear','foreign','factor','policy'])
def test_late_drift_refuses_read_and_retry_without_repair(seams,monkeypatch,mutation):
    s=seams;s.m.install();original=s.mhc.MHCPreOp
    if mutation=='clear':s.selection.clear()
    elif mutation=='foreign':
        class Foreign(original):pass
        s.selection[original]=Foreign
    elif mutation=='factor':s.envs.environment_variables[s.m.COMPILE_FACTOR]=lambda:'foreign'
    else:monkeypatch.setenv('HCU_VLLM_MHC_IMPL','stock')
    with pytest.raises(s.m.MhcCompatibilityError):s.m.require_healthy()
    with pytest.raises(s.m.MhcCompatibilityError):s.m.install()
    assert len(s.calls)==2


def test_invalid_policy_direct_retry_latches_before_restoring_mode(seams,monkeypatch):
    s=seams
    s.m.install()
    monkeypatch.setenv('HCU_VLLM_MHC_IMPL','invalid')
    with pytest.raises(s.m.MhcCompatibilityError,match='policy invalid'):
        s.m.install()
    monkeypatch.setenv('HCU_VLLM_MHC_IMPL','torch')
    for action in (s.m.require_healthy,s.m.install):
        with pytest.raises(s.m.MhcCompatibilityError):action()
    assert len(s.calls)==2


def test_invalid_policy_before_publication_does_not_poison_process(seams,monkeypatch):
    monkeypatch.setenv('HCU_VLLM_MHC_IMPL','invalid')
    with pytest.raises(ValueError):seams.m.install()
    assert not seams.calls and not seams.records
    monkeypatch.setenv('HCU_VLLM_MHC_IMPL','torch')
    assert seams.m.install()
    seams.m.require_healthy()


@pytest.mark.parametrize('format_error',[AssertionError,SystemExit])
def test_partial_publication_latches_before_bad_exception_format(seams,format_error):
    s=seams
    class BadError(RuntimeError):
        def __str__(self):
            assert s.m._failure is not None
            with pytest.raises(s.m.MhcCompatibilityError):s.m.require_healthy()
            raise format_error('diagnostic formatting failed')
    def broken(name,function,**kwargs):
        s.calls.append(name)
        if len(s.calls)==2:raise BadError()
    s.modules['vllm.utils.torch_utils'].direct_register_custom_op=broken
    with pytest.raises(s.m.MhcCompatibilityError,match='publication failed'):s.m.install()
    with pytest.raises(s.m.MhcCompatibilityError):s.m.require_healthy()
    with pytest.raises(s.m.MhcCompatibilityError):s.m.install()
    assert len(s.calls)==2 and not s.records


@pytest.mark.skipif(not hasattr(os,'fork'),reason='fork required')
def test_publication_failure_is_inherited_during_exception_formatting(seams):
    s=seams;children=[]
    class ForkError(RuntimeError):
        def __str__(self):
            assert s.m._failure is not None
            pid=os.fork()
            if pid==0:
                try:
                    for action in (s.m.require_healthy,s.m.install):
                        with pytest.raises(s.m.MhcCompatibilityError):action()
                except BaseException:os._exit(1)
                os._exit(0)
            _,status=os.waitpid(pid,0)
            children.append(os.waitstatus_to_exitcode(status))
            return 'fork refused after publication failure'
    def broken(name,function,**kwargs):
        s.calls.append(name)
        raise ForkError()
    s.modules['vllm.utils.torch_utils'].direct_register_custom_op=broken
    with pytest.raises(s.m.MhcCompatibilityError):s.m.install()
    assert children==[0] and len(s.calls)==1
    with pytest.raises(s.m.MhcCompatibilityError):s.m.require_healthy()


@pytest.mark.skipif(not hasattr(os,'fork'),reason='fork required')
def test_successful_fork_requires_child_activation_then_adopts_without_registering(seams):
    s=seams;s.m.install()
    pid=os.fork()
    if pid==0:
        try:
            with pytest.raises(s.m.MhcCompatibilityError,match='another PID'):s.m.require_healthy()
            s.m.install();s.m.require_healthy();assert len(s.calls)==2
        except BaseException:os._exit(1)
        os._exit(0)
    _,status=os.waitpid(pid,0);assert os.waitstatus_to_exitcode(status)==0
    s.m.require_healthy()


def _clean_mode_environment(monkeypatch):
    """Remove every knob that could outrank the registry default."""
    for name in ('HCU_VLLM_MHC_IMPL','HCU_DISABLE_OPS','HCU_CONFIG'):
        monkeypatch.delenv(name,raising=False)


def test_default_mode_is_aiter_without_any_environment(monkeypatch):
    """Branch default: an unconfigured process selects the AITER tier."""
    _clean_mode_environment(monkeypatch)
    value,source=module.hcu_env.resolve(module.MODE_VARIABLE)
    assert value=='aiter' and source=='default'
    assert module.selected_mode()=='aiter'
    assert module.selected_split_k('aiter')=='auto'


def test_tilelang_stage2_is_default_on_with_explicit_opt_out(monkeypatch):
    _clean_mode_environment(monkeypatch)
    monkeypatch.delenv('VLLM_HCU_USE_AITER_TILELANG_MHC',raising=False)
    value,source=module.hcu_env.resolve('VLLM_HCU_USE_AITER_TILELANG_MHC')
    assert value=='1' and source=='default'
    assert module.hcu_env.enabled('VLLM_HCU_USE_AITER_TILELANG_MHC')
    monkeypatch.setenv('VLLM_HCU_USE_AITER_TILELANG_MHC','0')
    assert not module.hcu_env.enabled('VLLM_HCU_USE_AITER_TILELANG_MHC')


def test_kill_switch_still_forces_stock_over_the_default(monkeypatch):
    """HCU_DISABLE_OPS outranks the new default, not the other way round."""
    _clean_mode_environment(monkeypatch)
    monkeypatch.setenv('HCU_DISABLE_OPS','vllm_mhc')
    assert module.selected_mode()=='stock'


def test_explicit_stock_still_opts_out(monkeypatch):
    """Explicit HCU_VLLM_MHC_IMPL=stock remains a valid opt-out."""
    _clean_mode_environment(monkeypatch)
    monkeypatch.setenv('HCU_VLLM_MHC_IMPL','stock')
    assert module.selected_mode()=='stock'


# The AR host check runs INSIDE the host runtime callback, i.e. before the
# owner install steps that same callback performs. Owner aggregation there is
# a boot-order fact, not a failure; the post-activation gate stays the real
# "requested but never activated" defense.
_OWNER_BOOT_CASES=(
    ('mhc-aiter',{'HCU_VLLM_MHC_IMPL':'aiter','HCU_VLLM_SPARSE_INDEXER_IMPL':'stock'}),
    ('indexer-native',{'HCU_VLLM_MHC_IMPL':'stock','HCU_VLLM_SPARSE_INDEXER_IMPL':'native'}),
)


def _owner_modules(monkeypatch,env):
    monkeypatch.delenv('HCU_DISABLE_OPS',raising=False)
    monkeypatch.delenv('HCU_CONFIG',raising=False)
    for name,value in env.items():
        monkeypatch.setenv(name,value)
    importlib.reload(module)
    indexer=importlib.import_module('hcu_adapter.bridges.vllm_sparse_indexer')
    importlib.reload(indexer)
    assert module._publication is None and indexer._publication is None
    return indexer


@pytest.fixture
def booting(monkeypatch):
    """A fresh host lifecycle, so boot-order cases do not share latches."""
    actions=host.ProcessLocalActions()
    monkeypatch.setattr(host,'_RUNTIME_ACTIONS',actions)
    return actions


@pytest.mark.parametrize('label,env',_OWNER_BOOT_CASES)
def test_boot_time_host_health_check_precedes_owner_install(booting,monkeypatch,label,env):
    """RED before the ordering fix: the host check refused an uninstalled owner."""
    _owner_modules(monkeypatch,env)
    checked=[]
    def during_boot():
        host.require_runtime_healthy()
        checked.append(label)
    booting.run_once('vllm-runtime',during_boot)
    assert checked==[label]


@pytest.mark.parametrize('label,env',_OWNER_BOOT_CASES)
def test_post_activation_host_health_check_still_refuses_unactivated_owner(
        booting,monkeypatch,label,env):
    """The fix must not narrow the gate that runs after activation."""
    _owner_modules(monkeypatch,env)
    booting.run_once('vllm-runtime',lambda:None)
    with pytest.raises(RuntimeError,match='before runtime activation'):
        host.require_runtime_healthy()
