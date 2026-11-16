# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Mamba scatter binding, fallback, retained aliases and real Graph wrapper."""

import hashlib
import types
import pytest
from hcu_adapter import mamba_state_scatter as bridge


class Kernel:
    def __init__(self, name, error=None):
        self.name,self.error,self.calls=name,error,[]
        self.src="reviewed upstream source"
        self.arg_names=list(bridge._ARGUMENTS)

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid,args,kwargs))
            if self.error:raise self.error
            return self.name
        return launch


def make_torch(arch="gfx936:sramecc+:xnack-", error=False):
    probes=[]
    def properties(device):
        probes.append(device)
        if error:raise RuntimeError("probe unavailable")
        return types.SimpleNamespace(gcnArchName=arch)
    return types.SimpleNamespace(float32="fp32",cuda=types.SimpleNamespace(get_device_properties=properties),probes=probes)


def tensor(device="cuda:1", dtype="fp32", is_cuda=True):
    return types.SimpleNamespace(device=device,dtype=dtype,is_cuda=is_cuda)


def module(monkeypatch):
    mod=types.ModuleType(bridge._MODULE)
    old=Kernel("stock")
    mod.__dict__[bridge._SYMBOL]=old
    exec(
        "def fused_mamba_state_scatter_with_mask(dst,src,dst_indices_raw,step_indices_raw,src_indices_raw=None):\n"
        "    return _fused_mamba_state_scatter_with_mask_kernel[(16,48,384)]"
        "(src,dst,dst_indices_raw,step_indices_raw,src_indices_raw,393216,1,2,3,4,5,18,4,80,BLOCK_SIZE=1024)\n",
        mod.__dict__,
    )
    monkeypatch.setattr(bridge,"_UPSTREAM_SHA256",hashlib.sha256(old.src.encode()).hexdigest())
    return mod,old


def test_binding_retains_aliases_and_defers_device_probe(monkeypatch):
    mod,old=module(monkeypatch)
    alias=mod.fused_mamba_state_scatter_with_mask
    replacement,torch_api,records=Kernel("flat"),make_torch(),[]
    d=bridge._bind(mod,replacement,torch_api,lambda *a,**kw: records.append((a,kw)))
    assert torch_api.probes==[]
    rows=object()
    assert alias(tensor(),tensor(),[3],[1],rows)=="flat"
    assert alias is mod.fused_mamba_state_scatter_with_mask
    assert replacement.calls[0][0]==(6144,48,1)
    assert replacement.calls[0][1][4] is rows
    assert replacement.calls[0][2]=={"BLOCK_SIZE":1024}
    assert torch_api.probes==["cuda:1"]
    alias(tensor(),tensor(),[3],[1],rows)
    assert torch_api.probes==["cuda:1"]
    assert d.fast_calls==2 and old.calls==[]
    assert records[0][0][0]=="mamba_state_scatter"
    assert bridge._bind(mod,replacement,torch_api,lambda *a,**k: pytest.fail("registered twice")) is d


@pytest.mark.parametrize("drift",["source","jit_abi","wrapper_abi","namespace"])
def test_upstream_drift_refuses_without_mutation(monkeypatch,drift):
    mod,old=module(monkeypatch)
    if drift=="source":old.src+=" changed"
    elif drift=="jit_abi":old.arg_names.append("new_argument")
    elif drift=="wrapper_abi":mod.fused_mamba_state_scatter_with_mask=lambda dst,src: None
    else:
        fn=mod.fused_mamba_state_scatter_with_mask
        mod.fused_mamba_state_scatter_with_mask=types.FunctionType(fn.__code__,{})
    with pytest.raises(RuntimeError):
        bridge._bind(mod,Kernel("flat"),make_torch(),lambda *a,**kw: pytest.fail("registered"))
    assert getattr(mod,bridge._SYMBOL) is old


@pytest.mark.parametrize("case",["arch","probe","dtype","device","cpu","small"])
def test_decline_keeps_original_launch_and_arguments(case):
    old,new=Kernel("stock"),Kernel("flat")
    api=make_torch(arch="gfx938" if case=="arch" else "gfx936",error=case=="probe")
    src,dst=tensor(),tensor()
    if case=="dtype":src.dtype="bf16"
    if case=="device":dst.device="cuda:0"
    if case=="cpu":src.is_cuda=False
    args=(src,dst,object(),object(),object(),7 if case=="small" else 1025)
    d=bridge._GridDispatcher(old,new,api)
    assert d[(17,3,2)](*args,BLOCK_SIZE=1024)=="stock"
    assert old.calls==[((17,3,2),args,{"BLOCK_SIZE":1024})]
    assert new.calls==[]


def test_unsupported_grid_and_provider_error_do_not_retry():
    old,new=Kernel("stock"),Kernel("flat",RuntimeError("launched"))
    d=bridge._GridDispatcher(old,new,make_torch())
    assert d[(1,2)]()=="stock"
    with pytest.raises(RuntimeError,match="launched"):
        d[(16,48,384)](tensor(),tensor(),None,None,None,393216)
    assert len(old.calls)==1 and len(new.calls)==1


def test_registration_failure_rolls_back(monkeypatch):
    mod,old=module(monkeypatch)
    def fail(*a,**kw):raise RuntimeError("registry rejected")
    with pytest.raises(RuntimeError,match="registry rejected"):
        bridge._bind(mod,Kernel("flat"),make_torch(),fail)
    assert getattr(mod,bridge._SYMBOL) is old


def test_explicit_disable_precedes_import(monkeypatch):
    from hcu_adapter import kill_switch
    monkeypatch.setattr(kill_switch,"op_disabled",lambda op: op=="mamba_state_scatter")
    monkeypatch.setattr(bridge.importlib,"import_module",lambda *a,**k: pytest.fail("imported"))
    assert bridge.install() is False


def test_real_upstream_alias_graph_and_strided_indices(monkeypatch):
    import importlib
    import torch
    if not torch.cuda.is_available():pytest.skip("requires assigned HCU")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]!="gfx936":
        pytest.skip("flattened route validated on gfx936")
    from hcu_adapter import registry,kill_switch
    monkeypatch.setattr(kill_switch,"op_disabled",lambda op: False)
    monkeypatch.setattr(registry,"_RECORDS",{})
    monkeypatch.setattr(registry,"_BOUND",{})
    mod=importlib.import_module(bridge._MODULE)
    installed = getattr(mod, bridge._SYMBOL)
    old = getattr(installed, "original", installed)
    alias = mod.fused_mamba_state_scatter_with_mask
    # Full-suite startup may have already installed the production route.
    # Isolate this fixture, then restore that exact binding on exit.
    setattr(mod, bridge._SYMBOL, old)
    try:
        dispatch=bridge.install()
        assert isinstance(dispatch,bridge._GridDispatcher)
        assert alias is mod.fused_mamba_state_scatter_with_mask
        src=torch.randn(3,7,4,1031,device="cuda",dtype=torch.float32)
        store=torch.full((3,18,1031),-13.,device="cuda")
        dst=store[:,::2]
        di_base=torch.arange(6,device="cuda",dtype=torch.int64)
        st_base=torch.tensor([0,9,1,9,2,9],device="cuda",dtype=torch.int64)
        sr_base=torch.tensor([6,9,2,9,4,9],device="cuda",dtype=torch.int64)
        di,st,sr=di_base[::2],st_base[::2],sr_base[::2]
        alias(dst,src,di,st,sr)
        assert dispatch.fast_calls==1
        dispatch.device_arches.clear()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):alias(dst,src,di,st,sr)
        def reference():
            ref=torch.full_like(store,-13.)
            for d,s,row in zip(di.cpu().tolist(),st.cpu().tolist(),sr.cpu().tolist()):
                if 0<=d<dst.shape[1] and 0<=s<4 and 0<=row<7:
                    ref[:,2*d].copy_(src[:,row,s])
            return ref
        store.fill_(-13.);graph.replay();torch.cuda.synchronize()
        assert torch.equal(store,reference())
        for wave in range(3):
            src.uniform_(-3.-wave,3.+wave)
            di.copy_(torch.tensor([1,4,7],device="cuda",dtype=torch.int64))
            st.copy_(torch.tensor([3-wave,-1,wave],device="cuda",dtype=torch.int64))
            sr.copy_(torch.tensor([wave,6-wave,2],device="cuda",dtype=torch.int64))
            store.fill_(-13.);graph.replay();torch.cuda.synchronize()
            assert torch.equal(store,reference()),wave
        assert dispatch.fast_calls>=2 and dispatch.stock_calls==0
    finally:
        setattr(mod, bridge._SYMBOL, installed)

@pytest.mark.parametrize("disabled", ["mamba_state_scatter", "aiter_ar;mamba_state_scatter"])
def test_fresh_process_explicit_disable_uses_registered_environment(disabled):
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env.update(
        HCU_ADAPTER_HOST="none", _HCU_ADAPTER_ACTIVE_HOST="none",
        HCU_DISABLE_OPS=disabled, HIP_VISIBLE_DEVICES="",
    )
    code = """
import builtins
from hcu_adapter import mamba_state_scatter
original_import = builtins.__import__
def refuse(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "sglang", "triton"}:
        raise AssertionError("disabled route imported a provider: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = refuse
assert mamba_state_scatter.install() is False
print("DISABLED_BEFORE_PROVIDER_IMPORT")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True,
        text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DISABLED_BEFORE_PROVIDER_IMPORT" in result.stdout


def _spawn_state_copy(queue):
    try:
        import importlib
        import torch
        from hcu_adapter import mamba_state_scatter as route

        torch.cuda.set_device(0)
        mod = importlib.import_module(route._MODULE)
        alias = mod.fused_mamba_state_scatter_with_mask
        dispatch = route.install()
        src = torch.arange(3 * 4 * 1031, device="cuda", dtype=torch.float32)
        src = src.reshape(1, 3, 4, 1031)
        dst = torch.full((1, 5, 1031), -1.0, device="cuda")
        indices = [
            torch.tensor(values, device="cuda", dtype=torch.int32)
            for values in ([1, 4], [3, 1], [2, 0])
        ]
        alias(dst, src, *indices)
        expected = torch.full_like(dst, -1.0)
        expected[:, 1].copy_(src[:, 2, 3])
        expected[:, 4].copy_(src[:, 0, 1])
        torch.cuda.synchronize()
        queue.put({
            "correct": torch.equal(dst, expected),
            "alias_retained": alias is mod.fused_mamba_state_scatter_with_mask,
            "fast_calls": dispatch.fast_calls,
        })
    except BaseException as exc:
        queue.put({"error": repr(exc)})
        raise


def test_spawn_process_installs_route_and_preserves_source_rows(monkeypatch):
    import multiprocessing
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires assigned HCU")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx936":
        pytest.skip("flattened route validated on gfx936")
    # Child explicitly selects its device before the bridge loads providers.
    monkeypatch.setenv("HCU_ADAPTER_HOST", "none")
    monkeypatch.setenv("_HCU_ADAPTER_ACTIVE_HOST", "none")
    monkeypatch.delenv("HCU_DISABLE_OPS", raising=False)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_spawn_state_copy, args=(queue,))
    process.start()
    process.join(120)
    if process.is_alive():
        process.terminate()
        process.join(15)
        pytest.fail("state-scatter spawn probe timed out")
    assert process.exitcode == 0
    result = queue.get(timeout=10)
    queue.close()
    queue.join_thread()
    assert result == {"correct": True, "alias_retained": True, "fast_calls": 1}
