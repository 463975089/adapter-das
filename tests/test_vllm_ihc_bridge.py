# SPDX-License-Identifier: Apache-2.0
"""Provider ABI and BF16 storage boundaries; no GPU or real vendor numerics."""
import importlib.util
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType
import torch


def bridge():
    explicit=os.environ.get('HCU_IHC_BRIDGE_SOURCE')
    if explicit is None:
        return importlib.import_module('hcu_adapter.bridges.vllm_ihc')
    path=Path(explicit)
    spec=importlib.util.spec_from_file_location('ihc_bridge_candidate',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def install(monkeypatch):
    calls=[]
    parent=ModuleType('boltops');parent.__path__=[]
    vendor=ModuleType('boltops.ihc')
    def pre(x,weight,scale,base,rms_eps,hc_eps,magnitude):
        calls.append(('pre',x.dtype,weight,scale,base,rms_eps,hc_eps,magnitude))
        return x.sum(1).to(x.dtype),torch.ones(x.shape[:2],dtype=torch.float32)
    def post(x,residual,gates):
        calls.append(('post',x.dtype,residual.dtype,gates.dtype))
        return (x.float().unsqueeze(1)*gates.unsqueeze(-1)+residual.float()).to(residual.dtype)
    vendor.ihc_pre=pre;vendor.ihc_post=post
    monkeypatch.setitem(sys.modules,'boltops',parent);monkeypatch.setitem(sys.modules,'boltops.ihc',vendor)
    vllm=ModuleType('vllm');vllm.__path__=[]
    ops=ModuleType('vllm._custom_ops')
    def rms(out,x,weight,eps):
        calls.append(('rms',x.dtype,weight,eps));out.copy_(x)
    ops.rms_norm=rms
    monkeypatch.setitem(sys.modules,'vllm',vllm);monkeypatch.setitem(sys.modules,'vllm._custom_ops',ops)
    return calls


def test_pre_keeps_native_argument_order(monkeypatch):
    calls=install(monkeypatch);module=bridge()
    x=torch.ones(2,4,3,dtype=torch.bfloat16)
    result=module.pre(x,'weight','scale','base',2.5,1e-6,1e-5)
    assert calls==[('pre',torch.bfloat16,'weight','scale','base',1e-5,1e-6,2.5)]
    assert result[0].dtype==torch.bfloat16 and result[1].dtype==torch.float32


def test_chain_preserves_bf16_rounding_and_returns_fresh_storage(monkeypatch):
    calls=install(monkeypatch);module=bridge()
    x=torch.full((2,3),.123,dtype=torch.bfloat16)
    residual=torch.full((2,4,3),.234,dtype=torch.bfloat16)
    gates=torch.full((2,4),.345,dtype=torch.float32)
    originals=[t.clone() for t in [x,residual,gates]]
    out,post,updated=module.post_pre_rms_norm(x,residual,gates,'weight','scale','base','norm',2.5,1e-6,1e-5)
    assert calls==[('post',torch.bfloat16,torch.bfloat16,torch.float32),
                  ('pre',torch.bfloat16,'weight','scale','base',1e-5,1e-6,2.5),
                  ('rms',torch.bfloat16,'norm',1e-5)]
    assert out.dtype==updated.dtype==torch.bfloat16 and post.dtype==torch.float32
    expected=(x.float().unsqueeze(1)*gates.unsqueeze(-1)+residual.float()).to(torch.bfloat16)
    assert torch.equal(updated,expected) and torch.equal(out,expected.sum(1).to(torch.bfloat16))
    assert all(torch.equal(a,b) for a,b in zip([x,residual,gates],originals))
    assert all(t.data_ptr()!=out.data_ptr() and t.data_ptr()!=updated.data_ptr() for t in [x,residual,gates])
