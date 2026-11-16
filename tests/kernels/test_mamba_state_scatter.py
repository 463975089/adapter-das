# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Mamba state copy vs independent Torch reference; no host-framework imports.
Run: HIP_VISIBLE_DEVICES=0 HCU_ADAPTER_HOST=none pytest -q tests/kernels/test_mamba_state_scatter.py
The state copy must preserve bits, source rows, masked slots and changed-input Graph.
"""

import math
import pytest
import torch
from hcu_adapter.kernels.mamba_state_scatter_triton import _fused_mamba_state_scatter_flat_kernel as kernel
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="requires assigned HCU")


def launch(dst,src,di,st,sr):
    n=st.numel()
    if n==0:return
    elem=math.prod(dst.shape[2:])
    kernel[(n*((elem+1023)//1024),dst.shape[0],1)](
        src,dst,di,st,sr,elem,src.stride(0),src.stride(1),src.stride(2),
        dst.stride(0),dst.stride(1),src.shape[1],src.shape[2],dst.shape[1],BLOCK_SIZE=1024)


def reference(storage,src,di,st,sr):
    ref=storage.clone()
    for d,s,row in zip(di,st,sr):
        if 0<=d<ref.shape[1]//2 and 0<=s<src.shape[2] and 0<=row<src.shape[1]:
            ref[:,2*d].copy_(src[:,row,s])
    return ref


CASES=[(0,7,"valid"),(1,1023,"valid"),(1,1024,"valid"),(3,1025,"valid"),
       (3,1031,"half"),(3,1031,"invalid"),(16,3072,"valid"),(17,16385,"half"),(17,1025,"bounds")]


@pytest.mark.parametrize("n,entry,mask",CASES)
def test_copy_contract(n,entry,mask):
    torch.manual_seed(19)
    layers,rows,steps,slots=3,max(n+2,7),4,max(n+4,9)
    src=torch.randn(layers,rows,steps,entry,device="cuda",dtype=torch.float32)
    src.view(torch.int32).flatten()[:4].copy_(
        torch.tensor([0,-2147483648,2139095040,2143294004],device="cuda",dtype=torch.int32))
    src_before=src.clone()
    storage=torch.full((layers,slots*2,entry),-13.,device="cuda")
    dst=storage[:,::2]
    di=list(range(n));sr=[(rows-1-i)%rows for i in range(n)];st=[i%steps for i in range(n)]
    if mask=="half":st=[s if i%2==0 else -1 for i,s in enumerate(st)]
    if mask=="invalid":st=[-1]*n
    if mask=="bounds":
        di[0]=-1;di[1]=slots;sr[2]=-1;sr[3]=rows;st[4]=steps
    ref=reference(storage,src,di,st,sr)
    ids=[torch.tensor(v,device="cuda",dtype=torch.int32) for v in [di,st,sr]]
    launch(dst,src,*ids);torch.cuda.synchronize()
    assert torch.equal(storage.view(torch.int32),ref.view(torch.int32))
    assert torch.equal(src.view(torch.int32),src_before.view(torch.int32))


def test_real_tp2_state_shape_and_permuted_source_rows():
    torch.manual_seed(42)
    shape=(24,128,128)
    src=torch.randn(48,18,4,*shape,device="cuda",dtype=torch.float32)
    storage=torch.full((48,40,*shape),-13.,device="cuda")
    dst=storage[:,::2]
    di=list(range(16));sr=list(range(15,-1,-1));st=[i%4 for i in range(16)]
    ref=reference(storage,src,di,st,sr)
    ids=[torch.tensor(x,device="cuda",dtype=torch.int32) for x in [di,st,sr]]
    launch(dst,src,*ids);torch.cuda.synchronize()
    assert torch.equal(storage,ref)


def test_graph_replay_changes_values_slots_steps_and_source_rows():
    src=torch.randn(3,9,4,1031,device="cuda",dtype=torch.float32)
    storage=torch.full((3,20,1031),-13.,device="cuda")
    dst=storage[:,::2]
    lists=[[1,3,7],[0,1,3],[8,3,0]]
    ids=[torch.tensor(x,device="cuda",dtype=torch.int32) for x in lists]
    launch(dst,src,*ids)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):launch(dst,src,*ids)
    storage.fill_(-13.);graph.replay();torch.cuda.synchronize()
    expected=reference(torch.full_like(storage,-13.),src,*lists)
    assert torch.equal(storage,expected)
    for wave in range(3):
        src.uniform_(-2-wave,2+wave)
        lists=[[wave,4,8],[3-wave,-1,wave],[wave,8-wave,2]]
        for target,v in zip(ids,lists):target.copy_(torch.tensor(v,device="cuda",dtype=torch.int32))
        storage.fill_(-13.);graph.replay();torch.cuda.synchronize()
        expected=reference(torch.full_like(storage,-13.),src,*lists)
        assert torch.equal(storage,expected),wave
