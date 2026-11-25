# SPDX-License-Identifier: Apache-2.0
"""Deterministic small-row BF16 router projection with FP32 accumulation.

The six 64-expert blocks start their K traversal 128 elements apart, matching
our native Tensile BSS router reduction order. The K512 load tile uses two
software-pipeline stages and wraps element offsets before loading, preserving
the same ordered MFMA accumulations even at 128-element stagger boundaries.
Tiling is independent of request
and prompt lengths. Only the qualified model dimensions / dtype / device use
this implementation; consumers must retain their original unsupported path.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _router_stagger_pipelined_gate(A,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                 BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
                 STAGGER:tl.constexpr):
    rm=tl.program_id(0)*BM+tl.arange(0,BM)
    rn=tl.program_id(1)*BN+tl.arange(0,BN)
    rk=tl.arange(0,BK)
    accum=tl.zeros((BM,BN),tl.float32)
    offset=((tl.program_id(1)*BN)//64)*STAGGER
    kk=(offset+rk)%K
    for step in tl.range(tl.cdiv(K,BK),num_stages=2):
        a=tl.load(A+rm[:,None].to(tl.int64)*K+kk[None,:],
                  (rm[:,None]<M)&(kk[None,:]<K),other=0)
        b=tl.load(W+rn[None,:].to(tl.int64)*K+kk[:,None],
                  (rn[None,:]<N)&(kk[:,None]<K),other=0)
        accum=tl.dot(a,b,accum)
        kk+=BK
        kk=tl.where(kk<K,kk,kk-K)
    tl.store(Y+rm[:,None].to(tl.int64)*N+rn[None,:],accum,
             (rm[:,None]<M)&(rn[None,:]<N))


def supported(a,w):
    if (not isinstance(a,torch.Tensor) or not isinstance(w,torch.Tensor)
            or torch.is_grad_enabled() or torch.is_autocast_enabled('cuda')
            or a.ndim!=2 or w.ndim!=2 or a.device.type!='cuda'
            or a.device!=w.device or a.dtype!=torch.bfloat16
            or w.dtype!=torch.bfloat16 or not a.is_contiguous()
            or not w.is_contiguous() or a.shape[1]!=5120
            or tuple(w.shape)!=(384,5120)
            or not (0<=a.shape[0]<=4 or (32<=a.shape[0]<=64 and a.shape[0]%2==0))):
        return False
    return (torch.version.hip is not None
            and torch.cuda.get_device_properties(a.device).gcnArchName.split(':')[0]=='gfx938')


def project(a,w):
    if not supported(a,w):
        raise ValueError('Unqualified BF16 FP32 router projection contract')
    m,k=a.shape;n=w.shape[0]
    y=torch.empty((m,n),device=a.device,dtype=torch.float32)
    if m:
        _router_stagger_pipelined_gate[(triton.cdiv(m,32),triton.cdiv(n,16))](
            a,w,y,m,n,k,32,16,512,128,num_warps=4,num_stages=3,
            matrix_instr_nonkdim=16)
    return y
