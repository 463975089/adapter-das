# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Q/KV RMSNorm with the BF16 Q boundary before dynamic-token FP8.

The arithmetic contract is norm -> BF16 -> the selected existing quantizer.
The optional consumer must own admission, scale ABI and its native fallback.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _qkv_norm_token_fp8(X, QW, KW, Q, S, KV,
                       STRIDE: tl.constexpr, Q_SIZE: tl.constexpr,
                       KV_SIZE: tl.constexpr, EPS: tl.constexpr,
                       BLOCK: tl.constexpr, LIGHTOP: tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    task=tl.program_id(1)
    col=tl.arange(0,BLOCK)
    if task==0:
        size=Q_SIZE
        source=X+row*STRIDE
        weight=QW
    else:
        size=KV_SIZE
        source=X+row*STRIDE+Q_SIZE
        weight=KW
    gamma=tl.load(weight+col,col<size,other=0).to(tl.float32)
    value=tl.load(source+col,col<size,other=0).to(tl.float32)
    variance=tl.sum(value*value,axis=0)/size
    inverse=tl.rsqrt(variance+EPS)
    normalized=(value*inverse*gamma).to(tl.bfloat16)
    if task==0:
        bf=normalized.to(tl.float32)
        maximum=tl.max(tl.where(col<Q_SIZE,tl.abs(bf),0),axis=0)
        scale=tl.div_rn(maximum,448.0)
        if LIGHTOP:
            divided=bf*tl.div_rn(448.0,tl.where(maximum==0,1.0,maximum))
        else:
            scale=tl.maximum(scale,1.0/(448.0*512.0))
            divided=tl.div_rn(bf,scale)
        quant=tl.minimum(tl.maximum(divided,-448.0),448.0)
        if LIGHTOP:
            nan=tl.full((),0xffc00000,tl.uint32).to(tl.float32,bitcast=True)
            quant=tl.where(divided!=divided,nan,quant)
        tl.store(Q+row*Q_SIZE+col,quant,col<Q_SIZE)
        tl.store(S+row,scale)
    else:
        tl.store(KV+row*KV_SIZE+col,normalized,col<KV_SIZE)


def supported(x,q_weight,kv_weight,eps):
    if (not all(isinstance(t,torch.Tensor) for t in (x,q_weight,kv_weight))
            or torch.is_grad_enabled() or torch.is_autocast_enabled('cuda')
            or x.ndim!=2 or x.dtype!=torch.bfloat16 or x.device.type!='cuda'
            or x.stride(1)!=1 or x.stride(0)<x.shape[1]
            or q_weight.ndim!=1 or kv_weight.ndim!=1
            or q_weight.dtype not in (torch.bfloat16,torch.float32)
            or kv_weight.dtype not in (torch.bfloat16,torch.float32)
            or not q_weight.is_contiguous() or not kv_weight.is_contiguous()
            or q_weight.device!=x.device or kv_weight.device!=x.device
            or x.shape[1]!=q_weight.numel()+kv_weight.numel()
            or not 0<q_weight.numel()<=8192 or not 0<kv_weight.numel()<=8192
            or not isinstance(eps,(int,float)) or not math.isfinite(eps) or eps<=0):
        return False
    props=torch.cuda.get_device_properties(x.device)
    return (torch.version.hip is not None
            and props.gcnArchName.split(':')[0]=='gfx938')


def norm_quantize(x,q_weight,kv_weight,eps,*,lightop=False):
    if not supported(x,q_weight,kv_weight,eps):
        raise ValueError('Unqualified Q/KV BF16 RMSNorm token-FP8 contract')
    m=x.shape[0];q_size=q_weight.numel();kv_size=kv_weight.numel()
    if lightop and q_size%16:
        raise ValueError('LightOp token quantization requires vector-aligned Q')
    q=torch.empty((m,q_size),device=x.device,dtype=torch.float8_e4m3fn)
    scales=torch.empty((m,1),device=x.device,dtype=torch.float32)
    kv=torch.empty((m,kv_size),device=x.device,dtype=torch.bfloat16)
    block=triton.next_power_of_2(max(q_size,kv_size))
    if m:
        _qkv_norm_token_fp8[(m,2)](
            x,q_weight,kv_weight,q,scales,kv,x.stride(0),q_size,kv_size,
            float(eps),block,lightop,num_warps=8 if block>=2048 else 4)
    return q,scales,kv
