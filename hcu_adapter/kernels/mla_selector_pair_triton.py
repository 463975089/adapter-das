# SPDX-License-Identifier: Apache-2.0
"""Independent sparse selector banks in one grid; exact native sort/counts."""
import torch
import triton
import triton.language as tl
from .mla_selectors_triton import _selectors_fused

@triton.jit
def _selector_banks(V0,I0,L0,O0,C0,V1,I1,L1,O1,C1,
    R0:tl.constexpr,S0:tl.constexpr,IP0:tl.constexpr,LP0:tl.constexpr,
    W0:tl.constexpr,CAP0:tl.constexpr,SLOTS0:tl.constexpr,CSR0:tl.constexpr,
    B0:tl.constexpr,P0:tl.constexpr,
    R1:tl.constexpr,S1:tl.constexpr,IP1:tl.constexpr,LP1:tl.constexpr,
    W1:tl.constexpr,CAP1:tl.constexpr,SLOTS1:tl.constexpr,CSR1:tl.constexpr,
    B1:tl.constexpr,P1:tl.constexpr):
    if tl.program_id(1)==0:
        _selectors_fused(V0,I0,L0,O0,C0,R0,S0,IP0,LP0,W0,CAP0,SLOTS0,CSR0,B0,P0)
    else:
        _selectors_fused(V1,I1,L1,O1,C1,R1,S1,IP1,LP1,W1,CAP1,SLOTS1,CSR1,B1,P1)

def _arguments(bank,rows,slots):
    dense,lengths,flat,indptr=bank
    if flat is not None:
        width=flat.numel()//rows
        return (flat,indptr,flat),(0,flat.stride(0),indptr.stride(0),0,width,flat.numel(),slots,True)
    values=dense.reshape(rows,-1);width=values.shape[1]
    return (values,values,lengths),(*values.stride(),0,lengths.stride(0),width,0,slots,False)

def try_selector_banks(main,extra,rows,main_slots,extra_slots):
    """Return two (padded ids, counts) pairs, or None outside fusion bounds.

    Caller retains its selector validation. No device values are inspected here.
    """
    if rows<=0:return None
    source0=main[2] if main[2] is not None else main[0]
    source1=extra[2] if extra[2] is not None else extra[0]
    if source0 is None or source1 is None or not source0.is_cuda or source1.device!=source0.device:return None
    prop=torch.cuda.get_device_properties(source0.device)
    if getattr(prop,'gcnArchName','').split(':')[0]!='gfx938' or prop.multi_processor_count!=64:return None
    widths=(source0.numel()//rows,source1.numel()//rows)
    if any(not 0<w<=4096 for w in widths):return None
    padded=[max(64,triton.cdiv(w,64)*64) for w in widths]
    blocks=[triton.next_power_of_2(p) for p in padded]
    if rows*sum(blocks)>262144:return None
    pointers0,args0=_arguments(main,rows,main_slots)
    pointers1,args1=_arguments(extra,rows,extra_slots)
    outputs=[(torch.empty((rows,1,p),device=source0.device,dtype=torch.int32),
              torch.empty(rows,device=source0.device,dtype=torch.int32)) for p in padded]
    _selector_banks[(rows,2)](*pointers0,*outputs[0],*pointers1,*outputs[1],
        *args0,blocks[0],padded[0],*args1,blocks[1],padded[1],num_warps=4)
    return outputs
